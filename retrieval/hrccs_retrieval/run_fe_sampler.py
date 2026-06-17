"""Run a first Fe-only dynesty sampler with the HRCCS matched-filter objective."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from retrieval.prt_emission_model import setup_logging

from .data_loading import block_summary, load_hrccs_data, load_project_modules, parse_int_list, split_cli_list
from .model_builder import build_prt_xcorr_template, load_retrieval_config_and_parameters
from .sampler_common import (
    draw_benchmark_thetas,
    dynesty_call_count,
    init_sampler_worker,
    log_likelihood_from_state,
    multiprocessing_context,
    parameter_names,
    prior_bounds,
    prior_transform_from_state,
    read_worker_init_records,
    validate_beta_configuration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_path")
    parser.add_argument("--retrieval-config", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--nights", nargs="+", default=None)
    parser.add_argument("--cameras", nargs="+", default=None)
    parser.add_argument("--orders", nargs="+", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--nlive", type=int, default=100)
    parser.add_argument("--maxcall", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel dynesty likelihood workers. Use 1 for serial mode.")
    parser.add_argument("--fix-kp", type=float, default=None)
    parser.add_argument("--fix-vsys", type=float, default=None)
    parser.add_argument("--sample-kp-vsys", action="store_true", help="Also sample Kp and Vsys. Default keeps them fixed.")
    parser.add_argument("--resume", action="store_true", help="Placeholder flag; current implementation starts a fresh dynesty run.")
    parser.add_argument("--test", action="store_true", help="Use a small nlive/maxcall for a fast plumbing test.")
    parser.add_argument(
        "--benchmark-likelihood-calls",
        type=int,
        default=None,
        help="Evaluate N representative likelihood calls with the same pool/cache setup, then exit before dynesty.",
    )
    parser.add_argument(
        "--objective",
        choices=["matched_filter_loglike", "ccf_peak_value"],
        default="matched_filter_loglike",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir, "fe_hrccs_sampler.log")
    if args.resume:
        logger.warning("--resume was supplied, but checkpoint resume is not implemented yet; starting fresh.")
    n_jobs = int(args.n_jobs)
    if n_jobs < 1:
        raise ValueError(f"--n-jobs must be >= 1; got {n_jobs}.")

    try:
        import dynesty
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise RuntimeError("dynesty and numpy are required for the HRCCS sampler.") from exc

    exopipe_config, exopipe_params = load_project_modules(args.project_path)
    retrieval_config, initial = load_retrieval_config_and_parameters(args.retrieval_config)

    fixed_kp = initial["Kp"] if args.fix_kp is None else float(args.fix_kp)
    fixed_vsys = initial["Vsys"] if args.fix_vsys is None else float(args.fix_vsys)
    initial["Kp"] = float(fixed_kp)
    initial["Vsys"] = float(fixed_vsys)

    if args.test:
        args.nlive = min(int(args.nlive), 25)
        args.maxcall = 200 if args.maxcall is None else min(int(args.maxcall), 200)

    initial_template = build_prt_xcorr_template(retrieval_config, exopipe_config, initial, logger=logger)
    blocks = load_hrccs_data(
        config=exopipe_config,
        params=exopipe_params,
        k=args.k,
        nights=split_cli_list(args.nights),
        cameras=split_cli_list(args.cameras),
        model_array=initial_template["model_array"],
        orders=parse_int_list(args.orders),
        logger=logger,
    )

    names = parameter_names(args.sample_kp_vsys, retrieval_config)
    validate_beta_configuration(names, args.objective)
    bounds = prior_bounds(retrieval_config, names)
    logger.info("Sampling parameters: %s", names)
    logger.info("Fixed Kp=%.3f Vsys=%.3f unless sampled", fixed_kp, fixed_vsys)
    logger.info("Dynesty execution mode: %s with n_jobs=%d", "serial" if n_jobs == 1 else "parallel", n_jobs)

    worker_init_log_path = output_dir / "fe_hrccs_worker_initialization.jsonl"
    worker_init_log_path.write_text("", encoding="utf-8")
    sampler_state = {
        "retrieval_config": retrieval_config,
        # Only ghost_res is needed by the template builder.  Avoid putting the
        # imported project config module itself into multiprocessing state.
        "exopipe_config": SimpleNamespace(ghost_res=float(exopipe_config.ghost_res)),
        "blocks": blocks,
        "initial": initial,
        "names": names,
        "bounds": bounds,
        "fixed_kp": float(fixed_kp),
        "fixed_vsys": float(fixed_vsys),
        "sample_kp_vsys": bool(args.sample_kp_vsys),
        "objective": args.objective,
        "cache_prt_atmosphere": n_jobs > 1,
        "worker_init_log_path": str(worker_init_log_path),
    }
    init_sampler_worker({**sampler_state, "logger": logger, "cache_prt_atmosphere": False})

    serial_calls = 0

    def serial_log_likelihood(theta: Any) -> float:
        nonlocal serial_calls
        serial_calls += 1
        start = time.perf_counter()
        value = log_likelihood_from_state(theta)
        if serial_calls % 10 == 0:
            logger.info("sampler call %d loglike=%.6e seconds=%.2f", serial_calls, value, time.perf_counter() - start)
        return value

    def run_sampler_with_callables(loglike: Any, prior: Any, pool: Any = None, queue_size: int | None = None) -> Any:
        kwargs = {}
        if pool is not None:
            kwargs["pool"] = pool
            kwargs["queue_size"] = int(queue_size or n_jobs)
            kwargs["use_pool"] = {
                "loglikelihood": True,
                "prior_transform": False,
                "propose_point": False,
                "update_bound": False,
            }
        sampler = dynesty.NestedSampler(
            loglike,
            prior,
            ndim=len(names),
            nlive=int(args.nlive),
            bound="multi",
            sample="rwalk",
            **kwargs,
        )
        sampler.run_nested(maxcall=args.maxcall)
        return sampler

    use_pool_settings = None
    if n_jobs > 1:
        use_pool_settings = {
            "loglikelihood": True,
            "prior_transform": False,
            "propose_point": False,
            "update_bound": False,
        }

    if args.benchmark_likelihood_calls is not None:
        benchmark_calls = int(args.benchmark_likelihood_calls)
        if benchmark_calls < 1:
            raise ValueError("--benchmark-likelihood-calls must be >= 1 when supplied.")

        theta_points = draw_benchmark_thetas(benchmark_calls, bounds)
        benchmark_start = time.perf_counter()
        if n_jobs == 1:
            values = [serial_log_likelihood(theta) for theta in theta_points]
            multiprocessing_start_method = "serial"
            queue_size = 1
        else:
            ctx = multiprocessing_context()
            multiprocessing_start_method = ctx.get_start_method()
            queue_size = n_jobs
            logger.info(
                "Starting likelihood benchmark pool with %d workers, start_method=%s",
                n_jobs,
                multiprocessing_start_method,
            )
            with ctx.Pool(
                processes=n_jobs,
                initializer=init_sampler_worker,
                initargs=(sampler_state,),
            ) as pool:
                values = pool.map(log_likelihood_from_state, list(theta_points), chunksize=1)

        walltime_seconds = float(time.perf_counter() - benchmark_start)
        n_calls = int(len(values))
        calls_per_second = float(n_calls / walltime_seconds) if walltime_seconds > 0 else 0.0
        average_seconds_per_call = float(walltime_seconds / n_calls) if n_calls > 0 else None

        np.savez_compressed(
            output_dir / "fe_hrccs_likelihood_benchmark.npz",
            theta=theta_points,
            log_likelihood=np.asarray(values, dtype=float),
            parameter_names=np.asarray(names, dtype="U64"),
        )
        summary = {
            "project_path": str(args.project_path),
            "retrieval_config": str(args.retrieval_config),
            "sysrem_iteration": int(args.k),
            "objective": args.objective,
            "parameter_names": names,
            "benchmark_likelihood_calls": benchmark_calls,
            "n_jobs": int(n_jobs),
            "parallel_mode": "serial" if n_jobs == 1 else "multiprocessing",
            "multiprocessing_start_method": multiprocessing_start_method,
            "queue_size": int(queue_size),
            "use_pool": use_pool_settings,
            "prt_atmosphere_cached_per_worker": bool(n_jobs > 1),
            "worker_initialization_log": str(worker_init_log_path),
            "worker_initialization": read_worker_init_records(worker_init_log_path),
            "walltime_seconds": walltime_seconds,
            "n_calls": n_calls,
            "calls_per_second": calls_per_second,
            "average_seconds_per_call": average_seconds_per_call,
            "data": block_summary(blocks),
        }
        with (output_dir / "fe_hrccs_likelihood_benchmark_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        logger.info(
            "Likelihood benchmark finished in %.2fs with n_calls=%d, calls_per_second=%.4f",
            walltime_seconds,
            n_calls,
            calls_per_second,
        )
        logger.info("Saved likelihood benchmark outputs to %s", output_dir)
        return

    sampler_start = time.perf_counter()
    multiprocessing_start_method = "serial"
    queue_size = 1
    if n_jobs == 1:
        sampler = run_sampler_with_callables(serial_log_likelihood, prior_transform_from_state)
    else:
        ctx = multiprocessing_context()
        multiprocessing_start_method = ctx.get_start_method()
        queue_size = n_jobs
        logger.info(
            "Starting dynesty multiprocessing pool with %d workers, start_method=%s, queue_size=%d",
            n_jobs,
            multiprocessing_start_method,
            queue_size,
        )
        with ctx.Pool(
            processes=n_jobs,
            initializer=init_sampler_worker,
            initargs=(sampler_state,),
        ) as pool:
            sampler = run_sampler_with_callables(
                log_likelihood_from_state,
                prior_transform_from_state,
                pool=pool,
                queue_size=queue_size,
            )
    walltime_seconds = float(time.perf_counter() - sampler_start)
    results = sampler.results

    n_calls = dynesty_call_count(results, fallback=serial_calls)
    calls_per_second = float(n_calls / walltime_seconds) if walltime_seconds > 0 else 0.0
    average_seconds_per_call = float(walltime_seconds / n_calls) if n_calls > 0 else None

    logger.info(
        "Dynesty finished in %.2fs with n_calls=%d, calls_per_second=%.4f",
        walltime_seconds,
        n_calls,
        calls_per_second,
    )

    samples = np.asarray(results.samples)
    logl = np.asarray(results.logl)
    if hasattr(results, "logwt") and hasattr(results, "logz"):
        weights = np.exp(np.asarray(results.logwt) - float(results.logz[-1]))
        weights = weights / np.sum(weights)
        evidence = float(results.logz[-1])
    else:
        weights = np.ones(samples.shape[0]) / samples.shape[0]
        evidence = None

    np.savez_compressed(
        output_dir / "fe_hrccs_dynesty_samples.npz",
        samples=samples,
        weights=weights,
        log_likelihood=logl,
        parameter_names=np.asarray(names, dtype="U64"),
    )

    best_index = int(np.nanargmax(logl))
    best = {name: float(value) for name, value in zip(names, samples[best_index])}
    if not args.sample_kp_vsys:
        best["Kp"] = float(fixed_kp)
        best["Vsys"] = float(fixed_vsys)

    summary = {
        "project_path": str(args.project_path),
        "retrieval_config": str(args.retrieval_config),
        "sysrem_iteration": int(args.k),
        "objective": args.objective,
        "parameter_names": names,
        "best_fit_parameters": best,
        "best_log_likelihood": float(logl[best_index]),
        "log_evidence": evidence,
        "nlive": int(args.nlive),
        "maxcall": args.maxcall,
        "n_jobs": int(n_jobs),
        "parallel_mode": "serial" if n_jobs == 1 else "multiprocessing",
        "multiprocessing_start_method": multiprocessing_start_method,
        "queue_size": int(queue_size),
        "use_pool": use_pool_settings,
        "prt_atmosphere_cached_per_worker": bool(n_jobs > 1),
        "worker_initialization_log": str(worker_init_log_path),
        "worker_initialization": read_worker_init_records(worker_init_log_path),
        "walltime_seconds": walltime_seconds,
        "n_calls": int(n_calls),
        "calls_per_second": calls_per_second,
        "average_seconds_per_call": average_seconds_per_call,
        "data": block_summary(blocks),
    }
    with (output_dir / "fe_hrccs_dynesty_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    try:
        import corner
        import matplotlib.pyplot as plt

        fig = corner.corner(samples, weights=weights, labels=names, show_titles=True)
        fig.savefig(output_dir / "fe_hrccs_corner.png", dpi=250, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        logger.warning("corner is not installed; skipping corner plot.")

    logger.info("Sampler complete. Best parameters: %s", best)
    logger.info("Saved HRCCS sampler outputs to %s", output_dir)


if __name__ == "__main__":
    main()
