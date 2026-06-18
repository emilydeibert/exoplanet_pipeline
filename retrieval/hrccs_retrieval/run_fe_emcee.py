"""Run an Fe-only HRCCS retrieval with emcee.

This is an alternate sampler pathway for posterior exploration.  It reuses the
same pRT model builder, HRCCS likelihood, YAML priors, fixed/sampled Kp-Vsys
handling, and multiprocessing worker cache used by ``run_fe_sampler.py``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from retrieval.prt_emission_model import (
    derived_temperature_pressure_parameters,
    setup_logging,
    temperature_pressure_parameter_report,
)

from .data_loading import block_summary, load_hrccs_data, load_project_modules, parse_int_list, split_cli_list
from .model_builder import build_prt_xcorr_template, load_retrieval_config_and_parameters
from .sampler_common import (
    beta_configuration,
    beta_mode_label,
    fixed_parameters_from_config,
    init_sampler_worker,
    initialize_walkers,
    log_probability_from_state,
    multiprocessing_context,
    parameter_names,
    prior_bounds,
    read_worker_init_records,
    summarize_derived_parameters,
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
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel emcee likelihood workers. Use 1 for serial mode.")
    parser.add_argument("--fix-kp", type=float, default=None)
    parser.add_argument("--fix-vsys", type=float, default=None)
    parser.add_argument("--sample-kp-vsys", action="store_true", help="Also sample Kp and Vsys. Default keeps them fixed.")
    parser.add_argument(
        "--objective",
        choices=["matched_filter_loglike", "ccf_peak_value"],
        default="matched_filter_loglike",
    )

    parser.add_argument("--n-walkers", type=int, default=None, help="Number of emcee walkers. Default is max(32, 4*ndim).")
    parser.add_argument("--n-steps", type=int, default=1000, help="Number of emcee steps to run in this invocation.")
    parser.add_argument("--burn-in", type=int, default=0, help="Discard this many initial steps when flattening samples.")
    parser.add_argument("--thin", type=int, default=1, help="Thin post-burn-in samples by this factor.")
    parser.add_argument(
        "--initial-spread",
        type=float,
        default=1.0e-3,
        help="Gaussian initialization width as a fraction of each prior width.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from fe_hrccs_emcee_backend.h5 in the output directory.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing emcee backend when --resume is not supplied.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for walker initialization.")
    parser.add_argument("--progress", action="store_true", help="Show emcee progress bars.")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Write lightweight checkpoint metadata every N steps. The HDF backend is still updated every step.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, ndim: int) -> int:
    """Validate emcee-specific arguments and return the resolved walker count."""

    n_jobs = int(args.n_jobs)
    if n_jobs < 1:
        raise ValueError(f"--n-jobs must be >= 1; got {n_jobs}.")
    if int(args.n_steps) < 1:
        raise ValueError(f"--n-steps must be >= 1; got {args.n_steps}.")
    if int(args.burn_in) < 0:
        raise ValueError(f"--burn-in must be >= 0; got {args.burn_in}.")
    if int(args.thin) < 1:
        raise ValueError(f"--thin must be >= 1; got {args.thin}.")
    if int(args.checkpoint_every) < 0:
        raise ValueError(f"--checkpoint-every must be >= 0; got {args.checkpoint_every}.")
    if float(args.initial_spread) < 0:
        raise ValueError(f"--initial-spread must be >= 0; got {args.initial_spread}.")

    n_walkers = int(args.n_walkers) if args.n_walkers is not None else max(32, 4 * int(ndim))
    if n_walkers < 2 * int(ndim):
        raise ValueError(
            f"emcee needs at least 2*ndim walkers for the default moves; "
            f"got n_walkers={n_walkers}, ndim={ndim}."
        )
    return n_walkers


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with NumPy-safe values."""

    def default(value: Any) -> Any:
        try:
            import numpy as np

            if isinstance(value, np.integer):
                return int(value)
            if isinstance(value, np.floating):
                return float(value)
            if isinstance(value, np.ndarray):
                return value.tolist()
        except Exception:
            pass
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=default)


def summarize_flat_samples(
    samples: Any,
    names: list[str],
    bounds: list[tuple[float, float]],
    logger: Any,
    edge_fraction_width: float = 0.02,
    edge_warning_threshold: float = 0.20,
) -> dict[str, Any]:
    """Return percentile and prior-edge diagnostics for flat emcee samples."""

    import numpy as np

    samples = np.asarray(samples, dtype=float)
    summaries: dict[str, Any] = {}
    if samples.size == 0 or samples.shape[0] == 0:
        for name, _bound in zip(names, bounds):
            summaries[name] = {
                "median": None,
                "p16": None,
                "p84": None,
                "fraction_within_2pct_lower_prior": None,
                "fraction_within_2pct_upper_prior": None,
            }
        return summaries

    for idx, (name, (lo, hi)) in enumerate(zip(names, bounds)):
        p16, median, p84 = np.nanpercentile(samples[:, idx], [16.0, 50.0, 84.0])
        width = float(hi) - float(lo)
        edge_width = float(edge_fraction_width) * width
        finite = np.isfinite(samples[:, idx])
        if np.any(finite):
            lower_fraction = float(np.mean(samples[finite, idx] <= float(lo) + edge_width))
            upper_fraction = float(np.mean(samples[finite, idx] >= float(hi) - edge_width))
        else:
            lower_fraction = float("nan")
            upper_fraction = float("nan")
        summaries[name] = {
            "median": float(median),
            "p16": float(p16),
            "p84": float(p84),
            "fraction_within_2pct_lower_prior": lower_fraction,
            "fraction_within_2pct_upper_prior": upper_fraction,
        }
        if lower_fraction > edge_warning_threshold:
            logger.warning(
                "parameter %s has substantial posterior mass near lower prior edge: %.3f",
                name,
                lower_fraction,
            )
        if upper_fraction > edge_warning_threshold:
            logger.warning(
                "parameter %s has substantial posterior mass near upper prior edge: %.3f",
                name,
                upper_fraction,
            )
    return summaries


def best_parameters_from_chain(
    chain: Any,
    log_probability: Any,
    names: list[str],
    fixed_parameters: dict[str, float],
) -> tuple[dict[str, float], float | None]:
    """Return the highest-log-probability sample from the raw chain."""

    import numpy as np

    chain = np.asarray(chain, dtype=float)
    log_probability = np.asarray(log_probability, dtype=float)
    finite = np.isfinite(log_probability)
    if chain.size == 0 or not np.any(finite):
        return dict(fixed_parameters), None

    flat_chain = chain.reshape((-1, chain.shape[-1]))
    flat_logp = log_probability.reshape(-1)
    best_index = int(np.nanargmax(flat_logp))
    best = {name: float(value) for name, value in zip(names, flat_chain[best_index])}
    best.update({key: float(value) for key, value in fixed_parameters.items()})
    return best, float(flat_logp[best_index])


def save_trace_plot(path: Path, chain: Any, names: list[str]) -> bool:
    """Save a simple walker trace plot."""

    import numpy as np

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    chain = np.asarray(chain, dtype=float)
    if chain.size == 0:
        return False
    ndim = len(names)
    fig, axes = plt.subplots(ndim, 1, figsize=(10, max(2.0, 1.7 * ndim)), sharex=True)
    if ndim == 1:
        axes = [axes]
    alpha = min(0.35, max(0.03, 6.0 / max(chain.shape[1], 1)))
    for idx, axis in enumerate(axes):
        axis.plot(chain[:, :, idx], color="black", alpha=alpha, linewidth=0.5)
        axis.set_ylabel(names[idx])
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def save_corner_plot(path: Path, samples: Any, names: list[str], logger: Any) -> bool:
    """Save a corner plot when enough samples and dependencies are available."""

    import numpy as np

    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] < max(5, samples.shape[1] + 1):
        logger.warning("Too few post-burn-in samples for a useful corner plot; skipping.")
        return False

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import corner
    except ImportError as exc:
        logger.warning("corner/matplotlib is unavailable; skipping corner plot: %s", exc)
        return False

    try:
        fig = corner.corner(samples, labels=names, show_titles=True)
        fig.savefig(path, dpi=250, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as exc:
        logger.warning("corner plot failed; continuing without it: %s", exc)
        return False


def write_checkpoint_summary(
    path: Path,
    backend: Any,
    completed_steps_this_run: int,
    start_iteration: int,
    walltime_seconds: float,
) -> None:
    """Write lightweight progress metadata during long emcee runs."""

    write_json(
        path,
        {
            "backend_iteration": int(backend.iteration),
            "start_iteration": int(start_iteration),
            "completed_steps_this_run": int(completed_steps_this_run),
            "walltime_seconds_so_far": float(walltime_seconds),
        },
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir, "fe_hrccs_emcee.log")

    try:
        import emcee
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise RuntimeError(
            "emcee and numpy are required for the HRCCS emcee sampler. "
            "Install emcee in the retrieval environment."
        ) from exc

    n_jobs = int(args.n_jobs)
    exopipe_config, exopipe_params = load_project_modules(args.project_path)
    retrieval_config, initial = load_retrieval_config_and_parameters(args.retrieval_config)

    fixed_kp = initial["Kp"] if args.fix_kp is None else float(args.fix_kp)
    fixed_vsys = initial["Vsys"] if args.fix_vsys is None else float(args.fix_vsys)
    initial["Kp"] = float(fixed_kp)
    initial["Vsys"] = float(fixed_vsys)

    names = parameter_names(args.sample_kp_vsys, retrieval_config)
    beta_config = beta_configuration(names, retrieval_config, args.objective)
    yaml_fixed_parameters = fixed_parameters_from_config(retrieval_config)
    bounds = prior_bounds(retrieval_config, names)
    ndim = len(names)
    n_walkers = validate_args(args, ndim=ndim)

    backend_path = output_dir / "fe_hrccs_emcee_backend.h5"
    if args.resume:
        if not backend_path.exists():
            raise FileNotFoundError(f"--resume was supplied, but no emcee backend exists at {backend_path}.")
    elif backend_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Emcee backend already exists: {backend_path}. "
                "Use --resume to continue it or --overwrite to replace it."
            )
        backend_path.unlink()

    logger.info("Sampling parameters: %s", names)
    logger.info("Prior bounds: %s", dict(zip(names, bounds)))
    logger.info("Fixed Kp=%.3f Vsys=%.3f unless sampled", fixed_kp, fixed_vsys)
    logger.info("beta mode: %s", beta_mode_label(beta_config))
    logger.info("emcee execution mode: %s with n_jobs=%d", "serial" if n_jobs == 1 else "parallel", n_jobs)
    initial_tp_report = temperature_pressure_parameter_report(
        {**initial, **yaml_fixed_parameters},
        retrieval_config,
    )
    logger.info(
        "Initial T-P pressure grid log10(bar): %.6g to %.6g",
        initial_tp_report["pressure_grid_log10_min"],
        initial_tp_report["pressure_grid_log10_max"],
    )
    logger.info("Initial sampled T-P parameters: %s", initial_tp_report["sampled"])
    if initial_tp_report["derived"]:
        logger.info("Initial derived T-P parameters: %s", initial_tp_report["derived"])

    initial_model_parameters = {**initial, **yaml_fixed_parameters}
    initial_template = build_prt_xcorr_template(
        retrieval_config,
        exopipe_config,
        initial_model_parameters,
        logger=logger,
    )
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

    worker_init_log_path = output_dir / "fe_hrccs_emcee_worker_initialization.jsonl"
    worker_init_log_path.write_text("", encoding="utf-8")
    sampler_state = {
        "retrieval_config": retrieval_config,
        # Only ghost_res is needed by the template builder. Avoid placing the
        # imported project config module itself in multiprocessing state.
        "exopipe_config": SimpleNamespace(ghost_res=float(exopipe_config.ghost_res)),
        "blocks": blocks,
        "initial": initial,
        "fixed_parameters": yaml_fixed_parameters,
        "names": names,
        "bounds": bounds,
        "fixed_kp": float(fixed_kp),
        "fixed_vsys": float(fixed_vsys),
        "sample_kp_vsys": bool(args.sample_kp_vsys),
        "objective": args.objective,
        "beta_config": beta_config,
        "cache_prt_atmosphere": True,
        "worker_init_log_path": str(worker_init_log_path),
    }
    init_sampler_worker(
        {
            **sampler_state,
            "logger": logger,
            # Serial emcee benefits from the same one-atmosphere cache.  In
            # parallel mode the real cache is initialized inside each worker.
            "cache_prt_atmosphere": n_jobs == 1,
        }
    )

    backend = emcee.backends.HDFBackend(str(backend_path))
    start_iteration = int(backend.iteration) if args.resume else 0

    if args.resume:
        if int(backend.iteration) <= 0:
            raise RuntimeError(f"Backend exists but has no saved iterations: {backend_path}")
        existing_chain = backend.get_chain()
        if existing_chain.shape[1] != n_walkers or existing_chain.shape[2] != ndim:
            raise ValueError(
                "Existing backend shape does not match requested run: "
                f"backend n_walkers={existing_chain.shape[1]}, ndim={existing_chain.shape[2]}; "
                f"requested n_walkers={n_walkers}, ndim={ndim}."
            )
        initial_state = backend.get_last_sample()
        logger.info("Resuming emcee backend at iteration %d", int(backend.iteration))
    else:
        backend.reset(n_walkers, ndim)
        initial_state = initialize_walkers(
            initial=initial,
            names=names,
            bounds=bounds,
            n_walkers=n_walkers,
            initial_spread=float(args.initial_spread),
            seed=args.seed,
            retrieval_config=retrieval_config,
            fixed_parameters=yaml_fixed_parameters,
        )
        for idx, name in enumerate(names):
            logger.info(
                "Initial walker range %s: %.6g to %.6g",
                name,
                float(np.nanmin(initial_state[:, idx])),
                float(np.nanmax(initial_state[:, idx])),
            )

    serial_calls = 0

    def serial_log_probability(theta: Any) -> float:
        nonlocal serial_calls
        serial_calls += 1
        return log_probability_from_state(theta)

    sampler_start = time.perf_counter()
    multiprocessing_start_method = "serial"
    pool = None
    log_probability = serial_log_probability
    if n_jobs > 1:
        ctx = multiprocessing_context()
        multiprocessing_start_method = ctx.get_start_method()
        logger.info(
            "Starting emcee multiprocessing pool with %d workers, start_method=%s",
            n_jobs,
            multiprocessing_start_method,
        )
        pool = ctx.Pool(
            processes=n_jobs,
            initializer=init_sampler_worker,
            initargs=(sampler_state,),
        )
        log_probability = log_probability_from_state

    completed_cleanly = False
    try:
        sampler = emcee.EnsembleSampler(
            n_walkers,
            ndim,
            log_probability,
            pool=pool,
            backend=backend,
        )

        completed = 0
        state = initial_state
        remaining = int(args.n_steps)
        while completed < remaining:
            chunk = remaining - completed
            if int(args.checkpoint_every) > 0:
                chunk = min(chunk, int(args.checkpoint_every))
            state = sampler.run_mcmc(state, chunk, progress=bool(args.progress))
            completed += chunk
            if int(args.checkpoint_every) > 0:
                write_checkpoint_summary(
                    output_dir / "fe_hrccs_emcee_checkpoint_summary.json",
                    backend=backend,
                    completed_steps_this_run=completed,
                    start_iteration=start_iteration,
                    walltime_seconds=float(time.perf_counter() - sampler_start),
                )
                logger.info(
                    "emcee checkpoint: completed %d/%d steps this run, backend iteration=%d",
                    completed,
                    remaining,
                    int(backend.iteration),
                )
        completed_cleanly = True
    finally:
        if pool is not None:
            if completed_cleanly:
                pool.close()
            else:
                pool.terminate()
            pool.join()

    walltime_seconds = float(time.perf_counter() - sampler_start)
    end_iteration = int(backend.iteration)
    steps_completed_this_run = int(end_iteration - start_iteration)

    chain = backend.get_chain()
    log_probability_chain = backend.get_log_prob()
    flat_samples = backend.get_chain(discard=int(args.burn_in), thin=int(args.thin), flat=True)
    flat_log_probability = backend.get_log_prob(discard=int(args.burn_in), thin=int(args.thin), flat=True)

    fixed_parameters = dict(yaml_fixed_parameters)
    if not args.sample_kp_vsys:
        fixed_parameters.update({"Kp": float(fixed_kp), "Vsys": float(fixed_vsys)})
    best_parameters, best_log_probability = best_parameters_from_chain(
        chain=chain,
        log_probability=log_probability_chain,
        names=names,
        fixed_parameters=fixed_parameters,
    )
    try:
        best_derived_parameters = derived_temperature_pressure_parameters(best_parameters, retrieval_config)
    except Exception as exc:
        best_derived_parameters = {"error": str(exc)}
        logger.warning("Could not derive best-fit T-P parameters: %s", exc)

    acceptance_fraction = np.asarray(sampler.acceptance_fraction, dtype=float)
    try:
        autocorr_time = np.asarray(sampler.get_autocorr_time(tol=0), dtype=float)
        autocorr_error = None
    except Exception as exc:
        autocorr_time = None
        autocorr_error = str(exc)
        logger.warning("Autocorrelation time estimate failed: %s", exc)

    if n_jobs == 1:
        n_likelihood_calls = int(serial_calls)
        n_likelihood_calls_is_estimated = False
    else:
        # emcee evaluates the initial walker log probabilities on a fresh run,
        # then one likelihood per walker per sampled step.  On resume it starts
        # from the backend's last saved state, so the extra initial call is not
        # expected.
        extra_initial = 0 if args.resume else 1
        n_likelihood_calls = int(n_walkers * (steps_completed_this_run + extra_initial))
        n_likelihood_calls_is_estimated = True
    calls_per_second = float(n_likelihood_calls / walltime_seconds) if walltime_seconds > 0 else 0.0
    average_seconds_per_call = float(walltime_seconds / n_likelihood_calls) if n_likelihood_calls > 0 else None

    chain_path = output_dir / "fe_hrccs_emcee_chain.npz"
    samples_path = output_dir / "fe_hrccs_emcee_samples.npz"
    summary_path = output_dir / "fe_hrccs_emcee_summary.json"
    corner_path = output_dir / "fe_hrccs_emcee_corner.png"
    trace_path = output_dir / "fe_hrccs_emcee_trace.png"

    np.savez_compressed(
        chain_path,
        chain=chain,
        log_probability=log_probability_chain,
        parameter_names=np.asarray(names, dtype="U64"),
    )
    np.savez_compressed(
        samples_path,
        samples=flat_samples,
        log_probability=flat_log_probability,
        parameter_names=np.asarray(names, dtype="U64"),
    )

    trace_saved = save_trace_plot(trace_path, chain=chain, names=names)
    corner_saved = save_corner_plot(corner_path, samples=flat_samples, names=names, logger=logger)

    output_files = {
        "log": str(output_dir / "fe_hrccs_emcee.log"),
        "backend": str(backend_path),
        "chain": str(chain_path),
        "samples": str(samples_path),
        "summary": str(summary_path),
        "trace": str(trace_path) if trace_saved else None,
        "corner": str(corner_path) if corner_saved else None,
        "worker_initialization_log": str(worker_init_log_path),
    }

    derived_parameter_summaries = summarize_derived_parameters(
        samples=flat_samples,
        names=names,
        initial=initial,
        fixed_parameters=fixed_parameters,
        retrieval_config=retrieval_config,
    )
    summary = {
        "sampler_type": "emcee",
        "project_path": str(args.project_path),
        "retrieval_config": str(args.retrieval_config),
        "sysrem_iteration": int(args.k),
        "objective": args.objective,
        "parameter_names": names,
        "prior_bounds": {name: [float(lo), float(hi)] for name, (lo, hi) in zip(names, bounds)},
        "fixed_parameters": fixed_parameters,
        "beta_mode": beta_config,
        "sample_kp_vsys": bool(args.sample_kp_vsys),
        "n_walkers": int(n_walkers),
        "n_steps": int(args.n_steps),
        "start_iteration": int(start_iteration),
        "end_iteration": int(end_iteration),
        "steps_completed_this_run": int(steps_completed_this_run),
        "burn_in": int(args.burn_in),
        "thin": int(args.thin),
        "initial_spread": float(args.initial_spread),
        "resume": bool(args.resume),
        "overwrite": bool(args.overwrite),
        "n_jobs": int(n_jobs),
        "parallel_mode": "serial" if n_jobs == 1 else "multiprocessing",
        "multiprocessing_start_method": multiprocessing_start_method,
        "seed": args.seed,
        "walltime_seconds": float(walltime_seconds),
        "n_likelihood_calls": int(n_likelihood_calls),
        "n_likelihood_calls_is_estimated": bool(n_likelihood_calls_is_estimated),
        "calls_per_second": float(calls_per_second),
        "average_seconds_per_call": average_seconds_per_call,
        "acceptance_fraction": acceptance_fraction,
        "median_acceptance_fraction": float(np.nanmedian(acceptance_fraction)),
        "autocorrelation_time": autocorr_time,
        "autocorrelation_time_error": autocorr_error,
        "parameter_summaries": summarize_flat_samples(flat_samples, names, bounds, logger),
        "derived_parameter_summaries": derived_parameter_summaries,
        "best_parameters": best_parameters,
        "derived_best_parameters": best_derived_parameters,
        "initial_temperature_pressure_report": initial_tp_report,
        "best_log_probability": best_log_probability,
        "backend_file": str(backend_path),
        "worker_initialization_log": str(worker_init_log_path),
        "worker_initialization": read_worker_init_records(worker_init_log_path),
        "data": block_summary(blocks),
        "output_files": output_files,
    }
    write_json(summary_path, summary)

    logger.info(
        "emcee finished in %.2fs with n_likelihood_calls=%d, calls_per_second=%.4f",
        walltime_seconds,
        n_likelihood_calls,
        calls_per_second,
    )
    logger.info("Best parameters: %s", best_parameters)
    if best_derived_parameters:
        logger.info("Best derived T-P parameters: %s", best_derived_parameters)
    logger.info("Saved HRCCS emcee outputs to %s", output_dir)


if __name__ == "__main__":
    main()
