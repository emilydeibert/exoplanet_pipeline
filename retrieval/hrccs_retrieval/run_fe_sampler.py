"""Run a first Fe-only dynesty sampler with the HRCCS matched-filter objective."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from retrieval.prt_emission_model import setup_logging

from .ccf_likelihood import evaluate_objective
from .data_loading import block_summary, load_hrccs_data, load_project_modules, parse_int_list, split_cli_list
from .model_builder import build_prt_xcorr_template, load_retrieval_config_and_parameters, parameters_with_updates


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
    parser.add_argument("--fix-kp", type=float, default=None)
    parser.add_argument("--fix-vsys", type=float, default=None)
    parser.add_argument("--sample-kp-vsys", action="store_true", help="Also sample Kp and Vsys. Default keeps them fixed.")
    parser.add_argument("--resume", action="store_true", help="Placeholder flag; current implementation starts a fresh dynesty run.")
    parser.add_argument("--test", action="store_true", help="Use a small nlive/maxcall for a fast plumbing test.")
    parser.add_argument(
        "--objective",
        choices=["matched_filter_loglike", "ccf_peak_value"],
        default="matched_filter_loglike",
    )
    return parser.parse_args()


def parameter_names(args: argparse.Namespace) -> list[str]:
    names = ["T_deep", "delta_T_inv", "log10_Fe"]
    if args.sample_kp_vsys:
        names = ["Kp", "Vsys"] + names
    return names


def prior_bounds(retrieval_config: dict[str, Any], names: list[str]) -> list[tuple[float, float]]:
    priors = retrieval_config.get("priors", {})
    bounds = []
    for name in names:
        if name not in priors:
            raise ValueError(f"Missing prior bounds for sampled parameter {name!r}.")
        lo, hi = priors[name]
        bounds.append((float(lo), float(hi)))
    return bounds


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir, "fe_hrccs_sampler.log")
    if args.resume:
        logger.warning("--resume was supplied, but checkpoint resume is not implemented yet; starting fresh.")

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

    names = parameter_names(args)
    bounds = prior_bounds(retrieval_config, names)
    logger.info("Sampling parameters: %s", names)
    logger.info("Fixed Kp=%.3f Vsys=%.3f unless sampled", fixed_kp, fixed_vsys)

    n_calls = 0

    def prior_transform(unit_cube: Any) -> Any:
        theta = np.empty(len(names), dtype=float)
        for idx, (lo, hi) in enumerate(bounds):
            theta[idx] = lo + (hi - lo) * unit_cube[idx]
        return theta

    def log_likelihood(theta: Any) -> float:
        nonlocal n_calls
        n_calls += 1
        updates = {name: float(value) for name, value in zip(names, theta)}
        parameters = parameters_with_updates(initial, updates)
        if not args.sample_kp_vsys:
            parameters["Kp"] = float(fixed_kp)
            parameters["Vsys"] = float(fixed_vsys)
        start = time.perf_counter()
        try:
            template = build_prt_xcorr_template(retrieval_config, exopipe_config, parameters, logger=None)
            result = evaluate_objective(
                blocks=blocks,
                F_model=template["F_model"],
                Kp=parameters["Kp"],
                Vsys=parameters["Vsys"],
                objective=args.objective,
            )
            value = float(result["objective_value"])
        except Exception as exc:
            logger.warning("Sampler model evaluation failed at call %d: %s", n_calls, exc)
            value = -np.inf
        if n_calls % 10 == 0:
            logger.info("sampler call %d loglike=%.6e seconds=%.2f", n_calls, value, time.perf_counter() - start)
        return value

    sampler = dynesty.NestedSampler(
        log_likelihood,
        prior_transform,
        ndim=len(names),
        nlive=int(args.nlive),
        bound="multi",
        sample="rwalk",
    )
    sampler.run_nested(maxcall=args.maxcall)
    results = sampler.results

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
        "n_calls": int(n_calls),
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
