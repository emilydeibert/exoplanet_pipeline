"""Diagnose HRCCS likelihood as a Kp/Vsys grid for fixed atmosphere."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from retrieval.prt_emission_model import setup_logging

from .ccf_likelihood import OBJECTIVE_CHOICES, evaluate_objective
from .data_loading import (
    block_summary,
    load_hrccs_data,
    load_project_modules,
    log_model_data_wavelength_padding,
    parse_int_list,
    split_cli_list,
)
from .model_builder import build_prt_xcorr_template, load_retrieval_config_and_parameters, parameters_with_updates
from .sampler_common import (
    alpha_configuration,
    alpha_from_state,
    alpha_mode_label,
    beta_configuration,
    beta_from_state,
    beta_mode_label,
    fixed_parameters_from_config,
    parameter_names,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_path")
    parser.add_argument("--retrieval-config", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--nights", nargs="+", default=None)
    parser.add_argument("--cameras", nargs="+", default=None)
    parser.add_argument("--orders", nargs="+", default=None)
    parser.add_argument("--parameters-json", required=True, help="Plain parameter JSON or emcee/dynesty summary JSON.")
    parser.add_argument("--kp-min", type=float, required=True)
    parser.add_argument("--kp-max", type=float, required=True)
    parser.add_argument("--kp-step", type=float, required=True)
    parser.add_argument("--vsys-min", type=float, required=True)
    parser.add_argument("--vsys-max", type=float, required=True)
    parser.add_argument("--vsys-step", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--objective",
        choices=OBJECTIVE_CHOICES,
        default="matched_filter_loglike",
    )
    return parser.parse_args()


def load_parameter_values(path: str | Path) -> dict[str, float]:
    """Load fixed parameter values from a plain JSON or previous summary."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    candidates: Any
    if isinstance(payload, dict) and "best_parameters" in payload:
        candidates = payload["best_parameters"]
    elif isinstance(payload, dict) and "best_fit_parameters" in payload:
        candidates = payload["best_fit_parameters"]
    elif isinstance(payload, dict) and "parameters" in payload:
        candidates = payload["parameters"]
    else:
        candidates = payload

    if not isinstance(candidates, dict):
        raise ValueError(f"Could not find a parameter mapping in {path}.")
    return {
        str(key): float(value)
        for key, value in candidates.items()
        if isinstance(value, (int, float))
    }


def inclusive_grid(lo: float, hi: float, step: float) -> Any:
    """Return an inclusive regular grid."""

    import numpy as np

    lo = float(lo)
    hi = float(hi)
    step = float(step)
    if step <= 0 or hi < lo:
        raise ValueError(f"Invalid grid bounds: lo={lo}, hi={hi}, step={step}.")
    return np.arange(lo, hi + 0.5 * step, step, dtype=float)


def save_grid_plot(path: Path, kp_grid: Any, vsys_grid: Any, loglike_map: Any, best: dict[str, float], objective: str) -> bool:
    """Save delta-loglike Kp/Vsys map."""

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    delta = loglike_map - np.nanmax(loglike_map)
    fig, ax = plt.subplots(figsize=(7, 5))
    mesh = ax.pcolormesh(vsys_grid, kp_grid, delta, shading="auto")
    ax.scatter([best["Vsys"]], [best["Kp"]], marker="x", color="white", s=80)
    ax.set_xlabel("Vsys [km/s]")
    ax.set_ylabel("Kp [km/s]")
    ax.set_title(str(objective))
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Delta objective")
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir, "diagnose_kp_vsys_likelihood_grid.log")

    import numpy as np

    exopipe_config, exopipe_params = load_project_modules(args.project_path)
    retrieval_config, initial = load_retrieval_config_and_parameters(args.retrieval_config)
    supplied = load_parameter_values(args.parameters_json)
    fixed_parameters = fixed_parameters_from_config(retrieval_config)
    parameters = parameters_with_updates(initial, supplied)
    parameters.update(fixed_parameters)
    base_parameters = dict(parameters)

    names = parameter_names(sample_kp_vsys=True, retrieval_config=retrieval_config)
    alpha_config = alpha_configuration(names, retrieval_config, args.objective)
    beta_config = beta_configuration(names, retrieval_config, args.objective)
    logger.info("Fixed atmospheric/nuisance parameters from %s: %s", args.parameters_json, parameters)
    logger.info("Objective: %s", args.objective)
    logger.info("alpha mode: %s", alpha_mode_label(alpha_config))
    logger.info("beta mode: %s", beta_mode_label(beta_config))

    template = build_prt_xcorr_template(retrieval_config, exopipe_config, base_parameters, logger=logger)
    blocks = load_hrccs_data(
        config=exopipe_config,
        params=exopipe_params,
        k=args.k,
        nights=split_cli_list(args.nights),
        cameras=split_cli_list(args.cameras),
        model_array=template["model_array"],
        orders=parse_int_list(args.orders),
        logger=logger,
    )
    wavelength_padding_summary = log_model_data_wavelength_padding(blocks, retrieval_config, logger=logger)

    kp_grid = inclusive_grid(args.kp_min, args.kp_max, args.kp_step)
    vsys_grid = inclusive_grid(args.vsys_min, args.vsys_max, args.vsys_step)
    loglike_map = np.full((kp_grid.size, vsys_grid.size), np.nan, dtype=float)
    scale_state = {"alpha_config": alpha_config, "beta_config": beta_config}

    for kp_idx, kp in enumerate(kp_grid):
        for vsys_idx, vsys in enumerate(vsys_grid):
            parameters = dict(base_parameters)
            parameters["Kp"] = float(kp)
            parameters["Vsys"] = float(vsys)
            result = evaluate_objective(
                blocks=blocks,
                F_model=template["F_model"],
                Kp=float(kp),
                Vsys=float(vsys),
                objective=args.objective,
                alpha=alpha_from_state(parameters, scale_state),
                beta=beta_from_state(parameters, scale_state),
            )
            loglike_map[kp_idx, vsys_idx] = float(result["objective_value"])

    best_flat = int(np.nanargmax(loglike_map))
    best_kp_idx, best_vsys_idx = np.unravel_index(best_flat, loglike_map.shape)
    best = {
        "Kp": float(kp_grid[best_kp_idx]),
        "Vsys": float(vsys_grid[best_vsys_idx]),
        "objective_value": float(loglike_map[best_kp_idx, best_vsys_idx]),
        "kp_index": int(best_kp_idx),
        "vsys_index": int(best_vsys_idx),
    }

    np.savez_compressed(
        output_dir / "kp_vsys_likelihood_grid.npz",
        Kp=kp_grid,
        Vsys=vsys_grid,
        log_likelihood=loglike_map,
        delta_log_likelihood=loglike_map - np.nanmax(loglike_map),
    )
    plot_saved = save_grid_plot(
        output_dir / "kp_vsys_likelihood_grid.png",
        kp_grid,
        vsys_grid,
        loglike_map,
        best,
        objective=args.objective,
    )

    summary = {
        "project_path": str(args.project_path),
        "retrieval_config": str(args.retrieval_config),
        "parameters_json": str(args.parameters_json),
        "objective": args.objective,
        "fixed_parameters_used": {key: float(value) for key, value in base_parameters.items()},
        "alpha_mode": alpha_config,
        "beta_mode": beta_config,
        "kp_grid": [float(kp_grid[0]), float(kp_grid[-1]), float(args.kp_step)],
        "vsys_grid": [float(vsys_grid[0]), float(vsys_grid[-1]), float(args.vsys_step)],
        "best": best,
        "plot_saved": bool(plot_saved),
        "data": block_summary(blocks),
        "wavelength_padding": wavelength_padding_summary,
    }
    with (output_dir / "kp_vsys_likelihood_grid_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    logger.info("Best Kp/Vsys grid point: %s", best)
    logger.info("Saved Kp/Vsys diagnostic outputs to %s", output_dir)


if __name__ == "__main__":
    main()
