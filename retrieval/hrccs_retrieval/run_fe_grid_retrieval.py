"""Run a small Fe-only HRCCS matched-filter grid before sampling."""

from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing as mp
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from retrieval.prt_emission_model import setup_logging

from .ccf_likelihood import evaluate_objective, write_json
from .data_loading import block_summary, load_hrccs_data, load_project_modules, parse_int_list, split_cli_list
from .model_builder import build_prt_xcorr_template, load_retrieval_config_and_parameters, parameters_with_updates


_GRID_STATE: dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_path")
    parser.add_argument("--retrieval-config", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--nights", nargs="+", default=None)
    parser.add_argument("--cameras", nargs="+", default=None)
    parser.add_argument("--orders", nargs="+", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fixed-kp", type=float, default=None)
    parser.add_argument("--fixed-vsys", type=float, default=None)
    parser.add_argument("--kp-grid", default=None, help='Grid string "min,max,step".')
    parser.add_argument("--vsys-grid", default=None, help='Grid string "min,max,step".')
    parser.add_argument("--T-deep-grid", default=None, help='Grid string "min,max,step".')
    parser.add_argument("--delta-T-grid", default=None, help='Grid string "min,max,step".')
    parser.add_argument("--logFe-grid", default=None, help='Grid string "min,max,step".')
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--objective",
        choices=["matched_filter_loglike", "ccf_peak_value"],
        default="matched_filter_loglike",
    )
    return parser.parse_args()


def parse_grid(spec: str | None, default: float) -> list[float]:
    """Parse inclusive min,max,step grid strings."""

    if spec in {None, ""}:
        return [float(default)]
    import numpy as np

    pieces = [float(piece.strip()) for piece in str(spec).split(",")]
    if len(pieces) != 3:
        raise ValueError(f"Grid spec must be min,max,step; got {spec!r}.")
    lo, hi, step = pieces
    if step <= 0 or hi < lo:
        raise ValueError(f"Invalid grid spec {spec!r}.")
    return [float(x) for x in np.arange(lo, hi + 0.5 * step, step)]


def build_tasks(args: argparse.Namespace, initial: dict[str, float]) -> list[dict[str, float]]:
    kp_default = initial["Kp"] if args.fixed_kp is None else float(args.fixed_kp)
    vsys_default = initial["Vsys"] if args.fixed_vsys is None else float(args.fixed_vsys)
    kp_values = [float(args.fixed_kp)] if args.fixed_kp is not None else parse_grid(args.kp_grid, kp_default)
    vsys_values = [float(args.fixed_vsys)] if args.fixed_vsys is not None else parse_grid(args.vsys_grid, vsys_default)
    t_values = parse_grid(args.T_deep_grid, initial["T_deep"])
    dt_values = parse_grid(args.delta_T_grid, initial["delta_T_inv"])
    fe_values = parse_grid(args.logFe_grid, initial["log10_Fe"])

    tasks = []
    for kp, vsys, t_deep, delta_t, log_fe in itertools.product(kp_values, vsys_values, t_values, dt_values, fe_values):
        tasks.append(
            {
                "Kp": float(kp),
                "Vsys": float(vsys),
                "T_deep": float(t_deep),
                "delta_T_inv": float(delta_t),
                "log10_Fe": float(log_fe),
            }
        )
    return tasks


def evaluate_grid_task(task: dict[str, float]) -> dict[str, Any]:
    start = time.perf_counter()
    parameters = parameters_with_updates(_GRID_STATE["initial"], task)
    template = build_prt_xcorr_template(
        retrieval_config=_GRID_STATE["retrieval_config"],
        exopipe_config=_GRID_STATE["exopipe_config"],
        parameters=parameters,
        logger=None,
    )
    result = evaluate_objective(
        blocks=_GRID_STATE["blocks"],
        F_model=template["F_model"],
        Kp=parameters["Kp"],
        Vsys=parameters["Vsys"],
        objective=_GRID_STATE["objective"],
    )
    result.update(task)
    result["seconds"] = float(time.perf_counter() - start)
    return result


def init_grid_worker(state: dict[str, Any]) -> None:
    global _GRID_STATE
    _GRID_STATE = state


def write_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_grid_plots(rows: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    for name in ["Kp", "Vsys", "T_deep", "delta_T_inv", "log10_Fe"]:
        xs = [row[name] for row in rows]
        ys = [row["objective_value"] for row in rows]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.scatter(xs, ys, s=18)
        ax.set_xlabel(name)
        ax.set_ylabel("objective_value")
        fig.tight_layout()
        fig.savefig(output_dir / f"grid_objective_vs_{name}.png", dpi=250)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir, "fe_hrccs_grid.log")

    exopipe_config, exopipe_params = load_project_modules(args.project_path)
    retrieval_config, initial = load_retrieval_config_and_parameters(args.retrieval_config)

    # Build one initial template first so order selection matches the trusted
    # xcorr model-overlap rule, then keep the data fixed for all evaluations.
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
    tasks = build_tasks(args, initial)
    logger.info("Running %d HRCCS grid evaluations with n_jobs=%d", len(tasks), int(args.n_jobs))

    state = {
        "retrieval_config": retrieval_config,
        "exopipe_config": SimpleNamespace(ghost_res=float(exopipe_config.ghost_res)),
        "blocks": blocks,
        "initial": initial,
        "objective": args.objective,
    }

    rows: list[dict[str, Any]] = []
    if int(args.n_jobs) == 1:
        init_grid_worker(state)
        for index, task in enumerate(tasks):
            row = evaluate_grid_task(task)
            rows.append(row)
            logger.info(
                "grid %d/%d Kp=%.2f Vsys=%.2f T=%.1f dT=%.1f logFe=%.2f objective=%.6e in %.2fs",
                index + 1,
                len(tasks),
                row["Kp"],
                row["Vsys"],
                row["T_deep"],
                row["delta_T_inv"],
                row["log10_Fe"],
                row["objective_value"],
                row["seconds"],
            )
    else:
        with mp.Pool(processes=int(args.n_jobs), initializer=init_grid_worker, initargs=(state,)) as pool:
            for index, row in enumerate(pool.imap_unordered(evaluate_grid_task, tasks, chunksize=1)):
                rows.append(row)
                logger.info("grid %d/%d objective=%.6e seconds=%.2f", index + 1, len(tasks), row["objective_value"], row["seconds"])

    rows = sorted(rows, key=lambda row: row["objective_value"], reverse=True)
    best = rows[0] if rows else None
    write_rows_csv(rows, output_dir / "fe_hrccs_grid_results.csv")
    save_grid_plots(rows, output_dir)
    write_json(
        output_dir / "fe_hrccs_grid_summary.json",
        {
            "project_path": str(args.project_path),
            "retrieval_config": str(args.retrieval_config),
            "sysrem_iteration": int(args.k),
            "objective": args.objective,
            "n_evaluations": len(rows),
            "best": best,
            "data": block_summary(blocks),
        },
    )
    logger.info("Best grid row: %s", best)
    logger.info("Saved HRCCS grid outputs to %s", output_dir)


if __name__ == "__main__":
    main()
