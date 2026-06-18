"""Decompose HRCCS matched-filter terms at selected Kp/Vsys points."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

from retrieval.prt_emission_model import setup_logging

from .ccf_likelihood import matched_filter_component_diagnostics
from .data_loading import (
    block_summary,
    load_hrccs_data,
    load_project_modules,
    log_model_data_wavelength_padding,
    parse_int_list,
    split_cli_list,
)
from .model_builder import build_prt_xcorr_template, load_retrieval_config_and_parameters, parameters_with_updates
from .sampler_common import fixed_parameters_from_config


CSV_FIELDS = [
    "point_label",
    "Kp",
    "Vsys",
    "variant",
    "order_label",
    "n_valid",
    "n_possible",
    "valid_fraction",
    "n_exposure_terms",
    "data_mean",
    "model_mean",
    "weighted_data_mean",
    "weighted_model_mean",
    "weighted_data_mean_after_centering",
    "weighted_model_mean_after_centering",
    "data_rms",
    "model_rms",
    "data_model",
    "model_power",
    "model_norm_term",
    "data_power",
    "amplitude",
    "matched_filter_improvement",
    "data_penalty_loglike",
    "fit_term_loglike",
    "chi2_best",
    "log_likelihood",
    "gaussian_normalization_term",
    "log_likelihood_with_noise_constant",
    "log_noise_term",
    "normalized_correlation",
    "ccf_peak_value",
    "delta_log_likelihood_vs_reference",
    "delta_data_model_vs_reference",
    "delta_model_power_vs_reference",
    "delta_data_power_vs_reference",
    "delta_n_valid_vs_reference",
    "delta_gaussian_normalization_vs_reference",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_path")
    parser.add_argument("--retrieval-config", required=True)
    parser.add_argument("--parameters-json", required=True, help="Plain parameter JSON or emcee/dynesty summary JSON.")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--nights", nargs="+", default=None)
    parser.add_argument("--cameras", nargs="+", default=None)
    parser.add_argument("--orders", nargs="+", default=None)
    parser.add_argument(
        "--kp-vsys",
        nargs=3,
        action="append",
        metavar=("KP", "VSYS", "LABEL"),
        required=True,
        help="Velocity point to diagnose. Repeat as: --kp-vsys 196.5 -1.0 ccf_peak",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def json_default(value: Any) -> Any:
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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)


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


def parse_velocity_points(raw_points: list[list[str]]) -> list[dict[str, Any]]:
    points = []
    for raw_kp, raw_vsys, raw_label in raw_points:
        points.append(
            {
                "Kp": float(raw_kp),
                "Vsys": float(raw_vsys),
                "label": str(raw_label),
            }
        )
    labels = [point["label"] for point in points]
    if len(labels) != len(set(labels)):
        raise ValueError(f"--kp-vsys labels must be unique; got {labels}.")
    return points


def scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Return only CSV/JSON scalar metrics from a component record."""

    return {
        key: metrics.get(key)
        for key in CSV_FIELDS
        if key
        not in {
            "point_label",
            "Kp",
            "Vsys",
            "variant",
            "order_label",
            "delta_log_likelihood_vs_reference",
            "delta_data_model_vs_reference",
            "delta_model_power_vs_reference",
            "delta_data_power_vs_reference",
            "delta_n_valid_vs_reference",
            "delta_gaussian_normalization_vs_reference",
        }
    }


def add_reference_deltas(row: dict[str, Any], reference: Mapping[str, Any] | None) -> dict[str, Any]:
    if reference is None:
        row.update(
            {
                "delta_log_likelihood_vs_reference": 0.0,
                "delta_data_model_vs_reference": 0.0,
                "delta_model_power_vs_reference": 0.0,
                "delta_data_power_vs_reference": 0.0,
                "delta_n_valid_vs_reference": 0,
                "delta_gaussian_normalization_vs_reference": 0.0,
            }
        )
        return row

    row["delta_log_likelihood_vs_reference"] = safe_delta(row, reference, "log_likelihood")
    row["delta_data_model_vs_reference"] = safe_delta(row, reference, "data_model")
    row["delta_model_power_vs_reference"] = safe_delta(row, reference, "model_power")
    row["delta_data_power_vs_reference"] = safe_delta(row, reference, "data_power")
    row["delta_n_valid_vs_reference"] = int(row.get("n_valid", 0)) - int(reference.get("n_valid", 0))
    row["delta_gaussian_normalization_vs_reference"] = safe_delta(row, reference, "gaussian_normalization_term")
    return row


def safe_delta(row: Mapping[str, Any], reference: Mapping[str, Any], key: str) -> float:
    try:
        return float(row.get(key)) - float(reference.get(key))
    except Exception:
        return float("nan")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def diagnose_driver_hint(winner: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    """Return simple component deltas explaining why one point won."""

    deltas = {
        "log_likelihood": safe_delta(winner, reference, "log_likelihood"),
        "dot_product_term": safe_delta(winner, reference, "data_model"),
        "model_norm_term": safe_delta(winner, reference, "model_power"),
        "data_power": safe_delta(winner, reference, "data_power"),
        "fit_term_loglike": safe_delta(winner, reference, "fit_term_loglike"),
        "data_penalty_loglike": safe_delta(winner, reference, "data_penalty_loglike"),
        "valid_pixels": float(int(winner.get("n_valid", 0)) - int(reference.get("n_valid", 0))),
        "gaussian_normalization_term": safe_delta(winner, reference, "gaussian_normalization_term"),
    }
    finite = {
        key: abs(value)
        for key, value in deltas.items()
        if isinstance(value, (int, float)) and value == value
    }
    largest = max(finite, key=finite.get) if finite else None
    return {
        "deltas_vs_reference": deltas,
        "largest_absolute_delta": largest,
        "note": (
            "For log_likelihood, the direct additive pieces are "
            "data_penalty_loglike=-0.5*data_power and "
            "fit_term_loglike=0.5*data_model^2/model_power. "
            "Dot/model-norm/valid-pixel deltas help diagnose why the fit term changed."
        ),
    }


def save_diagnostic_plots(output_dir: Path, velocity_rows: list[dict[str, Any]], order_rows: list[dict[str, Any]], reference_label: str) -> dict[str, str | None]:
    """Save optional component diagnostic plots."""

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return {
            "per_order_contribution": None,
            "model_mean_norm": None,
            "valid_pixels": None,
            "dot_vs_norm": None,
        }

    plot_paths: dict[str, str | None] = {}
    current_orders = [row for row in order_rows if row["variant"] == "current"]
    labels = sorted({row["point_label"] for row in current_orders})
    if reference_label in labels and len(labels) >= 2:
        compare_label = next(label for label in labels if label != reference_label)
        ref_rows = [row for row in current_orders if row["point_label"] == reference_label]
        cmp_rows = [row for row in current_orders if row["point_label"] == compare_label]
        ref_map = {row["order_label"]: row for row in ref_rows}
        cmp_map = {row["order_label"]: row for row in cmp_rows}
        common_orders = [order for order in ref_map if order in cmp_map]
        if common_orders:
            x = np.arange(len(common_orders))
            fig, ax = plt.subplots(figsize=(max(8, 0.28 * len(common_orders)), 4))
            ax.plot(x, [ref_map[order]["log_likelihood"] for order in common_orders], label=reference_label)
            ax.plot(x, [cmp_map[order]["log_likelihood"] for order in common_orders], label=compare_label)
            ax.set_xticks(x[:: max(1, len(x) // 12)])
            ax.set_xticklabels([common_orders[i] for i in x[:: max(1, len(x) // 12)]], rotation=45, ha="right")
            ax.set_ylabel("Per-order log likelihood")
            ax.legend(loc="best")
            fig.tight_layout()
            path = output_dir / "matched_filter_per_order_contribution.png"
            fig.savefig(path, dpi=220, bbox_inches="tight")
            plt.close(fig)
            plot_paths["per_order_contribution"] = str(path)
        else:
            plot_paths["per_order_contribution"] = None
    else:
        plot_paths["per_order_contribution"] = None

    current_velocity = [row for row in velocity_rows if row["variant"] == "current"]
    if current_velocity:
        labels = [row["point_label"] for row in current_velocity]
        x = np.arange(len(labels))

        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(x, [row["weighted_model_mean"] for row in current_velocity], marker="o", label="weighted model mean")
        ax1.set_ylabel("Weighted model mean")
        ax2 = ax1.twinx()
        ax2.plot(x, [row["model_power"] for row in current_velocity], marker="s", color="tab:orange", label="model power")
        ax2.set_ylabel("Model power")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=30, ha="right")
        fig.tight_layout()
        path = output_dir / "matched_filter_model_mean_norm.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        plot_paths["model_mean_norm"] = str(path)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, [row["n_valid"] for row in current_velocity], marker="o")
        ax.set_ylabel("Valid pixels")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        fig.tight_layout()
        path = output_dir / "matched_filter_valid_pixels.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        plot_paths["valid_pixels"] = str(path)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(
            [row["model_power"] for row in current_velocity],
            [row["data_model"] for row in current_velocity],
        )
        for row in current_velocity:
            ax.annotate(row["point_label"], (row["model_power"], row["data_model"]), fontsize=8)
        ax.set_xlabel("Model norm term")
        ax.set_ylabel("Data dot model term")
        fig.tight_layout()
        path = output_dir / "matched_filter_dot_vs_norm.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        plot_paths["dot_vs_norm"] = str(path)
    else:
        plot_paths.update({"model_mean_norm": None, "valid_pixels": None, "dot_vs_norm": None})

    return plot_paths


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir, "matched_filter_component.log")

    points = parse_velocity_points(args.kp_vsys)
    reference_label = points[0]["label"]

    exopipe_config, exopipe_params = load_project_modules(args.project_path)
    retrieval_config, initial = load_retrieval_config_and_parameters(args.retrieval_config)
    supplied = load_parameter_values(args.parameters_json)
    fixed_parameters = fixed_parameters_from_config(retrieval_config)
    parameters = parameters_with_updates(initial, supplied)
    parameters.update(fixed_parameters)

    logger.info("Fixed atmospheric/nuisance parameters from %s: %s", args.parameters_json, parameters)
    logger.info("Reference velocity point: %s", reference_label)

    template = build_prt_xcorr_template(retrieval_config, exopipe_config, parameters, logger=logger)
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

    diagnostics_by_point: dict[str, Any] = {}
    velocity_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    reference_global: dict[str, dict[str, Any]] = {}
    reference_order: dict[tuple[str, str], dict[str, Any]] = {}

    for point in points:
        label = point["label"]
        logger.info("Evaluating matched-filter components at Kp=%.3f Vsys=%.3f label=%s", point["Kp"], point["Vsys"], label)
        diagnostics = matched_filter_component_diagnostics(
            blocks=blocks,
            F_model=template["F_model"],
            Kp=point["Kp"],
            Vsys=point["Vsys"],
            include_per_order=True,
        )
        diagnostics_by_point[label] = diagnostics

        for variant, variant_result in diagnostics.items():
            global_metrics = scalar_metrics(variant_result["global"])
            logger.info(
                "label=%s variant=%s loglike=%.6e valid=%d/%d fraction=%.6f "
                "dot=%.6e model_power=%.6e amplitude=%.6e",
                label,
                variant,
                float(global_metrics["log_likelihood"]),
                int(global_metrics["n_valid"]),
                int(global_metrics["n_possible"]),
                float(global_metrics["valid_fraction"]),
                float(global_metrics["data_model"]),
                float(global_metrics["model_power"]),
                float(global_metrics["amplitude"]),
            )
            row = {
                "point_label": label,
                "Kp": float(point["Kp"]),
                "Vsys": float(point["Vsys"]),
                "variant": variant,
                "order_label": "GLOBAL",
                **global_metrics,
            }
            if label == reference_label:
                reference_global[variant] = dict(row)
            add_reference_deltas(row, reference_global.get(variant))
            velocity_rows.append(row)

            for order_metrics_in in variant_result.get("per_order", []):
                order_metrics = scalar_metrics(order_metrics_in)
                order_label = str(order_metrics_in["label"])
                order_row = {
                    "point_label": label,
                    "Kp": float(point["Kp"]),
                    "Vsys": float(point["Vsys"]),
                    "variant": variant,
                    "order_label": order_label,
                    **order_metrics,
                }
                key = (variant, order_label)
                if label == reference_label:
                    reference_order[key] = dict(order_row)
                add_reference_deltas(order_row, reference_order.get(key))
                order_rows.append(order_row)

    winner_summary: dict[str, Any] = {}
    for variant in sorted({row["variant"] for row in velocity_rows}):
        rows = [row for row in velocity_rows if row["variant"] == variant]
        best = max(rows, key=lambda row: row["log_likelihood"])
        reference = reference_global.get(variant, rows[0])
        winner_summary[variant] = {
            "best_point_label": best["point_label"],
            "best_Kp": best["Kp"],
            "best_Vsys": best["Vsys"],
            "best_log_likelihood": best["log_likelihood"],
            "reference_point_label": reference_label,
            "reference_log_likelihood": reference.get("log_likelihood"),
            "best_minus_reference": diagnose_driver_hint(best, reference),
        }

    velocity_csv = output_dir / "matched_filter_component_by_velocity.csv"
    order_csv = output_dir / "matched_filter_component_by_order.csv"
    write_csv(velocity_csv, velocity_rows)
    write_csv(order_csv, order_rows)
    plots = save_diagnostic_plots(output_dir, velocity_rows, order_rows, reference_label)

    summary = {
        "project_path": str(args.project_path),
        "retrieval_config": str(args.retrieval_config),
        "parameters_json": str(args.parameters_json),
        "parameters_used": {key: float(value) for key, value in parameters.items()},
        "sysrem_iteration": int(args.k),
        "nights": split_cli_list(args.nights),
        "cameras": split_cli_list(args.cameras),
        "orders": parse_int_list(args.orders),
        "velocity_points": points,
        "reference_point_label": reference_label,
        "variant_definitions": {
            "current": "Current matched-filter centering: data weighted-mean centered; model only pre-standardized before exact data/sigma overlap.",
            "zero_mean_model": "Current data treatment plus exact-overlap weighted model mean subtraction.",
            "zero_mean_data": "Exact-overlap weighted data mean subtraction only; equivalent to current data treatment in the current code.",
            "zero_mean_data_model": "Exact-overlap weighted mean subtraction from both data and model.",
        },
        "global_rows": velocity_rows,
        "winner_summary": winner_summary,
        "data": block_summary(blocks),
        "wavelength_padding": wavelength_padding_summary,
        "output_files": {
            "summary": str(output_dir / "matched_filter_component_summary.json"),
            "by_velocity_csv": str(velocity_csv),
            "by_order_csv": str(order_csv),
            "log": str(output_dir / "matched_filter_component.log"),
            **plots,
        },
    }
    write_json(output_dir / "matched_filter_component_summary.json", summary)

    logger.info("Winner summary by variant: %s", winner_summary)
    logger.info("Saved matched-filter component diagnostics to %s", output_dir)


if __name__ == "__main__":
    main()
