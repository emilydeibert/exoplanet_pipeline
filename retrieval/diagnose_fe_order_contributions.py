"""Diagnose per-order likelihood contributions at selected Kp/Vsys points."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from retrieval.likelihood import (
    RetrievalData,
    compute_log_likelihood,
    load_retrieval_data,
    model_wavelength_bounds_for_data,
)
from retrieval.prt_emission_model import (
    build_convolved_model,
    generate_prt_emission_model,
    load_yaml_config,
    log_run_summary,
    parameters_from_config,
    prepare_model_like_data,
    require_numpy,
    setup_logging,
    shifted_model_cube,
    wavelengths_cm_to_micron,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="retrieval/configs/mascara1b_fe_smoketest.yaml",
        help="YAML retrieval config.",
    )
    parser.add_argument(
        "--points",
        required=True,
        help='Semicolon-separated Kp,Vsys pairs, e.g. "194,0;210,1;220,1".',
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for CSV, JSON, and diagnostic plots.",
    )
    return parser.parse_args()


def parse_points(points_text: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for chunk in points_text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        pieces = [piece.strip() for piece in chunk.split(",")]
        if len(pieces) != 2:
            raise ValueError(
                "Each --points entry must be Kp,Vsys. "
                f"Could not parse {chunk!r} from {points_text!r}."
            )
        points.append((float(pieces[0]), float(pieces[1])))

    if not points:
        raise ValueError("--points did not contain any Kp,Vsys pairs.")
    return points


def selected_order_labels(config: dict[str, Any], n_orders: int) -> list[int]:
    orders = config.get("selection", {}).get("orders", None)
    if orders is None:
        return list(range(n_orders))
    labels = [int(order) for order in orders]
    if len(labels) != n_orders:
        return list(range(n_orders))
    return labels


def one_order_data(data: RetrievalData, order_index: int) -> RetrievalData:
    return replace(
        data,
        wavelengths_cm=data.wavelengths_cm[order_index : order_index + 1],
        flux=data.flux[order_index : order_index + 1],
        uncertainties=data.uncertainties[order_index : order_index + 1],
        good_mask=data.good_mask[order_index : order_index + 1],
    )


def order_wavelength_summary(wavelengths_cm: Any, good_mask: Any) -> tuple[float, float, float]:
    np = require_numpy()
    wave_um = wavelengths_cm_to_micron(wavelengths_cm)
    usable_pixels = np.any(good_mask, axis=0)
    finite = np.isfinite(wave_um) & usable_pixels
    if not finite.any():
        finite = np.isfinite(wave_um)
    if not finite.any():
        return float("nan"), float("nan"), float("nan")

    wave_min = float(np.nanmin(wave_um[finite]))
    wave_max = float(np.nanmax(wave_um[finite]))
    return wave_min, wave_max, 0.5 * (wave_min + wave_max)


def compute_order_rows(
    data: RetrievalData,
    prepared_model: Any,
    kp: float,
    vsys: float,
    point_index: int,
    order_labels: Sequence[int],
    reference_by_order: dict[int, float],
    use_delta: bool,
) -> list[dict[str, Any]]:
    np = require_numpy()
    rows: list[dict[str, Any]] = []

    for order_index in range(data.flux.shape[0]):
        order_data = one_order_data(data, order_index)
        order_model = prepared_model[order_index : order_index + 1]
        valid = (
            order_data.good_mask
            & np.isfinite(order_data.flux)
            & np.isfinite(order_data.uncertainties)
            & (order_data.uncertainties > 0)
            & np.isfinite(order_model)
        )
        n_valid_pixels = int(np.sum(valid))

        if n_valid_pixels == 0:
            log_likelihood = float("-inf")
            amplitude = float("nan")
        else:
            log_likelihood, amplitude = compute_log_likelihood(
                order_data,
                order_model,
                log_model_scale=0.0,
                fit_amplitude_analytically=True,
            )

        if point_index == 0:
            reference_by_order[order_index] = log_likelihood

        if use_delta:
            reference = reference_by_order.get(order_index, float("nan"))
            delta_log_likelihood = float(log_likelihood - reference)
        else:
            delta_log_likelihood = 0.0

        wave_min, wave_max, wave_mid = order_wavelength_summary(
            data.wavelengths_cm[order_index],
            data.good_mask[order_index],
        )

        rows.append(
            {
                "point_index": point_index,
                "Kp": float(kp),
                "Vsys": float(vsys),
                "order_index": order_index,
                "original_order": int(order_labels[order_index]),
                "log_likelihood": float(log_likelihood),
                "delta_log_likelihood": float(delta_log_likelihood),
                "analytic_amplitude": float(amplitude),
                "n_valid_pixels": n_valid_pixels,
                "wavelength_min_micron": wave_min,
                "wavelength_max_micron": wave_max,
                "wavelength_mid_micron": wave_mid,
            }
        )

    return rows


def write_csv(rows: Sequence[dict[str, Any]], filename: Path) -> None:
    fieldnames = [
        "point_index",
        "Kp",
        "Vsys",
        "order_index",
        "original_order",
        "log_likelihood",
        "delta_log_likelihood",
        "analytic_amplitude",
        "n_valid_pixels",
        "wavelength_min_micron",
        "wavelength_max_micron",
        "wavelength_mid_micron",
    ]
    with filename.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_order_diagnostics(rows: Sequence[dict[str, Any]], output_dir: Path) -> None:
    np = require_numpy()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise RuntimeError("Matplotlib is required for order diagnostic plots.") from exc

    point_indices = sorted({int(row["point_index"]) for row in rows})

    fig, ax = plt.subplots(figsize=(9, 5))
    for point_index in point_indices:
        subset = [row for row in rows if int(row["point_index"]) == point_index]
        x = np.asarray([row["wavelength_mid_micron"] for row in subset], dtype=float)
        y = np.asarray([row["delta_log_likelihood"] for row in subset], dtype=float)
        kp = subset[0]["Kp"]
        vsys = subset[0]["Vsys"]
        ax.plot(x, y, marker="o", lw=1.0, label=f"Kp={kp:g}, Vsys={vsys:g}")
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xlabel("Order midpoint wavelength [micron]")
    ax.set_ylabel("Delta log likelihood vs reference")
    ax.set_title("Per-order delta log likelihood")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_dir / "per_order_delta_log_likelihood.png", dpi=250)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for point_index in point_indices:
        subset = [row for row in rows if int(row["point_index"]) == point_index]
        x = np.asarray([row["wavelength_mid_micron"] for row in subset], dtype=float)
        y = np.asarray([row["analytic_amplitude"] for row in subset], dtype=float)
        kp = subset[0]["Kp"]
        vsys = subset[0]["Vsys"]
        ax.plot(x, y, marker="o", lw=1.0, label=f"Kp={kp:g}, Vsys={vsys:g}")
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xlabel("Order midpoint wavelength [micron]")
    ax.set_ylabel("Per-order analytic amplitude")
    ax.set_title("Per-order fitted amplitude")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_dir / "per_order_fitted_amplitude.png", dpi=250)
    plt.close(fig)


def summarize_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    np = require_numpy()
    summaries: list[dict[str, Any]] = []
    point_indices = sorted({int(row["point_index"]) for row in rows})

    for point_index in point_indices:
        subset = [row for row in rows if int(row["point_index"]) == point_index]
        loglikes = np.asarray([row["log_likelihood"] for row in subset], dtype=float)
        deltas = np.asarray([row["delta_log_likelihood"] for row in subset], dtype=float)
        amplitudes = np.asarray([row["analytic_amplitude"] for row in subset], dtype=float)
        valid_pixels = int(sum(int(row["n_valid_pixels"]) for row in subset))

        finite_delta = np.isfinite(deltas)
        if finite_delta.any():
            best_delta_row = subset[int(np.nanargmax(deltas))]
            worst_delta_row = subset[int(np.nanargmin(deltas))]
        else:
            best_delta_row = None
            worst_delta_row = None

        summaries.append(
            {
                "point_index": point_index,
                "Kp": float(subset[0]["Kp"]),
                "Vsys": float(subset[0]["Vsys"]),
                "total_log_likelihood": float(np.nansum(loglikes)),
                "total_delta_log_likelihood": float(np.nansum(deltas)),
                "median_analytic_amplitude": float(np.nanmedian(amplitudes)),
                "n_valid_pixels": valid_pixels,
                "best_delta_order": (
                    {
                        "order_index": int(best_delta_row["order_index"]),
                        "original_order": int(best_delta_row["original_order"]),
                        "delta_log_likelihood": float(best_delta_row["delta_log_likelihood"]),
                        "wavelength_mid_micron": float(best_delta_row["wavelength_mid_micron"]),
                    }
                    if best_delta_row is not None
                    else None
                ),
                "worst_delta_order": (
                    {
                        "order_index": int(worst_delta_row["order_index"]),
                        "original_order": int(worst_delta_row["original_order"]),
                        "delta_log_likelihood": float(worst_delta_row["delta_log_likelihood"]),
                        "wavelength_mid_micron": float(worst_delta_row["wavelength_mid_micron"]),
                    }
                    if worst_delta_row is not None
                    else None
                ),
            }
        )

    return summaries


def main() -> None:
    args = parse_args()
    require_numpy()

    config = load_yaml_config(args.config)
    points = parse_points(args.points)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir, "fe_order_contributions.log")

    parameters = parameters_from_config(config)
    data = load_retrieval_data(config, logger=logger)
    log_run_summary(logger, config, parameters, wavelengths_cm=data.wavelengths_cm, mask=data.good_mask)

    wavelength_bounds = model_wavelength_bounds_for_data(data, config)
    rest_wave_cm, rest_flux, metadata = generate_prt_emission_model(
        config=config,
        parameters=parameters,
        wavelength_boundaries_micron=wavelength_bounds,
        logger=logger,
    )
    logger.info("Model metadata: %s", metadata)

    convolved_model = build_convolved_model(
        rest_wave_cm,
        rest_flux,
        float(config["instrument"]["resolving_power"]),
    )

    order_labels = selected_order_labels(config, data.flux.shape[0])
    rows: list[dict[str, Any]] = []
    reference_by_order: dict[int, float] = {}
    use_delta = len(points) > 1

    for point_index, (kp, vsys) in enumerate(points):
        logger.info("Evaluating per-order point %d: Kp=%.6f Vsys=%.6f", point_index, kp, vsys)
        shifted = shifted_model_cube(
            rest_wavelengths_cm=rest_wave_cm,
            rest_flux=rest_flux,
            observed_wavelengths_cm=data.wavelengths_cm,
            phases=data.phases,
            Kp=kp,
            Vsys=vsys,
            resolving_power=float(config["instrument"]["resolving_power"]),
            barycentric_velocities=data.barycentric_velocities,
            velocity_config=config.get("velocity", {}),
            convolved_rest_flux=convolved_model.flux,
        )
        prepared = prepare_model_like_data(shifted, config, data_mask=data.good_mask)
        point_rows = compute_order_rows(
            data=data,
            prepared_model=prepared,
            kp=kp,
            vsys=vsys,
            point_index=point_index,
            order_labels=order_labels,
            reference_by_order=reference_by_order,
            use_delta=use_delta,
        )
        rows.extend(point_rows)

    csv_path = output_dir / "per_order_likelihood_contributions.csv"
    write_csv(rows, csv_path)
    plot_order_diagnostics(rows, output_dir)

    summary = {
        "config": str(args.config),
        "points": [
            {"point_index": index, "Kp": float(kp), "Vsys": float(vsys)}
            for index, (kp, vsys) in enumerate(points)
        ],
        "reference_point": (
            {"point_index": 0, "Kp": float(points[0][0]), "Vsys": float(points[0][1])}
            if len(points) > 1
            else None
        ),
        "n_orders": int(data.flux.shape[0]),
        "n_exposures": int(data.flux.shape[1]),
        "n_pixels": int(data.flux.shape[2]),
        "point_summaries": summarize_rows(rows),
        "csv": str(csv_path),
        "plots": {
            "delta_log_likelihood": str(output_dir / "per_order_delta_log_likelihood.png"),
            "fitted_amplitude": str(output_dir / "per_order_fitted_amplitude.png"),
        },
        "model_metadata": {
            "line_species": metadata.get("line_species", []),
            "gas_continuum_contributors": metadata.get("gas_continuum_contributors", []),
            "mass_fraction_keys": metadata.get("mass_fraction_keys", []),
        },
    }

    json_path = output_dir / "per_order_likelihood_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    logger.info("Saved per-order CSV to %s", csv_path)
    logger.info("Saved per-order summary to %s", json_path)
    logger.info("Saved per-order plots to %s", output_dir)


if __name__ == "__main__":
    main()
