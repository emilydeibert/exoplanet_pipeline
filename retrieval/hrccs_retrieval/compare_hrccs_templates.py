"""Compare two HRCCS pRT xcorr templates without running a retrieval."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

from retrieval.prt_emission_model import (
    build_pressure_grid,
    gas_continuum_contributors,
    requested_gas_continuum_contributors,
    resolve_prt_species_names,
    setup_logging,
    temperature_pressure_parameter_report,
    temperature_profile_type,
)

from .data_loading import load_project_modules
from .model_builder import (
    build_prt_xcorr_template,
    load_retrieval_config_and_parameters,
    parameters_with_updates,
    xcorr_processing_settings,
)
from .sampler_common import fixed_parameters_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_path")
    parser.add_argument("--config-a", required=True)
    parser.add_argument("--params-a-json", required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--config-b", required=True)
    parser.add_argument("--params-b-json", required=True)
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def json_default(value: Any) -> Any:
    """JSON fallback for NumPy scalars, arrays, and paths."""

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
    """Load parameters from a plain JSON mapping or sampler summary JSON."""

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


def sanitize_label(label: str) -> str:
    """Return a filesystem-safe label."""

    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label).strip())
    return clean or "template"


def finite_sorted_xy(wavelength: Any, values: Any) -> tuple[Any, Any]:
    """Return finite 1D wavelength/value arrays sorted by wavelength."""

    import numpy as np

    wavelength = np.asarray(wavelength, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)
    if wavelength.size != values.size:
        raise ValueError("Wavelength and template arrays must have the same length.")
    finite = np.isfinite(wavelength) & np.isfinite(values)
    wavelength = wavelength[finite]
    values = values[finite]
    order = np.argsort(wavelength)
    return wavelength[order], values[order]


def vector_stats(values: Any) -> dict[str, Any]:
    """Return finite-fraction and amplitude diagnostics for one vector."""

    import numpy as np

    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    stats: dict[str, Any] = {
        "n_pixels": int(arr.size),
        "n_finite": int(np.sum(finite)),
        "finite_fraction": float(np.sum(finite) / arr.size) if arr.size else 0.0,
    }
    if not finite.any():
        stats.update(
            {
                "min": None,
                "median": None,
                "max": None,
                "std": None,
                "rms": None,
                "l2_norm": None,
                "percentiles": {},
            }
        )
        return stats

    finite_values = arr[finite]
    percentile_levels = [0.1, 1.0, 5.0, 16.0, 50.0, 84.0, 95.0, 99.0, 99.9]
    percentiles = np.nanpercentile(finite_values, percentile_levels)
    stats.update(
        {
            "min": float(np.nanmin(finite_values)),
            "median": float(np.nanmedian(finite_values)),
            "max": float(np.nanmax(finite_values)),
            "std": float(np.nanstd(finite_values)),
            "rms": float(np.sqrt(np.nanmean(finite_values * finite_values))),
            "l2_norm": float(np.sqrt(np.nansum(finite_values * finite_values))),
            "percentiles": {
                f"p{level:g}": float(value)
                for level, value in zip(percentile_levels, percentiles)
            },
        }
    )
    return stats


def wavelength_stats(wavelength: Any) -> dict[str, Any]:
    """Return wavelength-grid diagnostics."""

    import numpy as np

    wave = np.asarray(wavelength, dtype=float).reshape(-1)
    finite = np.isfinite(wave)
    if not finite.any():
        return {
            "n_pixels": int(wave.size),
            "finite_fraction": 0.0,
            "min": None,
            "max": None,
            "monotonic_increasing": False,
        }
    finite_wave = wave[finite]
    return {
        "n_pixels": int(wave.size),
        "finite_fraction": float(np.sum(finite) / wave.size) if wave.size else 0.0,
        "min": float(np.nanmin(finite_wave)),
        "max": float(np.nanmax(finite_wave)),
        "monotonic_increasing": bool(np.all(np.diff(finite_wave) > 0)),
    }


def compact_array_metadata(value: Any) -> dict[str, Any]:
    """Return shape and stats for metadata arrays without embedding them in JSON."""

    import numpy as np

    arr = np.asarray(value, dtype=float)
    finite = np.isfinite(arr)
    out: dict[str, Any] = {
        "shape": list(arr.shape),
        "n_finite": int(np.sum(finite)),
        "finite_fraction": float(np.sum(finite) / arr.size) if arr.size else 0.0,
    }
    if finite.any():
        out.update(
            {
                "min": float(np.nanmin(arr[finite])),
                "median": float(np.nanmedian(arr[finite])),
                "max": float(np.nanmax(arr[finite])),
            }
        )
    return out


def safe_metadata_summary(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize model metadata while keeping large arrays out of JSON."""

    prt = dict(metadata.get("prt", {}))
    summary: dict[str, Any] = {
        "seconds": metadata.get("seconds"),
        "parameters": metadata.get("parameters", {}),
        "xcorr_processing": metadata.get("xcorr_processing", {}),
        "prt": {},
    }
    for key, value in prt.items():
        if key in {"pressures_bar", "temperatures"}:
            summary["prt"][key] = compact_array_metadata(value)
        else:
            summary["prt"][key] = value
    summary["raw_prt_arrays_available"] = "raw_prt_arrays" in metadata
    return summary


def resolve_parameters(
    config_path: str | Path,
    params_json: str | Path,
) -> tuple[dict[str, Any], dict[str, float], dict[str, float], dict[str, float]]:
    """Load config, initial parameters, supplied parameters, and final values."""

    retrieval_config, initial = load_retrieval_config_and_parameters(str(config_path))
    supplied = load_parameter_values(params_json)
    fixed = fixed_parameters_from_config(retrieval_config)
    parameters = parameters_with_updates(initial, supplied)
    parameters.update(fixed)
    return retrieval_config, initial, supplied, parameters


def build_template_record(
    *,
    label: str,
    config_path: str | Path,
    params_json: str | Path,
    project_config: Any,
    logger: Any,
) -> dict[str, Any]:
    """Build one template using the same path as the HRCCS samplers."""

    config, initial, supplied, parameters = resolve_parameters(config_path, params_json)
    logger.info("[%s] Building template from config %s", label, config_path)
    logger.info("[%s] Supplied parameters from %s: %s", label, params_json, supplied)
    logger.info("[%s] Final parameters used for model build: %s", label, parameters)

    template = build_prt_xcorr_template(
        config,
        project_config,
        parameters,
        logger=logger,
        include_raw_arrays=True,
    )
    return {
        "label": str(label),
        "config_path": str(config_path),
        "params_json": str(params_json),
        "config": config,
        "initial_parameters": initial,
        "supplied_parameters": supplied,
        "parameters": parameters,
        "template": template,
    }


def template_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    """Collect the requested pRT and model-processing metadata for one template."""

    import numpy as np

    config = record["config"]
    parameters = record["parameters"]
    template = record["template"]
    model_array = np.asarray(template["model_array"], dtype=float)
    final_wavelength_nm = np.asarray(template["model_wvl_nm"], dtype=float)
    final_template = np.asarray(template["model_dmag"], dtype=float)
    pressure_grid = np.asarray(build_pressure_grid(config), dtype=float)

    try:
        tp_report = temperature_pressure_parameter_report(parameters, config)
    except Exception as exc:
        tp_report = {"error": str(exc)}

    model_cfg = config.get("model", {})
    metadata = template.get("metadata", {})
    return {
        "label": record["label"],
        "config_path": record["config_path"],
        "params_json": record["params_json"],
        "parameters_used": {key: float(value) for key, value in parameters.items()},
        "pressure_grid": {
            "min_bar": float(np.nanmin(pressure_grid)),
            "max_bar": float(np.nanmax(pressure_grid)),
            "n_layers": int(pressure_grid.size),
            "log10_min": float(np.log10(np.nanmin(pressure_grid))),
            "log10_max": float(np.log10(np.nanmax(pressure_grid))),
        },
        "temperature_profile_type": temperature_profile_type(config),
        "temperature_pressure_report": tp_report,
        "line_species": resolve_prt_species_names(config),
        "yaml_requested_continuum_contributors": requested_gas_continuum_contributors(config),
        "radtrans_continuum_contributors": gas_continuum_contributors(config),
        "wavelength_boundaries_micron": model_cfg.get("wavelength_boundaries_micron"),
        "line_opacity_mode": model_cfg.get("line_opacity_mode", "lbl"),
        "line_by_line_opacity_sampling": model_cfg.get("line_by_line_opacity_sampling"),
        "model_representation": model_cfg.get("representation", "raw_flux"),
        "xcorr_processing_settings": xcorr_processing_settings(config),
        "model_processing_metadata": safe_metadata_summary(metadata),
        "xcorr_processed_model_array_angstrom": {
            "wavelength": wavelength_stats(model_array[:, 0]),
            "template": vector_stats(model_array[:, 1]),
        },
        "final_sampler_template_nm": {
            "wavelength": wavelength_stats(final_wavelength_nm),
            "template": vector_stats(final_template),
        },
    }


def save_template_arrays(output_dir: Path, record_a: Mapping[str, Any], record_b: Mapping[str, Any]) -> dict[str, str | None]:
    """Save native processed/final template arrays and raw pRT arrays if present."""

    import numpy as np

    label_a = sanitize_label(record_a["label"])
    label_b = sanitize_label(record_b["label"])
    template_a = record_a["template"]
    template_b = record_b["template"]
    arrays_path = output_dir / "template_comparison_arrays.npz"

    np.savez_compressed(
        arrays_path,
        label_a=np.asarray(str(record_a["label"])),
        label_b=np.asarray(str(record_b["label"])),
        a_xcorr_model_array=np.asarray(template_a["model_array"], dtype=float),
        b_xcorr_model_array=np.asarray(template_b["model_array"], dtype=float),
        a_final_wavelength_nm=np.asarray(template_a["model_wvl_nm"], dtype=float),
        a_final_template_dmag=np.asarray(template_a["model_dmag"], dtype=float),
        a_final_model_flux=np.asarray(template_a["model_flux"], dtype=float),
        a_final_model_convolved=np.asarray(template_a["model_conv"], dtype=float),
        b_final_wavelength_nm=np.asarray(template_b["model_wvl_nm"], dtype=float),
        b_final_template_dmag=np.asarray(template_b["model_dmag"], dtype=float),
        b_final_model_flux=np.asarray(template_b["model_flux"], dtype=float),
        b_final_model_convolved=np.asarray(template_b["model_conv"], dtype=float),
        a_pressures_bar=np.asarray(template_a["metadata"]["prt"].get("pressures_bar", []), dtype=float),
        a_temperatures=np.asarray(template_a["metadata"]["prt"].get("temperatures", []), dtype=float),
        b_pressures_bar=np.asarray(template_b["metadata"]["prt"].get("pressures_bar", []), dtype=float),
        b_temperatures=np.asarray(template_b["metadata"]["prt"].get("temperatures", []), dtype=float),
    )

    raw_paths: dict[str, str | None] = {"a": None, "b": None}
    for key, label, record in (("a", label_a, record_a), ("b", label_b, record_b)):
        raw = record["template"].get("metadata", {}).get("raw_prt_arrays")
        if not raw:
            continue
        raw_path = output_dir / f"raw_prt_{label}.npz"
        np.savez_compressed(
            raw_path,
            wavelength_cm=np.asarray(raw["wavelengths_cm"], dtype=float),
            flux=np.asarray(raw["flux"], dtype=float),
        )
        raw_paths[key] = str(raw_path)

    return {
        "template_arrays": str(arrays_path),
        "raw_prt_a": raw_paths["a"],
        "raw_prt_b": raw_paths["b"],
    }


def common_grid_diagnostics(record_a: Mapping[str, Any], record_b: Mapping[str, Any]) -> dict[str, Any]:
    """Interpolate the final sampler templates onto one common grid."""

    import numpy as np

    a_wave, a_template = finite_sorted_xy(
        record_a["template"]["model_wvl_nm"],
        record_a["template"]["model_dmag"],
    )
    b_wave, b_template = finite_sorted_xy(
        record_b["template"]["model_wvl_nm"],
        record_b["template"]["model_dmag"],
    )
    if a_wave.size < 2 or b_wave.size < 2:
        raise ValueError("Need at least two finite pixels in both templates for comparison.")

    overlap_min = max(float(a_wave[0]), float(b_wave[0]))
    overlap_max = min(float(a_wave[-1]), float(b_wave[-1]))
    if not overlap_max > overlap_min:
        raise ValueError(
            "Templates do not overlap in wavelength: "
            f"A={a_wave[0]}-{a_wave[-1]} nm, B={b_wave[0]}-{b_wave[-1]} nm."
        )

    common_wave = a_wave[(a_wave >= overlap_min) & (a_wave <= overlap_max)]
    if common_wave.size < 3:
        n_common = int(min(a_wave.size, b_wave.size))
        common_wave = np.linspace(overlap_min, overlap_max, max(3, n_common))

    a_common = np.interp(common_wave, a_wave, a_template, left=np.nan, right=np.nan)
    b_common = np.interp(common_wave, b_wave, b_template, left=np.nan, right=np.nan)
    finite = np.isfinite(common_wave) & np.isfinite(a_common) & np.isfinite(b_common)

    if np.sum(finite) < 3:
        pearson = float("nan")
        pearson_neg = float("nan")
        scale = float("nan")
        residual = np.full_like(common_wave, np.nan, dtype=float)
    else:
        a_fit = a_common[finite]
        b_fit = b_common[finite]
        if np.nanstd(a_fit) == 0 or np.nanstd(b_fit) == 0:
            pearson = float("nan")
        else:
            pearson = float(np.corrcoef(a_fit, b_fit)[0, 1])
        pearson_neg = float(-pearson) if np.isfinite(pearson) else float("nan")
        denom = float(np.nansum(b_fit * b_fit))
        scale = float(np.nansum(a_fit * b_fit) / denom) if denom > 0 else float("nan")
        residual = a_common - scale * b_common

    appears_sign_flipped = bool(
        np.isfinite(pearson)
        and np.isfinite(scale)
        and pearson < 0.0
        and scale < 0.0
        and abs(pearson) >= 0.2
    )

    return {
        "common_wavelength_nm": common_wave,
        "a_common": a_common,
        "b_common": b_common,
        "finite_mask": finite,
        "residual": residual,
        "summary": {
            "overlap_nm": [overlap_min, overlap_max],
            "n_common_pixels": int(common_wave.size),
            "n_common_finite_pixels": int(np.sum(finite)),
            "common_grid_source": str(record_a["label"]),
            "pearson_a_vs_b": pearson,
            "pearson_a_vs_negative_b": pearson_neg,
            "best_fit_scalar_b_to_a": scale,
            "residual_after_scaling": vector_stats(residual[finite] if np.any(finite) else residual),
            "appears_sign_flipped": appears_sign_flipped,
        },
    }


def save_common_arrays(output_dir: Path, common: Mapping[str, Any]) -> str:
    """Save common-grid interpolation products."""

    import numpy as np

    path = output_dir / "template_common_grid.npz"
    np.savez_compressed(
        path,
        wavelength_nm=np.asarray(common["common_wavelength_nm"], dtype=float),
        a_template=np.asarray(common["a_common"], dtype=float),
        b_template=np.asarray(common["b_common"], dtype=float),
        finite_mask=np.asarray(common["finite_mask"], dtype=bool),
        residual=np.asarray(common["residual"], dtype=float),
    )
    return str(path)


def plot_full_overlay(path: Path, record_a: Mapping[str, Any], record_b: Mapping[str, Any]) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    a_wave, a_template = finite_sorted_xy(record_a["template"]["model_wvl_nm"], record_a["template"]["model_dmag"])
    b_wave, b_template = finite_sorted_xy(record_b["template"]["model_wvl_nm"], record_b["template"]["model_dmag"])
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(a_wave, a_template, linewidth=0.6, alpha=0.8, label=str(record_a["label"]))
    ax.plot(b_wave, b_template, linewidth=0.6, alpha=0.8, label=str(record_b["label"]))
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Final HRCCS template [dmag]")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_zoom_chunks(path: Path, record_a: Mapping[str, Any], record_b: Mapping[str, Any], common: Mapping[str, Any]) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    wave = np.asarray(common["common_wavelength_nm"], dtype=float)
    finite_wave = wave[np.isfinite(wave)]
    if finite_wave.size < 3:
        return False
    span = float(finite_wave[-1] - finite_wave[0])
    width = max(1.0, min(5.0, span / 20.0))
    centers = np.quantile(finite_wave, [0.15, 0.35, 0.55, 0.75])
    a_wave, a_template = finite_sorted_xy(record_a["template"]["model_wvl_nm"], record_a["template"]["model_dmag"])
    b_wave, b_template = finite_sorted_xy(record_b["template"]["model_wvl_nm"], record_b["template"]["model_dmag"])

    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharey=True)
    for ax, center in zip(axes, centers):
        lo = float(center - 0.5 * width)
        hi = float(center + 0.5 * width)
        a_sel = (a_wave >= lo) & (a_wave <= hi)
        b_sel = (b_wave >= lo) & (b_wave <= hi)
        ax.plot(a_wave[a_sel], a_template[a_sel], linewidth=0.8, label=str(record_a["label"]))
        ax.plot(b_wave[b_sel], b_template[b_sel], linewidth=0.8, label=str(record_b["label"]))
        ax.set_xlim(lo, hi)
        ax.set_ylabel("dmag")
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Wavelength [nm]")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_scatter(path: Path, common: Mapping[str, Any], label_a: str, label_b: str) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    a = np.asarray(common["a_common"], dtype=float)
    b = np.asarray(common["b_common"], dtype=float)
    finite = np.asarray(common["finite_mask"], dtype=bool)
    if not np.any(finite):
        return False
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(a[finite], b[finite], s=2, alpha=0.25)
    ax.set_xlabel(f"{label_a} template [dmag]")
    ax.set_ylabel(f"{label_b} template [dmag]")
    ax.set_title("Common-grid template scatter")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_residual(path: Path, common: Mapping[str, Any], scale: float, label_a: str, label_b: str) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    wave = np.asarray(common["common_wavelength_nm"], dtype=float)
    residual = np.asarray(common["residual"], dtype=float)
    finite = np.isfinite(wave) & np.isfinite(residual)
    if not np.any(finite):
        return False
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(wave[finite], residual[finite], linewidth=0.6)
    ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel(f"{label_a} - ({scale:.6g}) * {label_b}")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def save_plots(output_dir: Path, record_a: Mapping[str, Any], record_b: Mapping[str, Any], common: Mapping[str, Any]) -> dict[str, str | None]:
    """Save overlay, zoom, scatter, and residual plots."""

    paths = {
        "full_overlay": output_dir / "template_overlay_full.png",
        "zoom_chunks": output_dir / "template_overlay_zoom_chunks.png",
        "scatter": output_dir / "template_scatter_common_grid.png",
        "residual": output_dir / "template_residual_scaled.png",
    }
    saved = {
        "full_overlay": str(paths["full_overlay"]) if plot_full_overlay(paths["full_overlay"], record_a, record_b) else None,
        "zoom_chunks": str(paths["zoom_chunks"]) if plot_zoom_chunks(paths["zoom_chunks"], record_a, record_b, common) else None,
        "scatter": str(paths["scatter"]) if plot_scatter(paths["scatter"], common, str(record_a["label"]), str(record_b["label"])) else None,
        "residual": str(paths["residual"])
        if plot_residual(
            paths["residual"],
            common,
            float(common["summary"]["best_fit_scalar_b_to_a"]),
            str(record_a["label"]),
            str(record_b["label"]),
        )
        else None,
    }
    return saved


def log_template_metadata(logger: Any, label: str, metadata: Mapping[str, Any]) -> None:
    """Print the key pRT/model-processing metadata to the run log/stdout."""

    logger.info("[%s] pressure grid: %s", label, metadata["pressure_grid"])
    logger.info("[%s] T-P profile: %s", label, metadata["temperature_profile_type"])
    logger.info("[%s] T-P report: %s", label, metadata["temperature_pressure_report"])
    logger.info("[%s] line species: %s", label, metadata["line_species"])
    logger.info(
        "[%s] continuum contributors YAML=%s Radtrans=%s",
        label,
        metadata["yaml_requested_continuum_contributors"],
        metadata["radtrans_continuum_contributors"],
    )
    logger.info("[%s] wavelength boundaries micron: %s", label, metadata["wavelength_boundaries_micron"])
    logger.info("[%s] model representation: %s", label, metadata["model_representation"])
    logger.info("[%s] xcorr processing settings: %s", label, metadata["xcorr_processing_settings"])
    logger.info("[%s] model-processing metadata: %s", label, metadata["model_processing_metadata"])


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir, "compare_hrccs_templates.log")

    import numpy as np

    exopipe_config, _ = load_project_modules(args.project_path)
    record_a = build_template_record(
        label=args.label_a,
        config_path=args.config_a,
        params_json=args.params_a_json,
        project_config=exopipe_config,
        logger=logger,
    )
    record_b = build_template_record(
        label=args.label_b,
        config_path=args.config_b,
        params_json=args.params_b_json,
        project_config=exopipe_config,
        logger=logger,
    )

    metadata_a = template_metadata(record_a)
    metadata_b = template_metadata(record_b)
    log_template_metadata(logger, str(args.label_a), metadata_a)
    log_template_metadata(logger, str(args.label_b), metadata_b)

    array_files = save_template_arrays(output_dir, record_a, record_b)
    common = common_grid_diagnostics(record_a, record_b)
    common_arrays_path = save_common_arrays(output_dir, common)
    plots = save_plots(output_dir, record_a, record_b, common)

    logger.info("Common-grid comparison summary: %s", common["summary"])
    if common["summary"]["appears_sign_flipped"]:
        logger.warning(
            "The final templates appear sign-flipped: corr(A,B)=%.4f, scale(B->A)=%.4g",
            common["summary"]["pearson_a_vs_b"],
            common["summary"]["best_fit_scalar_b_to_a"],
        )

    summary = {
        "project_path": str(args.project_path),
        "label_a": str(args.label_a),
        "label_b": str(args.label_b),
        "template_a": metadata_a,
        "template_b": metadata_b,
        "comparison": common["summary"],
        "output_files": {
            **array_files,
            "common_grid_arrays": common_arrays_path,
            "summary": str(output_dir / "template_comparison_summary.json"),
            "log": str(output_dir / "compare_hrccs_templates.log"),
            **plots,
        },
        "notes": [
            "The comparison uses the final HRCCS sampler template: model_dmag on model_wvl_nm.",
            "The two-column xcorr_processed model arrays are also saved before final convolution/template_to_dmag.",
            "Raw pRT arrays are saved only because this diagnostic requested include_raw_arrays=True; normal retrieval outputs are unchanged.",
        ],
    }
    write_json(output_dir / "template_comparison_summary.json", summary)

    print(
        json.dumps(
            {
                "pearson_a_vs_b": common["summary"]["pearson_a_vs_b"],
                "pearson_a_vs_negative_b": common["summary"]["pearson_a_vs_negative_b"],
                "best_fit_scalar_b_to_a": common["summary"]["best_fit_scalar_b_to_a"],
                "appears_sign_flipped": common["summary"]["appears_sign_flipped"],
                "output": str(output_dir),
            },
            indent=2,
            sort_keys=True,
            default=json_default,
        )
    )


if __name__ == "__main__":
    main()
