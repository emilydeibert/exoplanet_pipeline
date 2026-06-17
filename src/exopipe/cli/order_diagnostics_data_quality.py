from __future__ import annotations

from astropy.stats import sigma_clip
from pathlib import Path
import argparse
import importlib.util
import json
import csv
import warnings
from typing import Any

import numpy as np

try:
    import astropy.units as u
except ImportError:
    u = None


def as_float(value, unit=None):
    """Convert scalar or astropy Quantity to plain float."""
    if hasattr(value, "to_value"):
        if unit is not None:
            return float(value.to_value(unit))
        return float(value.value)
    return float(value)


def as_float_array(value, unit=None):
    """Convert array or astropy Quantity array to plain float numpy array."""
    if hasattr(value, "to_value"):
        if unit is not None:
            return np.asarray(value.to_value(unit), dtype=float)
        return np.asarray(value.value, dtype=float)
    return np.asarray(value, dtype=float)


# -----------------------------------------------------------------------------
# Philosophy
# -----------------------------------------------------------------------------
# This script is intentionally conservative about order selection.
#
# The recommended baseline order list is selected ONLY from data-quality metrics
# such as wavelength coverage, finite-pixel fractions, CCF finite coverage, and
# wavelength-column RMS/MAD outliers. Observed/delta CCF SNR values are reported
# as diagnostics only, not used to decide whether an order is included.
#
# This avoids the less defensible workflow of keeping/removing orders because
# they make the real planet detection stronger or weaker.


parser = argparse.ArgumentParser(
    description="Data-quality-first order diagnostics for MASCARA-1b/GHOST CCF products."
)

parser.add_argument("project_path", type=str)
parser.add_argument("--model", required=True)
parser.add_argument("--night", required=True)
parser.add_argument("--camera", required=True)
parser.add_argument("--k", type=int, required=True)

# Optional explicit map files. If not provided, the script uses the current
# pipeline naming convention.
parser.add_argument("--obs-file", default=None)
parser.add_argument("--positive-injection-file", default=None)
parser.add_argument("--negative-injection-file", default=None)
parser.add_argument("--analysis-ready-file", default=None)

# Optional spectral-column diagnostic file/key. If not supplied, the script
# attempts to use arrays in the analysis-ready file.
parser.add_argument("--column-data-file", default=None)
parser.add_argument("--column-data-key", default=None)

# Crop / expected location.
parser.add_argument("--rv-min", type=float, default=None)
parser.add_argument("--rv-max", type=float, default=None)
parser.add_argument("--kp-min", type=float, default=None)
parser.add_argument("--kp-max", type=float, default=None)
parser.add_argument("--expected-kp", type=float, default=None)
parser.add_argument("--expected-rv", type=float, default=None)
parser.add_argument("--kp-window", type=float, default=10.0)
parser.add_argument("--rv-window", type=float, default=5.0)
parser.add_argument("--aperture-kp-half-width", type=float, default=5.0)
parser.add_argument("--aperture-rv-half-width", type=float, default=3.0)
parser.add_argument("--edge-tolerance-pixels", type=int, default=2)

# Data-quality cuts for order inclusion.
parser.add_argument("--wavelength-min-nm", type=float, default=383.0)
parser.add_argument("--wavelength-max-nm", type=float, default=1000.0)
parser.add_argument("--min-overlap-fraction", type=float, default=0.5)
parser.add_argument("--min-valid-flux-fraction", type=float, default=0.5)
parser.add_argument("--min-ccf-finite-fraction", type=float, default=0.98)
parser.add_argument("--min-column-finite-fraction", type=float, default=0.8)
parser.add_argument("--column-rms-sigma", type=float, default=5.0)
parser.add_argument("--column-mad-sigma", type=float, default=5.0)
parser.add_argument("--max-bad-column-fraction", type=float, default=0.2)
parser.add_argument("--max-bad-edge-column-fraction", type=float, default=0.5)
parser.add_argument("--edge-pixels", type=int, default=20)

# Noise/SNR options.
parser.add_argument("--sigma-cut", type=float, default=3.0)
parser.add_argument(
    "--noise-method",
    choices=["clipped_std", "mad", "std"],
    default="mad",
    help="Noise estimator for CCF-map SNR diagnostics. MAD is usually safest for diagnostics.",
)
parser.add_argument(
    "--exclude-expected-window-from-noise",
    action="store_true",
    help="Exclude the expected planet window when estimating map noise.",
)
parser.add_argument("--map-sign", type=float, default=-1.0)

# Sign conventions for delta products.
parser.add_argument(
    "--pos-delta-peak-sign",
    choices=["negative", "positive", "absolute"],
    default="negative",
    help="Expected sign of positive-injection delta recovery after applying map_sign. "
         "The positive delta is defined as positive_injected_map - observed_map.",
)
parser.add_argument(
    "--neg-delta-peak-sign",
    choices=["negative", "positive", "absolute"],
    default="positive",
    help="Expected sign of negative-injection delta recovery after applying map_sign. "
         "The negative delta is defined as negative_injected_map - observed_map.",
)

parser.add_argument("--output-dir", default=None)

args = parser.parse_args()


# -----------------------------------------------------------------------------
# Loading helpers
# -----------------------------------------------------------------------------


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    spec.loader.exec_module(module)
    return module


project_path = Path(args.project_path)
config = load_module(project_path / "config.py", "config")
params = load_module(project_path / "parameters.py", "parameters")


# -----------------------------------------------------------------------------
# Generic array helpers
# -----------------------------------------------------------------------------


def finite_fraction(x: np.ndarray) -> float:
    if x.size == 0:
        return np.nan
    return float(np.mean(np.isfinite(x)))


def robust_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.nanmedian(x)
    return float(1.4826 * np.nanmedian(np.abs(x - med)))


def robust_median(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if not np.any(np.isfinite(x)):
        return np.nan
    return float(np.nanmedian(x))


def safe_ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return np.nan
    return float(a / b)


def nearest_index(values: np.ndarray, target: float) -> int:
    values = np.asarray(values, dtype=float)
    return int(np.nanargmin(np.abs(values - target)))


def mask_expected_window(
    shape: tuple[int, int],
    RV: np.ndarray,
    Kp: np.ndarray,
    expected_kp: float,
    expected_rv: float,
    kp_window: float,
    rv_window: float,
) -> np.ndarray:
    kp_mask = (Kp >= expected_kp - kp_window) & (Kp <= expected_kp + kp_window)
    rv_mask = (RV >= expected_rv - rv_window) & (RV <= expected_rv + rv_window)
    mask = np.zeros(shape, dtype=bool)
    mask[np.ix_(kp_mask, rv_mask)] = True
    return mask


# -----------------------------------------------------------------------------
# Order extraction helpers
# -----------------------------------------------------------------------------


def infer_order_axis(arr: np.ndarray, saved_orders: np.ndarray | None = None) -> int | None:
    """Infer which axis is order-like. Prefer an axis matching len(saved_orders)."""
    if arr.ndim < 2:
        return None

    if saved_orders is not None:
        n_orders = len(saved_orders)
        matches = [i for i, n in enumerate(arr.shape) if n == n_orders]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            # For spectral arrays, order axis is usually before exposure/pixel axes.
            return matches[0]

    # Fallback heuristics: GHOST order counts are usually much smaller than n_exp/n_pix.
    plausible = [i for i, n in enumerate(arr.shape) if 5 <= n <= 80]
    if len(plausible) == 1:
        return plausible[0]
    if len(plausible) > 1:
        return plausible[0]

    return None


def get_order_slice(arr: np.ndarray | None, row_idx: int, saved_orders: np.ndarray | None = None) -> np.ndarray | None:
    """Return arr for one order, using row_idx in the saved-order list.

    This avoids the dangerous ambiguity in code like arr[order], where `order`
    may accidentally index the exposure axis if the array shape is
    (n_exp, n_orders, n_pix).
    """
    if arr is None:
        return None

    arr = np.asarray(arr)

    if arr.ndim == 1:
        return arr

    axis = infer_order_axis(arr, saved_orders=saved_orders)
    if axis is None:
        return None

    if row_idx >= arr.shape[axis]:
        return None

    return np.take(arr, row_idx, axis=axis)


def get_order_wave_nm(wave: np.ndarray, row_idx: int, original_order: int, saved_orders: np.ndarray) -> np.ndarray:
    """Return 1-D wavelength vector for an order.

    Handles common shapes:
    - (n_orders, n_pix)
    - (n_orders, n_exp, n_pix)
    - (n_exp, n_orders, n_pix)
    """
    wave = np.asarray(wave)

    # First try row_idx-based slicing, because fmap rows correspond to saved_orders.
    order_wave = get_order_slice(wave, row_idx, saved_orders=saved_orders)

    # If that failed, fall back to original_order for older files where wave is
    # indexed by physical order number.
    if order_wave is None and wave.ndim >= 2:
        try:
            order_wave = np.take(wave, original_order, axis=0)
        except Exception:
            order_wave = None

    if order_wave is None:
        raise ValueError(f"Could not extract wavelength array for order {original_order}")

    order_wave = np.asarray(order_wave, dtype=float)

    if order_wave.ndim == 1:
        return order_wave
    if order_wave.ndim == 2:
        # Usually n_exp x n_pix. Use the median wavelength grid over exposures.
        return np.nanmedian(order_wave, axis=0)

    raise ValueError(f"Unexpected order wavelength shape for order {original_order}: {order_wave.shape}")


# -----------------------------------------------------------------------------
# Wavelength / column diagnostics
# -----------------------------------------------------------------------------


def overlap_fraction(order_wave: np.ndarray, wmin: float, wmax: float) -> float:
    finite = np.isfinite(order_wave)
    if np.sum(finite) == 0:
        return 0.0
    usable = finite & (order_wave >= wmin) & (order_wave <= wmax)
    return float(np.sum(usable) / np.sum(finite))


def choose_column_diagnostic_array(npz_data: dict[str, np.ndarray], requested_key: str | None) -> tuple[str | None, np.ndarray | None]:
    if requested_key is not None:
        if requested_key not in npz_data:
            raise KeyError(f"Requested column-data key {requested_key!r} not found. Available: {list(npz_data.keys())}")
        return requested_key, npz_data[requested_key]

    # Preference order: use post-SYSREM/residual arrays if present; otherwise use
    # normalized flux as a fallback. The fallback is still useful for finite-column
    # and gross-noise diagnostics, but it is not a true post-SYSREM RMS.
    candidate_keys = [
        "sysrem_residuals",
        "sysrem_resid",
        "residuals_sysrem",
        "residuals",
        "cleaned_flux",
        "flux_sysrem",
        "norm_flux_sysrem",
        "norm_flux",
        "flux",
    ]

    for key in candidate_keys:
        if key in npz_data:
            return key, npz_data[key]

    return None, None


def column_metrics_for_order(
    order_data: np.ndarray | None,
    order_wave: np.ndarray,
    original_order: int,
    edge_pixels: int,
    min_column_finite_fraction: float,
    rms_sigma: float,
    mad_sigma: float,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Return per-column rows and order-level column quality metrics.

    order_data should usually have shape (n_exp, n_pix). If unavailable or not
    interpretable, return NaNs rather than failing.
    """
    n_pix = len(order_wave)

    if order_data is None:
        summary = {
            "column_data_available": False,
            "bad_column_fraction": np.nan,
            "bad_edge_column_fraction": np.nan,
            "median_column_rms": np.nan,
            "median_column_mad": np.nan,
        }
        return [], summary

    order_data = np.asarray(order_data, dtype=float)

    # Reduce to something shaped exposure/time x wavelength-pixel.
    if order_data.ndim == 1:
        # A single spectrum is not useful for time-column RMS, but finite fraction
        # can still be reported.
        data2d = order_data[None, :]
    elif order_data.ndim == 2:
        data2d = order_data
    else:
        # Collapse all non-pixel axes into a single sample axis, assuming final
        # axis is wavelength/pixel after order extraction.
        data2d = order_data.reshape(-1, order_data.shape[-1])

    if data2d.shape[-1] != n_pix:
        # Try transposed orientation.
        if data2d.shape[0] == n_pix:
            data2d = data2d.T
        else:
            warnings.warn(
                f"Column data for order {original_order} has shape {data2d.shape}, "
                f"which does not match wavelength length {n_pix}. Column diagnostics skipped."
            )
            summary = {
                "column_data_available": False,
                "bad_column_fraction": np.nan,
                "bad_edge_column_fraction": np.nan,
                "median_column_rms": np.nan,
                "median_column_mad": np.nan,
            }
            return [], summary

    col_finite = np.mean(np.isfinite(data2d), axis=0)
    col_med = np.nanmedian(data2d, axis=0)

    centered = data2d - col_med[None, :]
    col_rms = np.nanstd(centered, axis=0)
    col_mad = np.array([robust_mad(centered[:, j]) for j in range(centered.shape[1])])

    med_rms = robust_median(col_rms)
    mad_of_rms = robust_mad(col_rms)
    med_mad = robust_median(col_mad)
    mad_of_mad = robust_mad(col_mad)

    if not np.isfinite(mad_of_rms) or mad_of_rms == 0:
        rms_outlier = np.zeros(n_pix, dtype=bool)
    else:
        rms_outlier = col_rms > med_rms + rms_sigma * mad_of_rms

    if not np.isfinite(mad_of_mad) or mad_of_mad == 0:
        mad_outlier = np.zeros(n_pix, dtype=bool)
    else:
        mad_outlier = col_mad > med_mad + mad_sigma * mad_of_mad

    finite_bad = col_finite < min_column_finite_fraction

    is_edge = np.zeros(n_pix, dtype=bool)
    edge_n = min(edge_pixels, n_pix // 2)
    if edge_n > 0:
        is_edge[:edge_n] = True
        is_edge[-edge_n:] = True

    bad_column = finite_bad | rms_outlier | mad_outlier

    rows = []
    for pixel in range(n_pix):
        rows.append(
            {
                "original_order": int(original_order),
                "pixel": int(pixel),
                "wavelength_nm": float(order_wave[pixel]) if np.isfinite(order_wave[pixel]) else np.nan,
                "column_finite_fraction": float(col_finite[pixel]),
                "column_median": float(col_med[pixel]) if np.isfinite(col_med[pixel]) else np.nan,
                "column_rms": float(col_rms[pixel]) if np.isfinite(col_rms[pixel]) else np.nan,
                "column_mad": float(col_mad[pixel]) if np.isfinite(col_mad[pixel]) else np.nan,
                "is_edge_pixel": bool(is_edge[pixel]),
                "bad_finite_column": bool(finite_bad[pixel]),
                "bad_rms_column": bool(rms_outlier[pixel]),
                "bad_mad_column": bool(mad_outlier[pixel]),
                "bad_column": bool(bad_column[pixel]),
            }
        )

    if np.any(is_edge):
        bad_edge_column_fraction = float(np.mean(bad_column[is_edge]))
    else:
        bad_edge_column_fraction = 0.0

    summary = {
        "column_data_available": True,
        "bad_column_fraction": float(np.mean(bad_column)),
        "bad_edge_column_fraction": bad_edge_column_fraction,
        "median_column_rms": med_rms,
        "median_column_mad": med_mad,
    }

    return rows, summary


# -----------------------------------------------------------------------------
# CCF map metrics
# -----------------------------------------------------------------------------


def calculate_noise(
    noise_map: np.ndarray,
    RV: np.ndarray | None = None,
    Kp: np.ndarray | None = None,
    expected_kp: float | None = None,
    expected_rv: float | None = None,
    kp_window: float | None = None,
    rv_window: float | None = None,
    method: str = "mad",
    sigma_cut: float = 3.0,
    exclude_expected_window: bool = False,
) -> float:
    x = np.asarray(noise_map, dtype=float) - np.nanmedian(noise_map)

    if exclude_expected_window:
        if RV is None or Kp is None or expected_kp is None or expected_rv is None:
            raise ValueError("RV, Kp, expected_kp, and expected_rv are required to exclude expected window")
        bad = mask_expected_window(
            x.shape,
            RV=RV,
            Kp=Kp,
            expected_kp=expected_kp,
            expected_rv=expected_rv,
            kp_window=kp_window if kp_window is not None else 0,
            rv_window=rv_window if rv_window is not None else 0,
        )
        x = x.copy()
        x[bad] = np.nan

    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.nan

    if method == "mad":
        return robust_mad(finite)

    if method == "std":
        return float(np.nanstd(finite))

    if method == "clipped_std":
        clipped = sigma_clip(
            finite,
            sigma_upper=sigma_cut,
            sigma_lower=sigma_cut,
        )
        return float(np.nanstd(clipped))

    raise ValueError(method)


def snr_map_from_noise(signal_map: np.ndarray, noise: float) -> np.ndarray:
    if not np.isfinite(noise) or noise == 0:
        return np.full_like(signal_map, np.nan, dtype=float)
    x = np.asarray(signal_map, dtype=float) - np.nanmedian(signal_map)
    return x / noise


def find_peak(snr_map: np.ndarray, RV: np.ndarray, Kp: np.ndarray) -> dict[str, Any]:
    if not np.any(np.isfinite(snr_map)):
        return {
            "snr": np.nan,
            "rv": np.nan,
            "kp": np.nan,
            "rv_idx": -1,
            "kp_idx": -1,
        }

    max_index = np.nanargmax(snr_map)
    kp_idx, rv_idx = np.unravel_index(max_index, snr_map.shape)

    return {
        "snr": float(snr_map[kp_idx, rv_idx]),
        "rv": float(RV[rv_idx]),
        "kp": float(Kp[kp_idx]),
        "rv_idx": int(rv_idx),
        "kp_idx": int(kp_idx),
    }


def find_peak_near_expected(
    snr_map: np.ndarray,
    RV: np.ndarray,
    Kp: np.ndarray,
    expected_kp: float,
    expected_rv: float,
    kp_window: float,
    rv_window: float,
) -> dict[str, Any]:
    kp_mask = (Kp >= expected_kp - kp_window) & (Kp <= expected_kp + kp_window)
    rv_mask = (RV >= expected_rv - rv_window) & (RV <= expected_rv + rv_window)

    masked = np.full_like(snr_map, np.nan, dtype=float)
    masked[np.ix_(kp_mask, rv_mask)] = snr_map[np.ix_(kp_mask, rv_mask)]

    return find_peak(masked, RV, Kp)


def aperture_mask(
    RV: np.ndarray,
    Kp: np.ndarray,
    expected_kp: float,
    expected_rv: float,
    kp_half_width: float,
    rv_half_width: float,
) -> np.ndarray:
    kp_mask = (Kp >= expected_kp - kp_half_width) & (Kp <= expected_kp + kp_half_width)
    rv_mask = (RV >= expected_rv - rv_half_width) & (RV <= expected_rv + rv_half_width)
    mask = np.zeros((len(Kp), len(RV)), dtype=bool)
    mask[np.ix_(kp_mask, rv_mask)] = True
    return mask


def is_peak_near_expected(
    peak: dict[str, Any],
    expected_kp: float,
    expected_rv: float,
    kp_window: float,
    rv_window: float,
) -> bool:
    if not np.isfinite(peak["kp"]) or not np.isfinite(peak["rv"]):
        return False
    return (
        abs(peak["kp"] - expected_kp) <= kp_window
        and abs(peak["rv"] - expected_rv) <= rv_window
    )


def is_peak_near_edge(peak: dict[str, Any], shape: tuple[int, int], edge_tolerance_pixels: int) -> bool:
    if peak["kp_idx"] < 0 or peak["rv_idx"] < 0:
        return False
    n_kp, n_rv = shape
    return (
        peak["kp_idx"] <= edge_tolerance_pixels
        or peak["kp_idx"] >= n_kp - 1 - edge_tolerance_pixels
        or peak["rv_idx"] <= edge_tolerance_pixels
        or peak["rv_idx"] >= n_rv - 1 - edge_tolerance_pixels
    )


def recovery_map(delta_snr_map: np.ndarray, sign: str) -> np.ndarray:
    if sign == "negative":
        return -1.0 * delta_snr_map
    if sign == "positive":
        return delta_snr_map
    if sign == "absolute":
        return np.abs(delta_snr_map)
    raise ValueError(sign)


def metrics_for_map(
    signal_map: np.ndarray,
    noise_map: np.ndarray,
    RV: np.ndarray,
    Kp: np.ndarray,
    expected_kp: float,
    expected_rv: float,
    kp_window: float,
    rv_window: float,
    aperture_kp_half_width: float,
    aperture_rv_half_width: float,
    sigma_cut: float,
    noise_method: str,
    exclude_expected_window_from_noise: bool,
    edge_tolerance_pixels: int,
    fixed_noise: float | None = None,
) -> dict[str, Any]:
    if fixed_noise is None:
        noise = calculate_noise(
            noise_map,
            RV=RV,
            Kp=Kp,
            expected_kp=expected_kp,
            expected_rv=expected_rv,
            kp_window=kp_window,
            rv_window=rv_window,
            method=noise_method,
            sigma_cut=sigma_cut,
            exclude_expected_window=exclude_expected_window_from_noise,
        )
    else:
        noise = fixed_noise

    snr_map = snr_map_from_noise(signal_map, noise)

    expected_peak = find_peak_near_expected(
        snr_map,
        RV,
        Kp,
        expected_kp,
        expected_rv,
        kp_window,
        rv_window,
    )

    global_peak = find_peak(snr_map, RV, Kp)

    fixed_kp_idx = nearest_index(Kp, expected_kp)
    fixed_rv_idx = nearest_index(RV, expected_rv)

    fixed_raw = float(signal_map[fixed_kp_idx, fixed_rv_idx]) if np.isfinite(signal_map[fixed_kp_idx, fixed_rv_idx]) else np.nan
    fixed_snr = float(snr_map[fixed_kp_idx, fixed_rv_idx]) if np.isfinite(snr_map[fixed_kp_idx, fixed_rv_idx]) else np.nan

    ap_mask = aperture_mask(
        RV,
        Kp,
        expected_kp,
        expected_rv,
        aperture_kp_half_width,
        aperture_rv_half_width,
    )

    ap_raw_values = signal_map[ap_mask]
    ap_snr_values = snr_map[ap_mask]

    aperture_raw_mean = float(np.nanmean(ap_raw_values)) if np.any(np.isfinite(ap_raw_values)) else np.nan
    aperture_raw_sum = float(np.nansum(ap_raw_values)) if np.any(np.isfinite(ap_raw_values)) else np.nan
    aperture_snr_mean = float(np.nanmean(ap_snr_values)) if np.any(np.isfinite(ap_snr_values)) else np.nan
    aperture_snr_sum = float(np.nansum(ap_snr_values)) if np.any(np.isfinite(ap_snr_values)) else np.nan

    return {
        "noise": float(noise) if np.isfinite(noise) else np.nan,
        "fixed_raw": fixed_raw,
        "fixed_snr": fixed_snr,
        "aperture_raw_mean": aperture_raw_mean,
        "aperture_raw_sum": aperture_raw_sum,
        "aperture_snr_mean": aperture_snr_mean,
        "aperture_snr_sum": aperture_snr_sum,
        "expected_peak_snr": expected_peak["snr"],
        "expected_peak_kp": expected_peak["kp"],
        "expected_peak_rv": expected_peak["rv"],
        "global_snr": global_peak["snr"],
        "global_kp": global_peak["kp"],
        "global_rv": global_peak["rv"],
        "global_near_expected": bool(is_peak_near_expected(global_peak, expected_kp, expected_rv, kp_window, rv_window)),
        "global_edge_flag": bool(is_peak_near_edge(global_peak, snr_map.shape, edge_tolerance_pixels)),
        "global_to_expected_ratio": safe_ratio(global_peak["snr"], expected_peak["snr"]),
    }


def prefixed(prefix: str, d: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{k}": v for k, v in d.items()}


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------


def main():
    RV = np.asarray(config.RV, dtype=float)
    Kp = np.asarray(config.Kp, dtype=float)

    RV_MIN = args.rv_min if args.rv_min is not None else (config.RV_MIN if hasattr(config, "RV_MIN") else -75)
    RV_MAX = args.rv_max if args.rv_max is not None else (config.RV_MAX if hasattr(config, "RV_MAX") else 75)
    KP_MIN = args.kp_min if args.kp_min is not None else (config.KP_MIN if hasattr(config, "KP_MIN") else np.nanmin(Kp))
    KP_MAX = args.kp_max if args.kp_max is not None else (config.KP_MAX if hasattr(config, "KP_MAX") else np.nanmax(Kp))

    rv_mask = (RV >= RV_MIN) & (RV <= RV_MAX)
    kp_mask = (Kp >= KP_MIN) & (Kp <= KP_MAX)

    RV_crop = RV[rv_mask]
    Kp_crop = Kp[kp_mask]

    expected_kp = (
        args.expected_kp
        if args.expected_kp is not None
        else as_float(params.K_p, "km/s")
    )

    expected_rv = (
        args.expected_rv
        if args.expected_rv is not None
        else as_float(params.Vsys, "km/s")
    )

    base = Path(config.path2reduced)

    obs_file = Path(args.obs_file) if args.obs_file is not None else base / "results" / f"{args.night}_{args.camera}_{args.model}_k{args.k}_iters.npz"
    pos_file = Path(args.positive_injection_file) if args.positive_injection_file is not None else base / "injected" / f"{args.night}_{args.camera}_{args.model}_{args.k}_iters_injected_positive.npz"
    neg_file = Path(args.negative_injection_file) if args.negative_injection_file is not None else base / "injected" / f"{args.night}_{args.camera}_{args.model}_{args.k}_iters_injected_negative.npz"
    analysis_ready_file = Path(args.analysis_ready_file) if args.analysis_ready_file is not None else base / f"{args.night}_{args.camera}_analysis_ready.npz"

    if not obs_file.exists():
        raise FileNotFoundError(obs_file)
    if not analysis_ready_file.exists():
        raise FileNotFoundError(analysis_ready_file)

    with np.load(obs_file) as data:
        obs_fmap = np.asarray(data["fmap"], dtype=float)
        saved_orders = np.asarray(data["orders"], dtype=int) if "orders" in data.files else np.arange(obs_fmap.shape[0], dtype=int)

    has_positive_injection = pos_file.exists()
    has_negative_injection = neg_file.exists()

    if has_positive_injection:
        with np.load(pos_file) as data:
            pos_fmap = np.asarray(data["fmap"], dtype=float)
    else:
        pos_fmap = None
        print(f"Warning: positive injection file not found: {pos_file}")

    if has_negative_injection:
        with np.load(neg_file) as data:
            neg_fmap = np.asarray(data["fmap"], dtype=float)
    else:
        neg_fmap = None
        print(f"Warning: negative injection file not found: {neg_file}")

    with np.load(analysis_ready_file) as data:
        wave = np.asarray(data["wave"], dtype=float)

        flux = None
        for flux_key in ["norm_flux", "flux"]:
            if flux_key in data.files:
                flux = np.asarray(data[flux_key], dtype=float)
                break

        analysis_ready_arrays = {key: np.asarray(data[key]) for key in data.files}

    # Column diagnostics can come from a separate file or analysis-ready file.
    if args.column_data_file is not None:
        with np.load(args.column_data_file) as data:
            column_arrays = {key: np.asarray(data[key]) for key in data.files}
        column_data_key, column_data = choose_column_diagnostic_array(column_arrays, args.column_data_key)
        column_data_source = str(args.column_data_file)
    else:
        column_data_key, column_data = choose_column_diagnostic_array(analysis_ready_arrays, args.column_data_key)
        column_data_source = str(analysis_ready_file)

    if args.output_dir is None:
        output_dir = base / "results" / f"order_diagnostics_data_quality_{args.night}_{args.camera}_{args.model}_k{args.k}"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    column_rows_all: list[dict[str, Any]] = []

    for row_idx, original_order in enumerate(saved_orders):
        original_order = int(original_order)

        order_wave = get_order_wave_nm(wave, row_idx, original_order, saved_orders)
        finite_wave = np.isfinite(order_wave)

        wave_min = float(np.nanmin(order_wave)) if np.any(finite_wave) else np.nan
        wave_max = float(np.nanmax(order_wave)) if np.any(finite_wave) else np.nan
        wave_median = float(np.nanmedian(order_wave)) if np.any(finite_wave) else np.nan

        overlap = overlap_fraction(order_wave, args.wavelength_min_nm, args.wavelength_max_nm)
        include_wavelength = overlap >= args.min_overlap_fraction

        order_flux = get_order_slice(flux, row_idx, saved_orders=saved_orders)
        valid_flux_fraction = finite_fraction(order_flux) if order_flux is not None else np.nan
        include_flux_quality = True if not np.isfinite(valid_flux_fraction) else valid_flux_fraction >= args.min_valid_flux_fraction

        obs_order_map_full = args.map_sign * obs_fmap[row_idx]
        obs_order_crop = obs_order_map_full[kp_mask][:, rv_mask]
        ccf_finite_fraction_full = finite_fraction(obs_order_map_full)
        ccf_finite_fraction_crop = finite_fraction(obs_order_crop)
        include_ccf_finite = ccf_finite_fraction_crop >= args.min_ccf_finite_fraction

        # Column-level quality diagnostics. These are independent of CCF signal strength.
        order_column_data = get_order_slice(column_data, row_idx, saved_orders=saved_orders)
        column_rows, column_summary = column_metrics_for_order(
            order_data=order_column_data,
            order_wave=order_wave,
            original_order=original_order,
            edge_pixels=args.edge_pixels,
            min_column_finite_fraction=args.min_column_finite_fraction,
            rms_sigma=args.column_rms_sigma,
            mad_sigma=args.column_mad_sigma,
        )
        column_rows_all.extend(column_rows)

        if column_summary["column_data_available"]:
            include_column_quality = (
                column_summary["bad_column_fraction"] <= args.max_bad_column_fraction
                and column_summary["bad_edge_column_fraction"] <= args.max_bad_edge_column_fraction
            )
        else:
            # Do not exclude if column data are absent; report this explicitly.
            include_column_quality = True

        # This is the ONLY baseline-inclusion logic. No observed/delta SNR enters.
        include_data_quality = bool(
            include_wavelength
            and include_flux_quality
            and include_ccf_finite
            and include_column_quality
        )

        data_quality_reasons = []
        if not include_wavelength:
            data_quality_reasons.append("outside_wavelength_range")
        if not include_flux_quality:
            data_quality_reasons.append("low_valid_flux_fraction")
        if not include_ccf_finite:
            data_quality_reasons.append("low_ccf_finite_fraction")
        if not include_column_quality:
            data_quality_reasons.append("bad_column_quality")

        obs_metrics = metrics_for_map(
            signal_map=obs_order_crop,
            noise_map=obs_order_crop,
            RV=RV_crop,
            Kp=Kp_crop,
            expected_kp=expected_kp,
            expected_rv=expected_rv,
            kp_window=args.kp_window,
            rv_window=args.rv_window,
            aperture_kp_half_width=args.aperture_kp_half_width,
            aperture_rv_half_width=args.aperture_rv_half_width,
            sigma_cut=args.sigma_cut,
            noise_method=args.noise_method,
            exclude_expected_window_from_noise=args.exclude_expected_window_from_noise,
            edge_tolerance_pixels=args.edge_tolerance_pixels,
        )

        row = {
            "row_idx": int(row_idx),
            "original_order": original_order,
            "wave_min_nm": wave_min,
            "wave_max_nm": wave_max,
            "wave_median_nm": wave_median,
            "usable_overlap_fraction": overlap,
            "valid_flux_fraction": valid_flux_fraction,
            "ccf_finite_fraction_full": ccf_finite_fraction_full,
            "ccf_finite_fraction_crop": ccf_finite_fraction_crop,
            "column_data_available": bool(column_summary["column_data_available"]),
            "bad_column_fraction": column_summary["bad_column_fraction"],
            "bad_edge_column_fraction": column_summary["bad_edge_column_fraction"],
            "median_column_rms": column_summary["median_column_rms"],
            "median_column_mad": column_summary["median_column_mad"],
            "include_wavelength": bool(include_wavelength),
            "include_flux_quality": bool(include_flux_quality),
            "include_ccf_finite": bool(include_ccf_finite),
            "include_column_quality": bool(include_column_quality),
            "include_data_quality": bool(include_data_quality),
            "data_quality_reasons": ";".join(data_quality_reasons),
        }

        row.update(prefixed("single_obs", obs_metrics))

        if has_positive_injection:
            pos_order_crop = (args.map_sign * pos_fmap[row_idx])[kp_mask][:, rv_mask]
            pos_delta_crop = pos_order_crop - obs_order_crop

            # Use observed-map noise for delta SNR by default.
            noise_obs_order = calculate_noise(
                obs_order_crop,
                RV=RV_crop,
                Kp=Kp_crop,
                expected_kp=expected_kp,
                expected_rv=expected_rv,
                kp_window=args.kp_window,
                rv_window=args.rv_window,
                method=args.noise_method,
                sigma_cut=args.sigma_cut,
                exclude_expected_window=args.exclude_expected_window_from_noise,
            )
            pos_delta_snr = snr_map_from_noise(pos_delta_crop, noise_obs_order)
            pos_delta_rec = recovery_map(pos_delta_snr, args.pos_delta_peak_sign)

            pos_delta_metrics = metrics_for_map(
                signal_map=pos_delta_rec,
                noise_map=pos_delta_rec,
                RV=RV_crop,
                Kp=Kp_crop,
                expected_kp=expected_kp,
                expected_rv=expected_rv,
                kp_window=args.kp_window,
                rv_window=args.rv_window,
                aperture_kp_half_width=args.aperture_kp_half_width,
                aperture_rv_half_width=args.aperture_rv_half_width,
                sigma_cut=args.sigma_cut,
                noise_method="std",  # rec map is already SNR-like; keep metrics simple.
                exclude_expected_window_from_noise=False,
                edge_tolerance_pixels=args.edge_tolerance_pixels,
                fixed_noise=1.0,
            )
            row.update(prefixed("single_pos_delta", pos_delta_metrics))
        else:
            row.update(prefixed("single_pos_delta", nan_metrics()))

        if has_negative_injection:
            neg_order_crop = (args.map_sign * neg_fmap[row_idx])[kp_mask][:, rv_mask]
            neg_delta_crop = neg_order_crop - obs_order_crop

            noise_obs_order = calculate_noise(
                obs_order_crop,
                RV=RV_crop,
                Kp=Kp_crop,
                expected_kp=expected_kp,
                expected_rv=expected_rv,
                kp_window=args.kp_window,
                rv_window=args.rv_window,
                method=args.noise_method,
                sigma_cut=args.sigma_cut,
                exclude_expected_window=args.exclude_expected_window_from_noise,
            )
            neg_delta_snr = snr_map_from_noise(neg_delta_crop, noise_obs_order)
            neg_delta_rec = recovery_map(neg_delta_snr, args.neg_delta_peak_sign)

            neg_delta_metrics = metrics_for_map(
                signal_map=neg_delta_rec,
                noise_map=neg_delta_rec,
                RV=RV_crop,
                Kp=Kp_crop,
                expected_kp=expected_kp,
                expected_rv=expected_rv,
                kp_window=args.kp_window,
                rv_window=args.rv_window,
                aperture_kp_half_width=args.aperture_kp_half_width,
                aperture_rv_half_width=args.aperture_rv_half_width,
                sigma_cut=args.sigma_cut,
                noise_method="std",
                exclude_expected_window_from_noise=False,
                edge_tolerance_pixels=args.edge_tolerance_pixels,
                fixed_noise=1.0,
            )
            row.update(prefixed("single_neg_delta", neg_delta_metrics))
        else:
            row.update(prefixed("single_neg_delta", nan_metrics()))

        rows.append(row)

    baseline_positions = np.array(
        [r["row_idx"] for r in rows if r["include_data_quality"]],
        dtype=int,
    )

    if len(baseline_positions) == 0:
        raise ValueError("No baseline orders survived data-quality cuts.")

    def sum_orders(fmap: np.ndarray, positions: np.ndarray) -> np.ndarray:
        # Important: nansum can hide all-NaN regions. We therefore also report
        # finite fractions above. For the actual combined map, nansum is kept to
        # match the existing pipeline convention.
        return args.map_sign * np.nansum(fmap[positions], axis=0)

    obs_baseline = sum_orders(obs_fmap, baseline_positions)
    obs_baseline_crop = obs_baseline[kp_mask][:, rv_mask]

    baseline_obs_noise = calculate_noise(
        obs_baseline_crop,
        RV=RV_crop,
        Kp=Kp_crop,
        expected_kp=expected_kp,
        expected_rv=expected_rv,
        kp_window=args.kp_window,
        rv_window=args.rv_window,
        method=args.noise_method,
        sigma_cut=args.sigma_cut,
        exclude_expected_window=args.exclude_expected_window_from_noise,
    )

    obs_baseline_metrics = metrics_for_map(
        signal_map=obs_baseline_crop,
        noise_map=obs_baseline_crop,
        RV=RV_crop,
        Kp=Kp_crop,
        expected_kp=expected_kp,
        expected_rv=expected_rv,
        kp_window=args.kp_window,
        rv_window=args.rv_window,
        aperture_kp_half_width=args.aperture_kp_half_width,
        aperture_rv_half_width=args.aperture_rv_half_width,
        sigma_cut=args.sigma_cut,
        noise_method=args.noise_method,
        exclude_expected_window_from_noise=args.exclude_expected_window_from_noise,
        edge_tolerance_pixels=args.edge_tolerance_pixels,
    )

    if has_positive_injection:
        pos_baseline = sum_orders(pos_fmap, baseline_positions)
        pos_baseline_crop = pos_baseline[kp_mask][:, rv_mask]
        pos_delta_baseline_crop = pos_baseline_crop - obs_baseline_crop
        pos_delta_baseline_snr = snr_map_from_noise(pos_delta_baseline_crop, baseline_obs_noise)
        pos_delta_baseline_rec = recovery_map(pos_delta_baseline_snr, args.pos_delta_peak_sign)
        pos_delta_baseline_metrics = metrics_for_map(
            signal_map=pos_delta_baseline_rec,
            noise_map=pos_delta_baseline_rec,
            RV=RV_crop,
            Kp=Kp_crop,
            expected_kp=expected_kp,
            expected_rv=expected_rv,
            kp_window=args.kp_window,
            rv_window=args.rv_window,
            aperture_kp_half_width=args.aperture_kp_half_width,
            aperture_rv_half_width=args.aperture_rv_half_width,
            sigma_cut=args.sigma_cut,
            noise_method="std",
            exclude_expected_window_from_noise=False,
            edge_tolerance_pixels=args.edge_tolerance_pixels,
            fixed_noise=1.0,
        )
    else:
        pos_delta_baseline_metrics = nan_metrics()

    if has_negative_injection:
        neg_baseline = sum_orders(neg_fmap, baseline_positions)
        neg_baseline_crop = neg_baseline[kp_mask][:, rv_mask]
        neg_delta_baseline_crop = neg_baseline_crop - obs_baseline_crop
        neg_delta_baseline_snr = snr_map_from_noise(neg_delta_baseline_crop, baseline_obs_noise)
        neg_delta_baseline_rec = recovery_map(neg_delta_baseline_snr, args.neg_delta_peak_sign)
        neg_delta_baseline_metrics = metrics_for_map(
            signal_map=neg_delta_baseline_rec,
            noise_map=neg_delta_baseline_rec,
            RV=RV_crop,
            Kp=Kp_crop,
            expected_kp=expected_kp,
            expected_rv=expected_rv,
            kp_window=args.kp_window,
            rv_window=args.rv_window,
            aperture_kp_half_width=args.aperture_kp_half_width,
            aperture_rv_half_width=args.aperture_rv_half_width,
            sigma_cut=args.sigma_cut,
            noise_method="std",
            exclude_expected_window_from_noise=False,
            edge_tolerance_pixels=args.edge_tolerance_pixels,
            fixed_noise=1.0,
        )
    else:
        neg_delta_baseline_metrics = nan_metrics()

    # Leave-one-out diagnostics. These are diagnostics only, not order-selection
    # criteria unless they point back to a data-quality issue.
    for r in rows:
        if not r["include_data_quality"]:
            r.update(prefixed("loo_obs", nan_metrics()))
            r.update(prefixed("loo_obs_common_noise", nan_metrics()))
            r.update(prefixed("loo_pos_delta", nan_metrics()))
            r.update(prefixed("loo_pos_delta_common_noise", nan_metrics()))
            r.update(prefixed("loo_neg_delta", nan_metrics()))
            r.update(prefixed("loo_neg_delta_common_noise", nan_metrics()))
            r["loo_obs_expected_minus_baseline"] = np.nan
            r["loo_pos_delta_expected_minus_baseline"] = np.nan
            r["loo_neg_delta_expected_minus_baseline"] = np.nan
            r["removed_order_obs_fixed_raw_contribution"] = np.nan
            r["removed_order_pos_delta_fixed_raw_contribution"] = np.nan
            r["removed_order_neg_delta_fixed_raw_contribution"] = np.nan
            continue

        leave_out = int(r["row_idx"])
        loo_positions = np.array([p for p in baseline_positions if p != leave_out], dtype=int)

        obs_loo = sum_orders(obs_fmap, loo_positions)
        obs_loo_crop = obs_loo[kp_mask][:, rv_mask]

        loo_obs_metrics = metrics_for_map(
            signal_map=obs_loo_crop,
            noise_map=obs_loo_crop,
            RV=RV_crop,
            Kp=Kp_crop,
            expected_kp=expected_kp,
            expected_rv=expected_rv,
            kp_window=args.kp_window,
            rv_window=args.rv_window,
            aperture_kp_half_width=args.aperture_kp_half_width,
            aperture_rv_half_width=args.aperture_rv_half_width,
            sigma_cut=args.sigma_cut,
            noise_method=args.noise_method,
            exclude_expected_window_from_noise=args.exclude_expected_window_from_noise,
            edge_tolerance_pixels=args.edge_tolerance_pixels,
        )
        loo_obs_common_noise_metrics = metrics_for_map(
            signal_map=obs_loo_crop,
            noise_map=obs_loo_crop,
            RV=RV_crop,
            Kp=Kp_crop,
            expected_kp=expected_kp,
            expected_rv=expected_rv,
            kp_window=args.kp_window,
            rv_window=args.rv_window,
            aperture_kp_half_width=args.aperture_kp_half_width,
            aperture_rv_half_width=args.aperture_rv_half_width,
            sigma_cut=args.sigma_cut,
            noise_method=args.noise_method,
            exclude_expected_window_from_noise=args.exclude_expected_window_from_noise,
            edge_tolerance_pixels=args.edge_tolerance_pixels,
            fixed_noise=baseline_obs_noise,
        )

        r.update(prefixed("loo_obs", loo_obs_metrics))
        r.update(prefixed("loo_obs_common_noise", loo_obs_common_noise_metrics))
        r["loo_obs_expected_minus_baseline"] = loo_obs_metrics["expected_peak_snr"] - obs_baseline_metrics["expected_peak_snr"]
        r["removed_order_obs_fixed_raw_contribution"] = obs_baseline_metrics["fixed_raw"] - loo_obs_metrics["fixed_raw"]

        if has_positive_injection:
            pos_loo = sum_orders(pos_fmap, loo_positions)
            pos_loo_crop = pos_loo[kp_mask][:, rv_mask]
            pos_delta_loo_crop = pos_loo_crop - obs_loo_crop

            loo_obs_noise = calculate_noise(
                obs_loo_crop,
                RV=RV_crop,
                Kp=Kp_crop,
                expected_kp=expected_kp,
                expected_rv=expected_rv,
                kp_window=args.kp_window,
                rv_window=args.rv_window,
                method=args.noise_method,
                sigma_cut=args.sigma_cut,
                exclude_expected_window=args.exclude_expected_window_from_noise,
            )
            pos_delta_loo_rec = recovery_map(snr_map_from_noise(pos_delta_loo_crop, loo_obs_noise), args.pos_delta_peak_sign)
            pos_delta_loo_common_rec = recovery_map(snr_map_from_noise(pos_delta_loo_crop, baseline_obs_noise), args.pos_delta_peak_sign)

            loo_pos_delta_metrics = metrics_for_map(
                signal_map=pos_delta_loo_rec,
                noise_map=pos_delta_loo_rec,
                RV=RV_crop,
                Kp=Kp_crop,
                expected_kp=expected_kp,
                expected_rv=expected_rv,
                kp_window=args.kp_window,
                rv_window=args.rv_window,
                aperture_kp_half_width=args.aperture_kp_half_width,
                aperture_rv_half_width=args.aperture_rv_half_width,
                sigma_cut=args.sigma_cut,
                noise_method="std",
                exclude_expected_window_from_noise=False,
                edge_tolerance_pixels=args.edge_tolerance_pixels,
                fixed_noise=1.0,
            )
            loo_pos_delta_common_metrics = metrics_for_map(
                signal_map=pos_delta_loo_common_rec,
                noise_map=pos_delta_loo_common_rec,
                RV=RV_crop,
                Kp=Kp_crop,
                expected_kp=expected_kp,
                expected_rv=expected_rv,
                kp_window=args.kp_window,
                rv_window=args.rv_window,
                aperture_kp_half_width=args.aperture_kp_half_width,
                aperture_rv_half_width=args.aperture_rv_half_width,
                sigma_cut=args.sigma_cut,
                noise_method="std",
                exclude_expected_window_from_noise=False,
                edge_tolerance_pixels=args.edge_tolerance_pixels,
                fixed_noise=1.0,
            )
            r.update(prefixed("loo_pos_delta", loo_pos_delta_metrics))
            r.update(prefixed("loo_pos_delta_common_noise", loo_pos_delta_common_metrics))
            r["loo_pos_delta_expected_minus_baseline"] = loo_pos_delta_metrics["expected_peak_snr"] - pos_delta_baseline_metrics["expected_peak_snr"]
            r["removed_order_pos_delta_fixed_raw_contribution"] = pos_delta_baseline_metrics["fixed_raw"] - loo_pos_delta_metrics["fixed_raw"]
        else:
            r.update(prefixed("loo_pos_delta", nan_metrics()))
            r.update(prefixed("loo_pos_delta_common_noise", nan_metrics()))
            r["loo_pos_delta_expected_minus_baseline"] = np.nan
            r["removed_order_pos_delta_fixed_raw_contribution"] = np.nan

        if has_negative_injection:
            neg_loo = sum_orders(neg_fmap, loo_positions)
            neg_loo_crop = neg_loo[kp_mask][:, rv_mask]
            neg_delta_loo_crop = neg_loo_crop - obs_loo_crop

            loo_obs_noise = calculate_noise(
                obs_loo_crop,
                RV=RV_crop,
                Kp=Kp_crop,
                expected_kp=expected_kp,
                expected_rv=expected_rv,
                kp_window=args.kp_window,
                rv_window=args.rv_window,
                method=args.noise_method,
                sigma_cut=args.sigma_cut,
                exclude_expected_window=args.exclude_expected_window_from_noise,
            )
            neg_delta_loo_rec = recovery_map(snr_map_from_noise(neg_delta_loo_crop, loo_obs_noise), args.neg_delta_peak_sign)
            neg_delta_loo_common_rec = recovery_map(snr_map_from_noise(neg_delta_loo_crop, baseline_obs_noise), args.neg_delta_peak_sign)

            loo_neg_delta_metrics = metrics_for_map(
                signal_map=neg_delta_loo_rec,
                noise_map=neg_delta_loo_rec,
                RV=RV_crop,
                Kp=Kp_crop,
                expected_kp=expected_kp,
                expected_rv=expected_rv,
                kp_window=args.kp_window,
                rv_window=args.rv_window,
                aperture_kp_half_width=args.aperture_kp_half_width,
                aperture_rv_half_width=args.aperture_rv_half_width,
                sigma_cut=args.sigma_cut,
                noise_method="std",
                exclude_expected_window_from_noise=False,
                edge_tolerance_pixels=args.edge_tolerance_pixels,
                fixed_noise=1.0,
            )
            loo_neg_delta_common_metrics = metrics_for_map(
                signal_map=neg_delta_loo_common_rec,
                noise_map=neg_delta_loo_common_rec,
                RV=RV_crop,
                Kp=Kp_crop,
                expected_kp=expected_kp,
                expected_rv=expected_rv,
                kp_window=args.kp_window,
                rv_window=args.rv_window,
                aperture_kp_half_width=args.aperture_kp_half_width,
                aperture_rv_half_width=args.aperture_rv_half_width,
                sigma_cut=args.sigma_cut,
                noise_method="std",
                exclude_expected_window_from_noise=False,
                edge_tolerance_pixels=args.edge_tolerance_pixels,
                fixed_noise=1.0,
            )
            r.update(prefixed("loo_neg_delta", loo_neg_delta_metrics))
            r.update(prefixed("loo_neg_delta_common_noise", loo_neg_delta_common_metrics))
            r["loo_neg_delta_expected_minus_baseline"] = loo_neg_delta_metrics["expected_peak_snr"] - neg_delta_baseline_metrics["expected_peak_snr"]
            r["removed_order_neg_delta_fixed_raw_contribution"] = neg_delta_baseline_metrics["fixed_raw"] - loo_neg_delta_metrics["fixed_raw"]
        else:
            r.update(prefixed("loo_neg_delta", nan_metrics()))
            r.update(prefixed("loo_neg_delta_common_noise", nan_metrics()))
            r["loo_neg_delta_expected_minus_baseline"] = np.nan
            r["removed_order_neg_delta_fixed_raw_contribution"] = np.nan

    csv_path = output_dir / "order_diagnostics.csv"
    if len(rows) > 0:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    column_csv_path = output_dir / "column_diagnostics.csv"
    if len(column_rows_all) > 0:
        with open(column_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(column_rows_all[0].keys()))
            writer.writeheader()
            writer.writerows(column_rows_all)
    else:
        column_csv_path = None

    baseline_orders = [int(r["original_order"]) for r in rows if r["include_data_quality"]]
    excluded_orders = [int(r["original_order"]) for r in rows if not r["include_data_quality"]]

    summary = {
        "project_path": str(project_path),
        "model": args.model,
        "night": args.night,
        "camera": args.camera,
        "k": args.k,
        "selection_philosophy": "baseline orders selected from data-quality metrics only; CCF SNR diagnostics are not used for order inclusion",
        "rv_min": float(RV_MIN),
        "rv_max": float(RV_MAX),
        "kp_min": float(KP_MIN),
        "kp_max": float(KP_MAX),
        "wavelength_min_nm": args.wavelength_min_nm,
        "wavelength_max_nm": args.wavelength_max_nm,
        "min_overlap_fraction": args.min_overlap_fraction,
        "min_valid_flux_fraction": args.min_valid_flux_fraction,
        "min_ccf_finite_fraction": args.min_ccf_finite_fraction,
        "min_column_finite_fraction": args.min_column_finite_fraction,
        "column_rms_sigma": args.column_rms_sigma,
        "column_mad_sigma": args.column_mad_sigma,
        "max_bad_column_fraction": args.max_bad_column_fraction,
        "max_bad_edge_column_fraction": args.max_bad_edge_column_fraction,
        "expected_kp": expected_kp,
        "expected_rv": expected_rv,
        "kp_window": args.kp_window,
        "rv_window": args.rv_window,
        "aperture_kp_half_width": args.aperture_kp_half_width,
        "aperture_rv_half_width": args.aperture_rv_half_width,
        "noise_method": args.noise_method,
        "exclude_expected_window_from_noise": args.exclude_expected_window_from_noise,
        "map_sign": args.map_sign,
        "pos_delta_peak_sign": args.pos_delta_peak_sign,
        "neg_delta_peak_sign": args.neg_delta_peak_sign,
        "baseline_orders": baseline_orders,
        "excluded_orders_data_quality": excluded_orders,
        "baseline_obs": obs_baseline_metrics,
        "baseline_pos_delta": pos_delta_baseline_metrics,
        "baseline_neg_delta": neg_delta_baseline_metrics,
        "obs_file": str(obs_file),
        "positive_injection_file": str(pos_file) if has_positive_injection else None,
        "negative_injection_file": str(neg_file) if has_negative_injection else None,
        "analysis_ready_file": str(analysis_ready_file),
        "column_data_source": column_data_source,
        "column_data_key": column_data_key,
        "order_csv": str(csv_path),
        "column_csv": str(column_csv_path) if column_csv_path is not None else None,
    }

    summary_path = output_dir / "order_diagnostics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(to_jsonable(summary), f, indent=4)

    print()
    print("=" * 72)
    print("Data-quality-first order diagnostics complete")
    print("=" * 72)
    print(f"Saved order CSV   : {csv_path}")
    if column_csv_path is not None:
        print(f"Saved column CSV  : {column_csv_path}")
    else:
        print("Saved column CSV  : none; no usable column diagnostic array found")
    print(f"Saved summary JSON: {summary_path}")
    print()
    print("Baseline orders selected by DATA QUALITY ONLY:")
    print(" ".join(str(o) for o in baseline_orders))
    print()
    print("Excluded orders from data-quality cuts:")
    print(" ".join(str(o) for o in excluded_orders) if excluded_orders else "none")
    print()
    print("Baseline observed expected-window peak SNR [diagnostic only]:")
    print(summary["baseline_obs"]["expected_peak_snr"])
    print()
    print("Baseline positive-delta expected-window peak SNR [diagnostic only]:")
    print(summary["baseline_pos_delta"]["expected_peak_snr"])
    print()
    print("Baseline negative-delta expected-window peak SNR [diagnostic only]:")
    print(summary["baseline_neg_delta"]["expected_peak_snr"])
    print("=" * 72)


# -----------------------------------------------------------------------------
# Misc helpers that depend on metric schema
# -----------------------------------------------------------------------------


def nan_metrics() -> dict[str, Any]:
    return {
        "noise": np.nan,
        "fixed_raw": np.nan,
        "fixed_snr": np.nan,
        "aperture_raw_mean": np.nan,
        "aperture_raw_sum": np.nan,
        "aperture_snr_mean": np.nan,
        "aperture_snr_sum": np.nan,
        "expected_peak_snr": np.nan,
        "expected_peak_kp": np.nan,
        "expected_peak_rv": np.nan,
        "global_snr": np.nan,
        "global_kp": np.nan,
        "global_rv": np.nan,
        "global_near_expected": False,
        "global_edge_flag": False,
        "global_to_expected_ratio": np.nan,
    }


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


if __name__ == "__main__":
    main()
