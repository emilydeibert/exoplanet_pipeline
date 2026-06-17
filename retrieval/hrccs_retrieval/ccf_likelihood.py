"""CCF validation maps and matched-filter likelihoods for HRCCS retrievals."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from retrieval.prt_emission_model import C_KM_S

from .data_loading import ObservationBlock, ensure_exopipe_importable, require_numpy


def compute_xcorr_detection_map(
    blocks: Sequence[ObservationBlock],
    F_model: Any,
    RV: Any,
    Kp: Any,
    save_per_order: bool = False,
    logger: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    """Reproduce the existing xcorr fmap calculation and map combination.

    Per order, this calls ``cc.modelCorrelation_weighted`` and
    ``cc.finalCorr_stack``.  The combined map is then summed over orders and
    multiplied by ``-1.0`` per night/camera, matching ``getResults.py``.
    """

    np = require_numpy()
    ensure_exopipe_importable()
    from exopipe import crosscorrelation as cc

    combined_map = np.zeros((len(Kp), len(RV)), dtype=float)
    per_order_maps = []
    per_order_labels = []
    start = time.perf_counter()

    for block in blocks:
        block_fmaps = []
        for local_idx, original_order in enumerate(block.orders):
            order_wave = block.wave[local_idx]
            cmap = cc.modelCorrelation_weighted(
                order_data=block.data_rest[local_idx],
                order_wave=order_wave,
                order_sigmag=block.error_rest[local_idx],
                RV=RV,
                wavMinMax=[np.nanmin(order_wave), np.nanmax(order_wave)],
                F_model=F_model,
            )
            fmap = cc.finalCorr_stack(Kp, RV, cmap, block.phase)
            block_fmaps.append(fmap)

            if save_per_order:
                per_order_maps.append(fmap)
                per_order_labels.append(f"{block.night}_{block.camera}_order{original_order}")

        block_sum = np.nansum(np.asarray(block_fmaps, dtype=float), axis=0)
        block_sum *= -1.0
        combined_map += block_sum
        if logger is not None:
            logger.info("Added xcorr map for %s %s with %d orders", block.night, block.camera, len(block.orders))

    result = {
        "combined_map": combined_map,
        "seconds": float(time.perf_counter() - start),
    }
    if save_per_order:
        result["per_order_maps"] = np.asarray(per_order_maps, dtype=float)
        result["per_order_labels"] = np.asarray(per_order_labels, dtype="U128")
    if logger is not None:
        logger.info("Computed combined xcorr validation map in %.2fs", result["seconds"])
    return result


def crop_like_getresults(config: Any, combined_map: Any) -> dict[str, Any]:
    """Apply the same RV/Kp crop convention as ``exopipe/cli/getResults.py``."""

    np = require_numpy()
    RV = np.asarray(config.RV, dtype=float)
    Kp = np.asarray(config.Kp, dtype=float)
    rv_min = float(config.RV_MIN) if hasattr(config, "RV_MIN") else -75.0
    rv_max = float(config.RV_MAX) if hasattr(config, "RV_MAX") else 75.0
    kp_min = float(config.KP_MIN) if hasattr(config, "KP_MIN") else 1.0
    kp_max = float(config.KP_MAX) if hasattr(config, "KP_MAX") else 300.0

    rv_mask = (RV >= rv_min) & (RV <= rv_max)
    kp_mask = (Kp >= kp_min) & (Kp <= kp_max)
    return {
        "combined_map": combined_map[kp_mask][:, rv_mask],
        "RV": RV[rv_mask],
        "Kp": Kp[kp_mask],
        "rv_mask": rv_mask,
        "kp_mask": kp_mask,
    }


def calculate_snr_map(kpvsys_map: Any, sigma_cut: float = 3.0) -> tuple[Any, float]:
    """Compute the plotted SNR map exactly like the existing result script."""

    np = require_numpy()
    try:
        from astropy.stats import sigma_clip
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise RuntimeError("Astropy is required for sigma-clipped SNR maps.") from exc

    kpvsys_map = np.asarray(kpvsys_map, dtype=float)
    centered = kpvsys_map - np.nanmedian(kpvsys_map)
    clipped = sigma_clip(centered, sigma_upper=float(sigma_cut), sigma_lower=100)
    noise = float(np.nanstd(clipped))
    if noise <= 0 or not np.isfinite(noise):
        raise ValueError("Could not compute a finite positive SNR-map noise.")
    return centered / noise, noise


def find_peak(snr_map: Any, RV: Any, Kp: Any) -> dict[str, Any]:
    """Return the peak SNR and coordinates."""

    np = require_numpy()
    max_index = int(np.nanargmax(snr_map))
    kp_idx, rv_idx = np.unravel_index(max_index, snr_map.shape)
    return {
        "snr": float(snr_map[kp_idx, rv_idx]),
        "rv": float(RV[rv_idx]),
        "kp": float(Kp[kp_idx]),
        "rv_idx": int(rv_idx),
        "kp_idx": int(kp_idx),
    }


def _standardized_model_at_velocity(order_wave: Any, F_model: Any, velocity_kms: float) -> Any:
    """Evaluate and standardize a shifted template like ``modelCorrelation_weighted``."""

    np = require_numpy()
    lam_rest = order_wave / (1.0 + float(velocity_kms) / C_KM_S)
    model = np.asarray(F_model(lam_rest), dtype=float)
    finite = np.isfinite(model)
    if np.sum(finite) < 10:
        return None
    model = model - np.nanmean(model[finite])
    std = np.nanstd(model[finite])
    if std > 0 and np.isfinite(std):
        model = model / std
    return model


def matched_filter_terms(
    blocks: Sequence[ObservationBlock],
    F_model: Any,
    Kp: float,
    Vsys: float,
) -> dict[str, Any]:
    """Compute weighted data-model terms directly along one planet trail.

    The model velocity sampled for each exposure is
    ``Vsys + Kp * sin(2*pi*phase)``, matching the trail stacked by
    ``finalCorr_stack``.  The objective is a Gaussian matched-filter
    likelihood with an analytic best-fit amplitude:

    ``log L = -0.5 * (D - C^2/M)``

    up to a data/noise constant, where ``C=sum(w*d*m)`` is the weighted
    data-model correlation, ``M=sum(w*m*m)`` is model power, and
    ``D=sum(w*d*d)`` is data power.  This is not the plotted SNR.
    """

    np = require_numpy()
    data_model = 0.0
    model_power = 0.0
    data_power = 0.0
    log_noise_term = 0.0
    n_valid = 0
    n_exposure_terms = 0

    for block in blocks:
        velocities = float(Vsys) + float(Kp) * np.sin(2.0 * np.pi * block.phase)

        for local_idx, _original_order in enumerate(block.orders):
            order_wave = block.wave[local_idx]
            data = block.data_rest[local_idx]
            sigma = block.error_rest[local_idx]

            for exp_idx, velocity in enumerate(velocities):
                model = _standardized_model_at_velocity(order_wave, F_model, velocity)
                if model is None:
                    continue

                row = np.asarray(data[exp_idx], dtype=float)
                sig = np.asarray(sigma[exp_idx], dtype=float)
                valid = np.isfinite(row) & np.isfinite(sig) & (sig > 0) & np.isfinite(model)
                if np.sum(valid) < 10:
                    continue

                w = 1.0 / (sig[valid] * sig[valid])
                d = row[valid]
                m = model[valid]

                # Same data-centering convention as weighted_ccf_per_rv.
                dmean = np.sum(w * d) / np.sum(w)
                d = d - dmean

                data_model += float(np.sum(w * d * m))
                model_power += float(np.sum(w * m * m))
                data_power += float(np.sum(w * d * d))
                log_noise_term += float(np.sum(np.log(2.0 * np.pi * sig[valid] * sig[valid])))
                n_valid += int(np.sum(valid))
                n_exposure_terms += 1

    if model_power <= 0 or not np.isfinite(model_power):
        return {
            "log_likelihood": -float("inf"),
            "amplitude": float("nan"),
            "data_model": data_model,
            "model_power": model_power,
            "data_power": data_power,
            "log_noise_term": log_noise_term,
            "n_valid": n_valid,
            "n_exposure_terms": n_exposure_terms,
        }

    amplitude = data_model / model_power
    chi2_best = data_power - data_model * data_model / model_power
    log_likelihood = -0.5 * chi2_best
    ccf_like_value = -data_model / (data_power * model_power) ** 0.5 if data_power > 0 else float("nan")

    return {
        "log_likelihood": float(log_likelihood),
        "amplitude": float(amplitude),
        "data_model": float(data_model),
        "model_power": float(model_power),
        "data_power": float(data_power),
        "chi2_best": float(chi2_best),
        "log_noise_term": float(log_noise_term),
        "ccf_peak_value": float(ccf_like_value),
        "n_valid": int(n_valid),
        "n_exposure_terms": int(n_exposure_terms),
    }


def evaluate_objective(
    blocks: Sequence[ObservationBlock],
    F_model: Any,
    Kp: float,
    Vsys: float,
    objective: str = "matched_filter_loglike",
    beta: Optional[float] = None,
) -> dict[str, Any]:
    """Evaluate the configured HRCCS objective at one parameter point."""

    terms = matched_filter_terms(blocks, F_model, Kp=Kp, Vsys=Vsys)
    objective = str(objective)
    if objective == "matched_filter_loglike":
        if beta is None:
            value = terms["log_likelihood"]
        else:
            np = require_numpy()
            beta = float(beta)
            if beta <= 0 or not np.isfinite(beta):
                raise ValueError(f"beta must be finite and positive; got {beta}.")
            # Current log_likelihood is -0.5*chi2_best up to constants.  If
            # sigma -> beta*sigma, chi2 scales as beta^-2 and the beta-dependent
            # Gaussian normalization contributes -N log(beta).  At beta=1 this
            # preserves the exact historical objective value.
            value = -0.5 * float(terms["chi2_best"]) / (beta * beta)
            value -= int(terms["n_valid"]) * float(np.log(beta))
    elif objective == "ccf_peak_value":
        if beta is not None:
            raise NotImplementedError(
                "beta/log_beta is only implemented for matched_filter_loglike. "
                "The ccf_peak_value debug objective has no meaningful variance "
                "normalization."
            )
        # Debug-only: this is a signed normalized correlation-like value, not
        # the retrieval likelihood to use for paper-grade runs.
        value = terms["ccf_peak_value"]
    else:
        raise ValueError("objective must be matched_filter_loglike or ccf_peak_value.")

    out = dict(terms)
    out["objective"] = objective
    out["objective_value"] = float(value)
    out["Kp"] = float(Kp)
    out["Vsys"] = float(Vsys)
    if beta is not None:
        out["beta"] = float(beta)
    return out


def save_validation_plot(RV: Any, Kp: Any, snr_map: Any, peak: dict[str, Any], filename: str | Path) -> None:
    """Save a compact map plus Kp/Vsys slices diagnostic plot."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise RuntimeError("Matplotlib is required for validation plots.") from exc

    np = require_numpy()
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 7))
    ax_rv = fig.add_subplot(221)
    ax_map = fig.add_subplot(223)
    ax_kp = fig.add_subplot(224)

    vmax = np.nanmax(np.abs(snr_map))
    im = ax_map.pcolormesh(RV, Kp, snr_map, shading="auto", vmin=-vmax, vmax=vmax)
    ax_map.set_xlabel("Vsys [km/s]")
    ax_map.set_ylabel("Kp [km/s]")

    ax_rv.plot(RV, snr_map[peak["kp_idx"], :])
    ax_rv.set_ylabel("SNR")
    ax_rv.set_xticks([])
    ax_rv.set_xlim(min(RV), max(RV))
    ax_rv.set_title(f"Kp = {peak['kp']:.1f} km/s")

    ax_kp.plot(snr_map[:, peak["rv_idx"]], Kp)
    ax_kp.set_xlabel("SNR")
    ax_kp.set_yticks([])
    ax_kp.set_ylim(min(Kp), max(Kp))
    ax_kp.set_title(f"Vsys = {peak['rv']:.1f} km/s")

    cbar = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
    cbar.set_label("SNR", rotation=270, labelpad=2)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with standard formatting."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
