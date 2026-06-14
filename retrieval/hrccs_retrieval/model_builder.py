"""Build pRT models in the same representation as the trusted xcorr templates."""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping, Optional

from retrieval.model_processing import (
    configured_primary_species,
    default_vacuum_to_air_for_species,
    process_prt_model_for_xcorr,
    resolve_bool_option,
)
from retrieval.prt_emission_model import (
    generate_prt_emission_model,
    load_yaml_config,
    parameters_from_config,
    require_numpy,
)
from .data_loading import ensure_exopipe_importable


def parameters_with_updates(base_parameters: Mapping[str, float], updates: Optional[Mapping[str, float]] = None) -> dict[str, float]:
    """Return a float parameter dictionary with optional overrides."""

    parameters = {key: float(value) for key, value in dict(base_parameters).items()}
    for key, value in dict(updates or {}).items():
        if value is not None:
            parameters[key] = float(value)
    return parameters


def xcorr_processing_settings(retrieval_config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the xcorr-processed model settings from the retrieval YAML."""

    model_cfg = retrieval_config.get("model", {})
    processing_cfg = model_cfg.get("xcorr_processed", {})
    species = str(processing_cfg.get("species", configured_primary_species(retrieval_config)))
    vacuum_to_air = resolve_bool_option(
        processing_cfg.get("vacuum_to_air", "auto"),
        default_vacuum_to_air_for_species(species),
    )
    remove_envelope = resolve_bool_option(processing_cfg.get("remove_envelope", True), True)
    return {
        "species": species,
        "stellar_temperature": float(processing_cfg.get("stellar_temperature", 9360.0)),
        "stellar_radius_rsun": float(processing_cfg.get("stellar_radius_rsun", 1.67)),
        "vacuum_to_air": bool(vacuum_to_air),
        "remove_envelope": bool(remove_envelope),
        "envelope_pixels": int(processing_cfg.get("envelope_pixels", 400)),
        "envelope_poly_order": int(processing_cfg.get("envelope_poly_order", 4)),
    }


def generate_xcorr_processed_model_array(
    retrieval_config: Mapping[str, Any],
    parameters: Mapping[str, float],
    wavelength_boundaries_micron: Optional[list[float]] = None,
    logger: Optional[logging.Logger] = None,
) -> tuple[Any, dict[str, Any]]:
    """Generate pRT Fe emission and convert it to the working xcorr template array.

    Returns a two-column array matching the existing xcorr convention:
    column 0 is wavelength in Angstrom, column 1 is the continuum-removed
    planet/star-contrast-like template.
    """

    np = require_numpy()
    start = time.perf_counter()
    wavelengths_cm, raw_flux, prt_metadata = generate_prt_emission_model(
        config=retrieval_config,
        parameters=parameters,
        wavelength_boundaries_micron=wavelength_boundaries_micron,
        logger=logger,
    )
    settings = xcorr_processing_settings(retrieval_config)
    processed = process_prt_model_for_xcorr(
        wavelengths_cm=wavelengths_cm,
        flux=raw_flux,
        logger=logger,
        **settings,
    )
    model_array = np.column_stack([processed.wavelength_angstrom, processed.template])
    metadata = {
        "prt": prt_metadata,
        "xcorr_processing": processed.metadata,
        "parameters": {key: float(value) for key, value in parameters.items()},
        "seconds": float(time.perf_counter() - start),
    }
    if logger is not None:
        logger.info(
            "Generated xcorr_processed pRT model with %d pixels in %.2fs",
            model_array.shape[0],
            metadata["seconds"],
        )
    return model_array, metadata


def build_template_interpolator(model_array: Any, ghost_res: float) -> dict[str, Any]:
    """Apply the exact xcorr-template preparation used by the detection script."""

    np = require_numpy()
    try:
        from astropy.convolution import Gaussian1DKernel, convolve
        from scipy import interpolate
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise RuntimeError("Astropy and SciPy are required to prepare xcorr templates.") from exc

    model_array = np.asarray(model_array, dtype=float)
    if model_array.ndim != 2 or model_array.shape[1] != 2:
        raise ValueError("model_array must have shape (n_wave, 2).")

    model_flux = model_array[:, 1]
    model_wvl_nm = model_array[:, 0] / 10.0
    model_conv = convolve(model_flux, Gaussian1DKernel(stddev=float(ghost_res) / 2.35))

    ensure_exopipe_importable()
    from exopipe import crosscorrelation as cc

    model_dmag = cc.template_to_dmag(model_conv)
    interpolator = interpolate.interp1d(
        model_wvl_nm,
        model_dmag,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    return {
        "F_model": interpolator,
        "model_wvl_nm": model_wvl_nm,
        "model_flux": model_flux,
        "model_conv": model_conv,
        "model_dmag": model_dmag,
    }


def build_prt_xcorr_template(
    retrieval_config: Mapping[str, Any],
    exopipe_config: Any,
    parameters: Mapping[str, float],
    logger: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    """Generate pRT, process it, convolve it, and return an interpolator."""

    model_array, metadata = generate_xcorr_processed_model_array(
        retrieval_config=retrieval_config,
        parameters=parameters,
        logger=logger,
    )
    template = build_template_interpolator(model_array, ghost_res=exopipe_config.ghost_res)
    template["model_array"] = model_array
    template["metadata"] = metadata
    return template


def load_retrieval_config_and_parameters(path: str) -> tuple[dict[str, Any], dict[str, float]]:
    """Convenience loader for scripts."""

    retrieval_config = load_yaml_config(path)
    parameters = parameters_from_config(retrieval_config)
    return retrieval_config, parameters
