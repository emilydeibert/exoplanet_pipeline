"""Reusable pRT rest-model processing helpers.

This module contains the legacy xcorr-template processing that was validated
against ``retrieval/generateModels_original.py``.  The retrieval/grid code can
use the same representation without importing or rewriting the old xcorr
scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from retrieval.prt_emission_model import ANGSTROM_TO_CM, MICRON_TO_CM, require_numpy


VACUUM_TO_AIR_DEFAULT_BY_SPECIES = {
    "Fe": True,
    "TiO": True,
    "Ti": True,
    "Ti+": False,
    "Tiplus": False,
    "Fe+": False,
    "Feplus": False,
    "Cr+": False,
    "Crplus": False,
    "V": True,
    "V+": True,
    "VO": True,
    "Ca": True,
    "Ca+": True,
    "Cr": True,
    "FeH": True,
    "K": True,
    "OH": True,
    "Mg": True,
    "Si": True,
    "Al": True,
    "CaH": True,
    "Na": True,
}


@dataclass
class ProcessedXcorrModel:
    """pRT model converted to the legacy xcorr template representation."""

    wavelength_cm: Any
    wavelength_angstrom: Any
    template: Any
    contrast: Any
    metadata: dict[str, Any]


@dataclass
class RestModelRepresentation:
    """Rest-frame model representation used before Doppler shifting."""

    name: str
    wavelengths_cm: Any
    flux: Any
    metadata: dict[str, Any]
    wavelength_angstrom: Optional[Any] = None
    contrast: Optional[Any] = None


def configured_primary_species(config: Mapping[str, Any]) -> str:
    """Return the first configured retrieval species, defaulting to Fe."""

    species = config.get("species", {}).get("line_species", ["Fe"])
    if not species:
        return "Fe"
    return str(species[0])


def default_vacuum_to_air_for_species(species: str) -> bool:
    """Return the legacy vacuum-to-air default for a species."""

    return bool(VACUUM_TO_AIR_DEFAULT_BY_SPECIES.get(str(species), True))


def resolve_bool_option(value: Any, default: bool) -> bool:
    """Resolve bool-like CLI/config values, with ``auto`` mapped to default."""

    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value

    value_norm = str(value).strip().lower()
    if value_norm in {"auto", ""}:
        return bool(default)
    if value_norm in {"true", "yes", "y", "1"}:
        return True
    if value_norm in {"false", "no", "n", "0"}:
        return False
    raise ValueError(f"Unknown boolean option value {value!r}.")


def vac2air(vacwave_angstrom: Any) -> Any:
    """Convert vacuum wavelengths in Angstrom to air wavelengths.

    This reproduces ``vac2air()`` in ``generateModels_original.py``.
    """

    np = require_numpy()
    vacwave = np.asarray(vacwave_angstrom, dtype=float)
    s = 1.0e4 / vacwave
    n = 1.0 + 8.34254e-5 + 2.406147e-2 / (130.0 - s**2) + 1.5998e-4 / (38.9 - s**2)
    return vacwave / n


def stellar_planck_hz(wavelengths_cm: Any, stellar_temperature: float) -> Any:
    """Compute the stellar Planck function following the old ``star()`` helper."""

    np = require_numpy()
    try:
        from petitRADTRANS import physics as phys
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise RuntimeError("petitRADTRANS is required for the stellar Planck function.") from exc

    wavelengths_cm = np.asarray(wavelengths_cm, dtype=float)
    frequencies_hz = phys.wavelength2frequency(wavelengths_cm)
    return phys.planck_function_hz(float(stellar_temperature), frequencies_hz)


def stellar_radius_rjup(stellar_radius_rsun: float) -> float:
    """Convert stellar radius from Rsun to Rjup using astropy units."""

    try:
        from astropy import units as u
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise RuntimeError("Astropy is required for stellar radius conversion.") from exc

    return (float(stellar_radius_rsun) * u.R_sun).to(u.R_jupiter).value


def remove_env(wave: Any, spec: Any, px: int, order: int) -> Any:
    """Remove the lower envelope using the old model-generation algorithm."""

    np = require_numpy()
    try:
        from scipy.stats import binned_statistic
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise RuntimeError("SciPy is required for envelope removal.") from exc

    wave = np.asarray(wave, dtype=float)
    spec = np.asarray(spec, dtype=float)
    if int(px) < 2:
        raise ValueError("envelope_pixels must be at least 2.")
    if int(order) < 0:
        raise ValueError("envelope_poly_order must be >= 0.")

    binned = binned_statistic(wave, spec, statistic="min", bins=int(px))
    bin_mids = binned[1][1:] - (binned[1][1] - binned[1][0]) / 2.0
    finite = np.isfinite(bin_mids) & np.isfinite(binned[0])
    if finite.sum() <= int(order):
        raise ValueError(
            "Not enough finite lower-envelope bins for the requested polynomial order. "
            f"finite_bins={finite.sum()}, order={order}."
        )
    fit = np.polyfit(bin_mids[finite], binned[0][finite], int(order))
    env = np.polyval(fit, wave)
    return spec - env


def finite_stats(values: Any) -> dict[str, float]:
    """Return min/median/max and finite fraction for logging."""

    np = require_numpy()
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return {
            "min": float("nan"),
            "median": float("nan"),
            "max": float("nan"),
            "finite_fraction": 0.0,
        }
    return {
        "min": float(np.nanmin(values[finite])),
        "median": float(np.nanmedian(values[finite])),
        "max": float(np.nanmax(values[finite])),
        "finite_fraction": float(np.sum(finite) / values.size),
    }


def is_strictly_monotonic(values: Any) -> bool:
    """Return True when finite values are strictly increasing or decreasing."""

    np = require_numpy()
    values = np.asarray(values, dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size < 2:
        return False
    diffs = np.diff(finite_values)
    return bool(np.all(diffs > 0) or np.all(diffs < 0))


def save_xy_plot(
    wavelength_angstrom: Any,
    values: Any,
    filename: Any,
    ylabel: str,
    title: str,
) -> None:
    """Save a quick wavelength/value diagnostic plot."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise RuntimeError("Matplotlib is required for export diagnostic plots.") from exc

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wavelength_angstrom, values, lw=0.7)
    ax.set_xlabel("Wavelength [Angstrom]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=250)
    plt.close(fig)


def log_array_diagnostics(logger: Any, label: str, values: Any) -> None:
    """Log finite min/median/max diagnostics for an array."""

    stats = finite_stats(values)
    logger.info(
        "%s min/median/max = %.6e / %.6e / %.6e; finite_fraction = %.6f",
        label,
        stats["min"],
        stats["median"],
        stats["max"],
        stats["finite_fraction"],
    )


def _validate_rest_model_arrays(wavelengths_cm: Any, flux: Any) -> tuple[Any, Any]:
    np = require_numpy()
    wavelengths_cm = np.asarray(wavelengths_cm, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if wavelengths_cm.ndim != 1 or flux.ndim != 1:
        raise ValueError("pRT rest wavelengths and flux must both be 1D arrays.")
    if wavelengths_cm.size != flux.size:
        raise ValueError(
            "pRT rest wavelengths and flux must have the same length. "
            f"Got {wavelengths_cm.size} and {flux.size}."
        )
    if wavelengths_cm.size < 3:
        raise ValueError("At least three pRT rest-model pixels are required.")
    if not np.isfinite(wavelengths_cm).all() or np.nanmin(wavelengths_cm) <= 0:
        raise ValueError("pRT rest wavelengths must be finite positive cm values.")
    if not np.isfinite(flux).all():
        raise ValueError("pRT rest flux contains NaN or infinite values.")
    if not is_strictly_monotonic(wavelengths_cm):
        raise ValueError("pRT rest wavelengths must be strictly monotonic.")
    return wavelengths_cm, flux


def process_prt_model_for_xcorr(
    wavelengths_cm: Any,
    flux: Any,
    species: str = "Fe",
    stellar_temperature: float = 9360.0,
    stellar_radius_rsun: float = 1.67,
    vacuum_to_air: bool = True,
    remove_envelope: bool = True,
    envelope_pixels: int = 400,
    envelope_poly_order: int = 4,
    logger: Optional[Any] = None,
) -> ProcessedXcorrModel:
    """Convert raw pRT emission flux to the legacy xcorr template format.

    Parameters
    ----------
    wavelengths_cm
        pRT rest wavelengths in cm.
    flux
        Raw pRT emission flux on ``wavelengths_cm``.
    species
        Primary opacity species, used only for metadata/logging here.
    stellar_temperature
        Stellar effective temperature in Kelvin for the Planck function.
    stellar_radius_rsun
        Stellar radius in solar radii, converted to Jupiter radii before
        applying the old contrast formula.
    vacuum_to_air
        If True, convert output wavelengths from vacuum Angstrom to air
        Angstrom using the old ``vac2air`` function.
    remove_envelope
        If True, subtract the lower envelope from the contrast using
        ``scipy.stats.binned_statistic(..., statistic="min")``.

    Returns
    -------
    ProcessedXcorrModel
        Processed wavelengths in cm and Angstrom, the continuum-removed
        template/depth, the contrast before envelope removal, and metadata.
    """

    np = require_numpy()
    wavelengths_cm, flux = _validate_rest_model_arrays(wavelengths_cm, flux)

    planck = stellar_planck_hz(wavelengths_cm, stellar_temperature)
    radius_rjup = stellar_radius_rjup(stellar_radius_rsun)
    contrast = flux / (planck * radius_rjup**2)

    wavelengths_angstrom_vacuum = wavelengths_cm / ANGSTROM_TO_CM
    wavelengths_micron_vacuum = wavelengths_cm / MICRON_TO_CM
    if remove_envelope:
        template = remove_env(
            wavelengths_micron_vacuum,
            contrast,
            int(envelope_pixels),
            int(envelope_poly_order),
        )
    else:
        template = contrast.copy()

    if vacuum_to_air:
        wavelengths_angstrom = vac2air(wavelengths_angstrom_vacuum)
    else:
        wavelengths_angstrom = wavelengths_angstrom_vacuum.copy()
    processed_wavelengths_cm = wavelengths_angstrom * ANGSTROM_TO_CM

    metadata = {
        "model_representation": "xcorr_processed",
        "species": str(species),
        "stellar_temperature": float(stellar_temperature),
        "stellar_radius_rsun": float(stellar_radius_rsun),
        "stellar_radius_rjup": float(radius_rjup),
        "vacuum_to_air": bool(vacuum_to_air),
        "remove_envelope": bool(remove_envelope),
        "envelope_pixels": int(envelope_pixels),
        "envelope_poly_order": int(envelope_poly_order),
        "envelope_wavelength_unit": "micron",
    }

    if logger is not None:
        logger.info("xcorr_processed species=%s", species)
        logger.info("stellar_temperature=%.3f K", float(stellar_temperature))
        logger.info(
            "stellar_radius=%.6f Rsun = %.6f Rjup",
            float(stellar_radius_rsun),
            radius_rjup,
        )
        logger.info("vacuum_to_air=%s", vacuum_to_air)
        logger.info("remove_envelope=%s", remove_envelope)
        logger.info(
            "envelope_pixels=%d envelope_poly_order=%d",
            int(envelope_pixels),
            int(envelope_poly_order),
        )

    return ProcessedXcorrModel(
        wavelength_cm=processed_wavelengths_cm,
        wavelength_angstrom=wavelengths_angstrom,
        template=np.asarray(template, dtype=float),
        contrast=np.asarray(contrast, dtype=float),
        metadata=metadata,
    )


def prepare_rest_model_for_representation(
    wavelengths_cm: Any,
    flux: Any,
    config: Mapping[str, Any],
    logger: Optional[Any] = None,
) -> RestModelRepresentation:
    """Apply the configured rest-model representation before Doppler shifting."""

    np = require_numpy()
    model_cfg = config.get("model", {})
    representation = str(model_cfg.get("representation", "raw_flux")).strip().lower()

    if representation == "raw_flux":
        rest_wave, rest_flux = _validate_rest_model_arrays(wavelengths_cm, flux)
        metadata = {"model_representation": "raw_flux"}
        if logger is not None:
            logger.info("Using rest-frame model representation: raw_flux")
        return RestModelRepresentation(
            name="raw_flux",
            wavelengths_cm=rest_wave,
            flux=np.asarray(rest_flux, dtype=float),
            metadata=metadata,
            wavelength_angstrom=rest_wave / ANGSTROM_TO_CM,
            contrast=None,
        )

    if representation != "xcorr_processed":
        raise ValueError(
            "model.representation must be one of: raw_flux, xcorr_processed. "
            f"Got {representation!r}."
        )

    processing_cfg = model_cfg.get("xcorr_processed", {})
    species = str(processing_cfg.get("species", configured_primary_species(config)))
    vacuum_to_air = resolve_bool_option(
        processing_cfg.get("vacuum_to_air", "auto"),
        default_vacuum_to_air_for_species(species),
    )
    remove_envelope = resolve_bool_option(processing_cfg.get("remove_envelope", True), True)
    processed = process_prt_model_for_xcorr(
        wavelengths_cm=wavelengths_cm,
        flux=flux,
        species=species,
        stellar_temperature=float(processing_cfg.get("stellar_temperature", 9360.0)),
        stellar_radius_rsun=float(processing_cfg.get("stellar_radius_rsun", 1.67)),
        vacuum_to_air=vacuum_to_air,
        remove_envelope=remove_envelope,
        envelope_pixels=int(processing_cfg.get("envelope_pixels", 400)),
        envelope_poly_order=int(processing_cfg.get("envelope_poly_order", 4)),
        logger=logger,
    )
    if logger is not None:
        logger.info("Using rest-frame model representation: xcorr_processed")
    return RestModelRepresentation(
        name="xcorr_processed",
        wavelengths_cm=processed.wavelength_cm,
        flux=processed.template,
        metadata=processed.metadata,
        wavelength_angstrom=processed.wavelength_angstrom,
        contrast=processed.contrast,
    )
