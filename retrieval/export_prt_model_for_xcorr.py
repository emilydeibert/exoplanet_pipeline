"""Export a pRT emission model in the legacy xcorr model .npy format."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

from retrieval.prt_emission_model import (
    ANGSTROM_TO_CM,
    generate_prt_emission_model,
    load_yaml_config,
    parameters_from_config,
    require_numpy,
    setup_logging,
)


VACUUM_TO_AIR_DEFAULT_BY_SPECIES = {
    "Fe": True,
    "TiO": True,
    "Ti": True,
    "Ti+": False,
    "Fe+": False,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="retrieval/configs/mascara1b_fe_smoketest.yaml",
        help="YAML retrieval config.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .npy path. Column 0 is wavelength Angstrom; column 1 depends on --output-format.",
    )
    parser.add_argument("--wavelength-min-micron", type=float, default=None)
    parser.add_argument("--wavelength-max-micron", type=float, default=None)
    parser.add_argument(
        "--output-format",
        choices=["raw_flux", "xcorr_processed"],
        default="raw_flux",
        help="raw_flux saves the pRT flux directly; xcorr_processed reproduces the old template processing.",
    )
    parser.add_argument("--stellar-temperature", type=float, default=9360.0)
    parser.add_argument("--stellar-radius-rsun", type=float, default=1.67)
    parser.add_argument(
        "--vacuum-to-air",
        choices=["auto", "true", "false"],
        default="auto",
        help="For xcorr_processed, convert vacuum wavelengths to air. auto uses the old species defaults.",
    )
    parser.add_argument(
        "--remove-envelope",
        choices=["auto", "true", "false"],
        default="auto",
        help="For xcorr_processed, subtract the lower envelope. auto means true.",
    )
    parser.add_argument("--envelope-pixels", type=int, default=400)
    parser.add_argument("--envelope-poly-order", type=int, default=4)
    parser.add_argument(
        "--save-with-continuum",
        default=None,
        help="Optional .npy path for the planet/star contrast before envelope removal.",
    )
    return parser.parse_args()


def wavelength_bounds_from_args(config: dict, args: argparse.Namespace) -> list[float]:
    model_cfg = config.get("model", {})
    default_bounds = list(model_cfg.get("wavelength_boundaries_micron", [0.383, 1.0]))
    wavelength_min = (
        float(args.wavelength_min_micron)
        if args.wavelength_min_micron is not None
        else float(default_bounds[0])
    )
    wavelength_max = (
        float(args.wavelength_max_micron)
        if args.wavelength_max_micron is not None
        else float(default_bounds[1])
    )
    if wavelength_min <= 0 or wavelength_max <= wavelength_min:
        raise ValueError(
            "--wavelength-min-micron and --wavelength-max-micron must satisfy "
            "0 < min < max."
        )
    return [wavelength_min, wavelength_max]


def configured_primary_species(config: dict[str, Any]) -> str:
    species = config.get("species", {}).get("line_species", ["Fe"])
    if not species:
        return "Fe"
    return str(species[0])


def resolve_bool_option(value: str, default: bool) -> bool:
    if value == "auto":
        return bool(default)
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"Unknown boolean option value {value!r}.")


def vac2air(vacwave_angstrom: Any) -> Any:
    """Convert vacuum wavelengths in Angstrom to air wavelengths.

    This reproduces the function in ``generateModels_original.py``.
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
    """Convert stellar radius from Rsun to Rjup using the old astropy convention."""

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
        raise ValueError("--envelope-pixels must be at least 2.")
    if int(order) < 0:
        raise ValueError("--envelope-poly-order must be >= 0.")

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
    filename: Path,
    ylabel: str,
    title: str,
) -> None:
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
    stats = finite_stats(values)
    logger.info(
        "%s min/median/max = %.6e / %.6e / %.6e; finite_fraction = %.6f",
        label,
        stats["min"],
        stats["median"],
        stats["max"],
        stats["finite_fraction"],
    )


def build_xcorr_processed_export(
    wavelengths_cm: Any,
    flux: Any,
    args: argparse.Namespace,
    config: dict[str, Any],
    logger: Any,
) -> tuple[Any, Any, Optional[Any], bool, bool]:
    """Return wavelength_Angstrom, processed template, contrast, conversion flags."""

    np = require_numpy()
    species = configured_primary_species(config)
    default_vacuum_to_air = VACUUM_TO_AIR_DEFAULT_BY_SPECIES.get(species, True)
    vacuum_to_air = resolve_bool_option(args.vacuum_to_air, default_vacuum_to_air)
    remove_envelope = resolve_bool_option(args.remove_envelope, True)

    planck = stellar_planck_hz(wavelengths_cm, args.stellar_temperature)
    radius_rjup = stellar_radius_rjup(args.stellar_radius_rsun)
    contrast = np.asarray(flux, dtype=float) / (planck * radius_rjup**2)

    wavelengths_angstrom_vacuum = np.asarray(wavelengths_cm, dtype=float) / ANGSTROM_TO_CM
    wavelengths_micron_vacuum = np.asarray(wavelengths_cm, dtype=float) / 1.0e-4
    if remove_envelope:
        processed = remove_env(
            wavelengths_micron_vacuum,
            contrast,
            args.envelope_pixels,
            args.envelope_poly_order,
        )
    else:
        processed = contrast.copy()

    if vacuum_to_air:
        wavelengths_angstrom = vac2air(wavelengths_angstrom_vacuum)
    else:
        wavelengths_angstrom = wavelengths_angstrom_vacuum

    logger.info("xcorr_processed species=%s", species)
    logger.info("stellar_temperature=%.3f K", float(args.stellar_temperature))
    logger.info("stellar_radius=%.6f Rsun = %.6f Rjup", float(args.stellar_radius_rsun), radius_rjup)
    logger.info("vacuum_to_air=%s", vacuum_to_air)
    logger.info("remove_envelope=%s", remove_envelope)
    logger.info("envelope_pixels=%d envelope_poly_order=%d", args.envelope_pixels, args.envelope_poly_order)
    return wavelengths_angstrom, processed, contrast, vacuum_to_air, remove_envelope


def main() -> None:
    args = parse_args()
    np = require_numpy()
    config = load_yaml_config(args.config)
    parameters = parameters_from_config(config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_path.parent, f"{output_path.stem}_export_prt_model_for_xcorr.log")
    wavelength_bounds = wavelength_bounds_from_args(config, args)

    logger.info("Exporting pRT model for xcorr pipeline")
    logger.info("Output file: %s", output_path)
    logger.info("Output format: %s", args.output_format)
    logger.info("Wavelength bounds: %.6f-%.6f micron", *wavelength_bounds)
    logger.info("Parameters: %s", parameters)

    wavelengths_cm, flux, metadata = generate_prt_emission_model(
        config=config,
        parameters=parameters,
        wavelength_boundaries_micron=wavelength_bounds,
        logger=logger,
    )

    raw_wavelengths_angstrom = wavelengths_cm / ANGSTROM_TO_CM
    raw_wavelengths_micron = wavelengths_cm / 1.0e-4
    logger.info(
        "Raw wavelength range: %.6f-%.6f Angstrom; %.6f-%.6f micron",
        float(np.nanmin(raw_wavelengths_angstrom)),
        float(np.nanmax(raw_wavelengths_angstrom)),
        float(np.nanmin(raw_wavelengths_micron)),
        float(np.nanmax(raw_wavelengths_micron)),
    )
    logger.info("Raw wavelength monotonic=%s", is_strictly_monotonic(raw_wavelengths_angstrom))
    log_array_diagnostics(logger, "raw pRT flux", flux)

    raw_plot_path = output_path.with_name(f"{output_path.stem}_raw_prt_flux.png")
    save_xy_plot(
        raw_wavelengths_angstrom,
        flux,
        raw_plot_path,
        ylabel="Raw pRT emission flux",
        title="Raw pRT emission spectrum",
    )

    contrast = None
    if args.output_format == "raw_flux":
        wavelengths_angstrom = raw_wavelengths_angstrom
        export_values = flux
        processed_for_stats = flux
    else:
        (
            wavelengths_angstrom,
            export_values,
            contrast,
            _vacuum_to_air,
            _remove_envelope,
        ) = build_xcorr_processed_export(
            wavelengths_cm=wavelengths_cm,
            flux=flux,
            args=args,
            config=config,
            logger=logger,
        )
        processed_for_stats = export_values

        contrast_plot_path = output_path.with_name(f"{output_path.stem}_contrast_with_continuum.png")
        save_xy_plot(
            wavelengths_angstrom,
            contrast,
            contrast_plot_path,
            ylabel="Planet/star contrast",
            title="Planet/star contrast before envelope removal",
        )
        logger.info("Saved contrast diagnostic plot to %s", contrast_plot_path)

        if args.save_with_continuum is not None:
            continuum_path = Path(args.save_with_continuum)
            continuum_path.parent.mkdir(parents=True, exist_ok=True)
            contrast_export = np.column_stack([wavelengths_angstrom, contrast])
            np.save(continuum_path, contrast_export)
            logger.info("Saved contrast-with-continuum model to %s", continuum_path)

    export = np.column_stack([wavelengths_angstrom, export_values])
    np.save(output_path, export)

    final_plot_path = output_path.with_name(f"{output_path.stem}_{args.output_format}.png")
    save_xy_plot(
        wavelengths_angstrom,
        export_values,
        final_plot_path,
        ylabel=("Raw pRT emission flux" if args.output_format == "raw_flux" else "Continuum-removed template"),
        title=("Raw pRT emission flux" if args.output_format == "raw_flux" else "Final xcorr processed template"),
    )

    logger.info(
        "Saved wavelength range: %.6f-%.6f Angstrom; %.6f-%.6f micron",
        float(np.nanmin(wavelengths_angstrom)),
        float(np.nanmax(wavelengths_angstrom)),
        float(np.nanmin(wavelengths_angstrom / 1.0e4)),
        float(np.nanmax(wavelengths_angstrom / 1.0e4)),
    )
    logger.info("Saved wavelength monotonic=%s", is_strictly_monotonic(wavelengths_angstrom))
    if contrast is not None:
        log_array_diagnostics(logger, "planet/star contrast", contrast)
    log_array_diagnostics(logger, "processed template", processed_for_stats)

    logger.info("Model metadata: %s", metadata)
    logger.info("Saved xcorr model array with shape %s to %s", export.shape, output_path)
    logger.info("Column 0 is wavelength in Angstrom; existing xcorr code divides by 10 to get nm.")
    if args.output_format == "raw_flux":
        logger.info("Column 1 is raw pRT emission flux before convolution/template_to_dmag.")
    else:
        logger.info("Column 1 is continuum-removed planet/star contrast template.")
    logger.info("Saved raw flux diagnostic plot to %s", raw_plot_path)
    logger.info("Saved final model diagnostic plot to %s", final_plot_path)


if __name__ == "__main__":
    main()
