"""Export a pRT emission model in the legacy xcorr model .npy format."""

from __future__ import annotations

import argparse
from pathlib import Path

from retrieval.model_processing import (
    configured_primary_species,
    default_vacuum_to_air_for_species,
    is_strictly_monotonic,
    log_array_diagnostics,
    process_prt_model_for_xcorr,
    resolve_bool_option,
    save_xy_plot,
)

from retrieval.prt_emission_model import (
    ANGSTROM_TO_CM,
    generate_prt_emission_model,
    load_yaml_config,
    parameters_from_config,
    require_numpy,
    setup_logging,
)


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
        species = configured_primary_species(config)
        vacuum_to_air = resolve_bool_option(
            args.vacuum_to_air,
            default_vacuum_to_air_for_species(species),
        )
        remove_envelope = resolve_bool_option(args.remove_envelope, True)
        processed_model = process_prt_model_for_xcorr(
            wavelengths_cm=wavelengths_cm,
            flux=flux,
            species=species,
            stellar_temperature=float(args.stellar_temperature),
            stellar_radius_rsun=float(args.stellar_radius_rsun),
            vacuum_to_air=vacuum_to_air,
            remove_envelope=remove_envelope,
            envelope_pixels=int(args.envelope_pixels),
            envelope_poly_order=int(args.envelope_poly_order),
            logger=logger,
        )
        wavelengths_angstrom = processed_model.wavelength_angstrom
        export_values = processed_model.template
        contrast = processed_model.contrast
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
