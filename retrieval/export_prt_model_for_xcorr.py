"""Export a pRT emission model in the legacy xcorr model .npy format."""

from __future__ import annotations

import argparse
from pathlib import Path

from retrieval.plotting import save_raw_spectrum_plot
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
        help="Output .npy path. The file will contain wavelength_Angstrom, raw_pRT_flux.",
    )
    parser.add_argument("--wavelength-min-micron", type=float, default=None)
    parser.add_argument("--wavelength-max-micron", type=float, default=None)
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
    logger.info("Wavelength bounds: %.6f-%.6f micron", *wavelength_bounds)
    logger.info("Parameters: %s", parameters)

    wavelengths_cm, flux, metadata = generate_prt_emission_model(
        config=config,
        parameters=parameters,
        wavelength_boundaries_micron=wavelength_bounds,
        logger=logger,
    )

    wavelengths_angstrom = wavelengths_cm / ANGSTROM_TO_CM
    export = np.column_stack([wavelengths_angstrom, flux])
    np.save(output_path, export)

    plot_path = output_path.with_suffix(".png")
    save_raw_spectrum_plot(wavelengths_cm, flux, plot_path)

    logger.info("Model metadata: %s", metadata)
    logger.info("Saved xcorr model array with shape %s to %s", export.shape, output_path)
    logger.info("Column 0 is wavelength in Angstrom; existing xcorr code divides by 10 to get nm.")
    logger.info("Column 1 is raw pRT emission flux before convolution/template_to_dmag.")
    logger.info("Saved raw model diagnostic plot to %s", plot_path)


if __name__ == "__main__":
    main()
