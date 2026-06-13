"""Run a minimal pRT Fe emission smoke test on one order/chunk."""

from __future__ import annotations

import argparse
from pathlib import Path

from retrieval.likelihood import (
    build_prepared_model_for_parameters,
    load_retrieval_data,
    model_wavelength_bounds_for_data,
)
from retrieval.plotting import (
    save_data_mask_plot,
    save_prepared_model_plot,
    save_raw_spectrum_plot,
    save_shifted_model_plot,
)
from retrieval.prt_emission_model import (
    generate_prt_emission_model,
    load_yaml_config,
    log_run_summary,
    parameters_from_config,
    require_numpy,
    setup_logging,
    shifted_model_cube,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="retrieval/configs/mascara1b_fe_smoketest.yaml",
        help="YAML retrieval config.",
    )
    parser.add_argument("--order", type=int, default=None, help="Override selection.orders with one order.")
    parser.add_argument("--pixel-start", type=int, default=None, help="Override selection.pixel_start.")
    parser.add_argument("--pixel-stop", type=int, default=None, help="Override selection.pixel_stop.")
    parser.add_argument("--wavelength-min-micron", type=float, default=None)
    parser.add_argument("--wavelength-max-micron", type=float, default=None)
    return parser.parse_args()


def apply_selection_overrides(config: dict, args: argparse.Namespace) -> None:
    selection = config.setdefault("selection", {})
    if args.order is not None:
        selection["orders"] = [args.order]
    if args.pixel_start is not None:
        selection["pixel_start"] = args.pixel_start
    if args.pixel_stop is not None:
        selection["pixel_stop"] = args.pixel_stop
    if args.wavelength_min_micron is not None:
        selection["wavelength_min_micron"] = args.wavelength_min_micron
    if args.wavelength_max_micron is not None:
        selection["wavelength_max_micron"] = args.wavelength_max_micron


def main() -> None:
    args = parse_args()
    np = require_numpy()
    config = load_yaml_config(args.config)
    apply_selection_overrides(config, args)

    output_dir = Path(config.get("output", {}).get("directory", "retrieval/results/mascara1b_fe_smoketest"))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir, "prt_smoketest.log")

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

    shifted = shifted_model_cube(
        rest_wavelengths_cm=rest_wave_cm,
        rest_flux=rest_flux,
        observed_wavelengths_cm=data.wavelengths_cm,
        phases=data.phases,
        Kp=parameters["Kp"],
        Vsys=parameters["Vsys"],
        resolving_power=float(config["instrument"]["resolving_power"]),
        barycentric_velocities=data.barycentric_velocities,
        velocity_config=config.get("velocity", {}),
    )
    prepared = build_prepared_model_for_parameters(
        data=data,
        rest_wavelengths_cm=rest_wave_cm,
        rest_flux=rest_flux,
        config=config,
        Kp=parameters["Kp"],
        Vsys=parameters["Vsys"],
    )

    np.savez_compressed(
        output_dir / "prt_smoketest_model_arrays.npz",
        rest_wavelengths_cm=rest_wave_cm,
        rest_flux=rest_flux,
        shifted_model=shifted,
        prepared_model=prepared,
        observed_wavelengths_cm=data.wavelengths_cm,
        phases=data.phases,
    )

    save_raw_spectrum_plot(rest_wave_cm, rest_flux, output_dir / "a_raw_prt_emission_spectrum.png")
    save_prepared_model_plot(data.wavelengths_cm, prepared, output_dir / "b_prepared_highpass_model.png")
    save_shifted_model_plot(
        data.wavelengths_cm,
        shifted,
        data.phases,
        output_dir / "c_model_shifted_over_phases.png",
    )
    save_data_mask_plot(data, shifted, output_dir / "d_data_dimensions_and_mask.png")

    logger.info("Smoke-test metadata: %s", metadata)
    logger.info("Saved pRT smoke-test diagnostics to %s", output_dir)


if __name__ == "__main__":
    main()
