"""Run the gated Fe-only dynesty retrieval after validating the Fe grid."""

from __future__ import annotations

import argparse
from pathlib import Path

from retrieval.likelihood import load_retrieval_data, model_wavelength_bounds_for_data, run_sampler
from retrieval.prt_emission_model import (
    load_yaml_config,
    log_run_summary,
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
        "--confirm-grid-validated",
        action="store_true",
        help="Allow the sampler to run after you have inspected the Fe Kp-Vsys grid.",
    )
    parser.add_argument("--order", type=int, default=None, help="Override selection.orders with one order/chunk.")
    parser.add_argument("--pixel-start", type=int, default=None)
    parser.add_argument("--pixel-stop", type=int, default=None)
    return parser.parse_args()


def apply_overrides(config: dict, args: argparse.Namespace) -> None:
    selection = config.setdefault("selection", {})
    if args.order is not None:
        selection["orders"] = [args.order]
    if args.pixel_start is not None:
        selection["pixel_start"] = args.pixel_start
    if args.pixel_stop is not None:
        selection["pixel_stop"] = args.pixel_stop
    if args.confirm_grid_validated:
        config.setdefault("sampler", {})["confirm_grid_validated"] = True


def main() -> None:
    args = parse_args()
    require_numpy()
    config = load_yaml_config(args.config)
    apply_overrides(config, args)

    output_dir = Path(config.get("output", {}).get("directory", "retrieval/results/mascara1b_fe_smoketest"))
    sampler_dir = output_dir / "fe_only_sampler"
    sampler_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(sampler_dir, "fe_only_sampler.log")

    parameters = parameters_from_config(config)
    data = load_retrieval_data(config, logger=logger)
    log_run_summary(logger, config, parameters, wavelengths_cm=data.wavelengths_cm, mask=data.good_mask)

    wavelength_bounds = model_wavelength_bounds_for_data(data, config)
    run_sampler(
        data=data,
        config=config,
        wavelength_boundaries_micron=wavelength_bounds,
        output_dir=sampler_dir,
        logger=logger,
    )


if __name__ == "__main__":
    main()
