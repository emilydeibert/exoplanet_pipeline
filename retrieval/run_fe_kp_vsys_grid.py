"""Run an Fe-only Kp-Vsys likelihood grid using the pRT emission model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from retrieval.likelihood import (
    build_prepared_model_for_parameters,
    inject_fake_signal,
    load_retrieval_data,
    model_wavelength_bounds_for_data,
    run_kp_vsys_grid,
    save_grid_results,
)
from retrieval.plotting import save_best_fit_model_plot, save_kp_vsys_likelihood_plot
from retrieval.prt_emission_model import (
    build_convolved_model,
    generate_prt_emission_model,
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
    parser.add_argument("--order", type=int, default=None, help="Override selection.orders with one order.")
    parser.add_argument("--pixel-start", type=int, default=None, help="Override selection.pixel_start.")
    parser.add_argument("--pixel-stop", type=int, default=None, help="Override selection.pixel_stop.")
    parser.add_argument("--inject-fake", action="store_true", help="Inject a fake prepared Fe model before mapping.")
    parser.add_argument("--noise-only", action="store_true", help="Use noise instead of real residuals for injection.")
    parser.add_argument("--injection-kp", type=float, default=None)
    parser.add_argument("--injection-vsys", type=float, default=None)
    parser.add_argument("--injection-scale", type=float, default=None)
    parser.add_argument("--n-jobs", type=int, default=None, help="Parallel grid workers. Use 1 for serial mode.")
    parser.add_argument(
        "--tiny-grid",
        action="store_true",
        help="Run a 3x3 grid around expected_detection for quick timing/debug checks.",
    )
    return parser.parse_args()


def apply_overrides(config: dict, args: argparse.Namespace) -> None:
    selection = config.setdefault("selection", {})
    if args.order is not None:
        selection["orders"] = [args.order]
    if args.pixel_start is not None:
        selection["pixel_start"] = args.pixel_start
    if args.pixel_stop is not None:
        selection["pixel_stop"] = args.pixel_stop
    grid = config.setdefault("grid", {})
    if args.n_jobs is not None:
        grid["n_jobs"] = args.n_jobs
    if args.tiny_grid:
        expected = config.get("expected_detection", {"Kp": 198.0, "Vsys": -2.0})
        expected_kp = float(expected.get("Kp", 198.0))
        expected_vsys = float(expected.get("Vsys", -2.0))
        grid["kp_min"] = expected_kp - 2.0
        grid["kp_max"] = expected_kp + 2.0
        grid["kp_step"] = 2.0
        grid["vsys_min"] = expected_vsys - 1.0
        grid["vsys_max"] = expected_vsys + 1.0
        grid["vsys_step"] = 1.0


def main() -> None:
    args = parse_args()
    require_numpy()
    config = load_yaml_config(args.config)
    apply_overrides(config, args)

    output_dir = Path(config.get("output", {}).get("directory", "retrieval/results/mascara1b_fe_smoketest"))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir, "fe_kp_vsys_grid.log")

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
    logger.info("Model metadata: %s", metadata)

    convolve_start = time.perf_counter()
    convolved_model = build_convolved_model(
        rest_wave_cm,
        rest_flux,
        float(config["instrument"]["resolving_power"]),
    )
    logger.info(
        "Built reusable convolved model cache at R=%.1f for grid/injection helpers in %.3fs",
        convolved_model.resolving_power,
        time.perf_counter() - convolve_start,
    )

    injection_summary = None
    if args.inject_fake:
        injection_cfg = config.get("injection", {})
        injection_kp = float(args.injection_kp if args.injection_kp is not None else injection_cfg.get("Kp", 198.0))
        injection_vsys = float(args.injection_vsys if args.injection_vsys is not None else injection_cfg.get("Vsys", -2.0))
        injection_scale = float(
            args.injection_scale if args.injection_scale is not None else injection_cfg.get("scale", 1.0)
        )
        injected_model = build_prepared_model_for_parameters(
            data=data,
            rest_wavelengths_cm=rest_wave_cm,
            rest_flux=rest_flux,
            config=config,
            Kp=injection_kp,
            Vsys=injection_vsys,
            convolved_rest_flux=convolved_model.flux,
        )
        data = inject_fake_signal(
            data=data,
            prepared_model_cube=injected_model,
            scale=injection_scale,
            noise_only=bool(args.noise_only),
            seed=int(injection_cfg.get("seed", 12345)),
        )
        injection_summary = {
            "Kp": injection_kp,
            "Vsys": injection_vsys,
            "scale": injection_scale,
            "noise_only": bool(args.noise_only),
        }
        logger.info("Injected fake signal: %s", injection_summary)

    results = run_kp_vsys_grid(
        data=data,
        rest_wavelengths_cm=rest_wave_cm,
        rest_flux=rest_flux,
        config=config,
        parameters=parameters,
        convolved_rest_flux=convolved_model.flux,
        n_jobs=int(config.get("grid", {}).get("n_jobs", 1)),
        logger=logger,
    )

    stem = "fe_kp_vsys_grid_injected" if args.inject_fake else "fe_kp_vsys_grid"
    save_grid_results(
        results,
        output_npz=output_dir / f"{stem}.npz",
        output_json=output_dir / f"{stem}_summary.json",
    )

    expected = config.get("expected_detection", {"Kp": 198.0, "Vsys": -2.0})
    save_kp_vsys_likelihood_plot(
        results["Kp_grid"],
        results["Vsys_grid"],
        results["log_likelihood"],
        output_dir / f"{stem}.png",
        expected=expected,
    )

    best = results["best"]
    best_model = build_prepared_model_for_parameters(
        data=data,
        rest_wavelengths_cm=rest_wave_cm,
        rest_flux=rest_flux,
        config=config,
        Kp=best["Kp"],
        Vsys=best["Vsys"],
        convolved_rest_flux=convolved_model.flux,
    )
    save_best_fit_model_plot(
        data,
        best_model,
        amplitude=best["amplitude"],
        filename=output_dir / f"{stem}_best_model_order0_exp0.png",
    )

    tolerance = config.get("expected_detection", {})
    kp_tol = float(tolerance.get("kp_tolerance", 10.0))
    vsys_tol = float(tolerance.get("vsys_tolerance", 5.0))
    near_expected = (
        abs(best["Kp"] - float(expected.get("Kp", 198.0))) <= kp_tol
        and abs(best["Vsys"] - float(expected.get("Vsys", -2.0))) <= vsys_tol
    )

    final_summary = {
        "best": best,
        "near_expected_detection": bool(near_expected),
        "expected": expected,
        "injection": injection_summary,
    }
    with (output_dir / f"{stem}_run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(final_summary, handle, indent=2, sort_keys=True)

    logger.info("Grid near expected detection: %s", near_expected)
    logger.info("Saved Fe Kp-Vsys grid outputs to %s", output_dir)


if __name__ == "__main__":
    main()
