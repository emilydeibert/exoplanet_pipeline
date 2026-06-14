"""Validate that a pRT xcorr_processed model reproduces the known CCF map."""

from __future__ import annotations

import argparse
from pathlib import Path

from retrieval.prt_emission_model import setup_logging

from .ccf_likelihood import (
    calculate_snr_map,
    compute_xcorr_detection_map,
    crop_like_getresults,
    find_peak,
    save_validation_plot,
    write_json,
)
from .data_loading import block_summary, load_hrccs_data, load_project_modules, parse_int_list, split_cli_list
from .model_builder import build_prt_xcorr_template, load_retrieval_config_and_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_path", help="Existing exopipe project directory containing config.py and parameters.py.")
    parser.add_argument("--retrieval-config", required=True, help="YAML pRT retrieval config.")
    parser.add_argument("--k", type=int, required=True, help="SYSREM iteration, using sysrem[k-1].")
    parser.add_argument("--nights", nargs="+", default=None, help="Nights to include. Also accepts comma-separated values.")
    parser.add_argument("--cameras", nargs="+", default=None, help="Cameras to include. Also accepts comma-separated values.")
    parser.add_argument("--orders", nargs="+", default=None, help="Original order indices. Also accepts comma-separated values.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--sigma-cut", type=float, default=3.0)
    parser.add_argument("--save-per-order", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir, "validate_xcorr_model.log")
    logger.info("Starting HRCCS xcorr validation")

    exopipe_config, exopipe_params = load_project_modules(args.project_path)
    retrieval_config, parameters = load_retrieval_config_and_parameters(args.retrieval_config)

    template = build_prt_xcorr_template(
        retrieval_config=retrieval_config,
        exopipe_config=exopipe_config,
        parameters=parameters,
        logger=logger,
    )

    nights = split_cli_list(args.nights)
    cameras = split_cli_list(args.cameras)
    orders = parse_int_list(args.orders)
    blocks = load_hrccs_data(
        config=exopipe_config,
        params=exopipe_params,
        k=args.k,
        nights=nights,
        cameras=cameras,
        model_array=template["model_array"],
        orders=orders,
        logger=logger,
    )

    maps = compute_xcorr_detection_map(
        blocks=blocks,
        F_model=template["F_model"],
        RV=exopipe_config.RV,
        Kp=exopipe_config.Kp,
        save_per_order=bool(args.save_per_order),
        logger=logger,
    )
    cropped = crop_like_getresults(exopipe_config, maps["combined_map"])
    snr_map, noise = calculate_snr_map(cropped["combined_map"], sigma_cut=args.sigma_cut)
    peak = find_peak(snr_map, cropped["RV"], cropped["Kp"])

    import numpy as np

    np.savez_compressed(
        output_dir / "validate_xcorr_model_maps.npz",
        combined_map=cropped["combined_map"],
        combined_map_full=maps["combined_map"],
        snr_map=snr_map,
        RV=cropped["RV"],
        Kp=cropped["Kp"],
        RV_full=exopipe_config.RV,
        Kp_full=exopipe_config.Kp,
        selected_orders=np.asarray([order for block in blocks for order in block.orders], dtype=int),
    )
    if args.save_per_order and "per_order_maps" in maps:
        np.savez_compressed(
            output_dir / "validate_xcorr_model_per_order_maps.npz",
            per_order_maps=maps["per_order_maps"],
            per_order_labels=maps["per_order_labels"],
            RV_full=exopipe_config.RV,
            Kp_full=exopipe_config.Kp,
        )

    save_validation_plot(cropped["RV"], cropped["Kp"], snr_map, peak, output_dir / "validate_xcorr_model.png")

    summary = {
        "project_path": str(args.project_path),
        "retrieval_config": str(args.retrieval_config),
        "sysrem_iteration": int(args.k),
        "nights": [block.night for block in blocks],
        "cameras": [block.camera for block in blocks],
        "data": block_summary(blocks),
        "prt_parameters": parameters,
        "peak": peak,
        "noise": float(noise),
        "sigma_cut": float(args.sigma_cut),
        "model_metadata": template["metadata"],
    }
    write_json(output_dir / "validate_xcorr_model_summary.json", summary)

    logger.info("Peak SNR %.3f at Kp=%.3f km/s, Vsys=%.3f km/s", peak["snr"], peak["kp"], peak["rv"])
    logger.info("Saved validation outputs to %s", output_dir)


if __name__ == "__main__":
    main()
