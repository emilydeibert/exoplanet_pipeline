from astropy.stats import sigma_clip
from pathlib import Path
import argparse
import importlib.util
import json
import csv
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument("project_path", type=str)
parser.add_argument("--model", required=True)
parser.add_argument("--night", required=True)
parser.add_argument("--camera", required=True)
parser.add_argument("--k", type=int, required=True)

parser.add_argument("--wavelength-min-nm", type=float, default=383.0)
parser.add_argument("--wavelength-max-nm", type=float, default=1000.0)
parser.add_argument("--min-overlap-fraction", type=float, default=0.5)

parser.add_argument("--min-valid-flux-fraction", type=float, default=0.5)
parser.add_argument("--expected-kp", type=float, default=None)
parser.add_argument("--expected-rv", type=float, default=None)
parser.add_argument("--kp-window", type=float, default=10.0)
parser.add_argument("--rv-window", type=float, default=5.0)
parser.add_argument("--sigma-cut", type=float, default=3.0)
parser.add_argument("--map-sign", type=float, default=-1.0)

parser.add_argument(
    "--delta-peak-sign",
    choices=["negative", "positive", "absolute"],
    default="negative",
    help="Expected sign of Delta CCF recovery peak after applying map_sign.",
)

parser.add_argument("--output-dir", default=None)

args = parser.parse_args()

project_path = Path(args.project_path)

config_file = project_path / "config.py"
config_spec = importlib.util.spec_from_file_location("config", str(config_file))
config = importlib.util.module_from_spec(config_spec)
config_spec.loader.exec_module(config)

params_file = project_path / "parameters.py"
params_spec = importlib.util.spec_from_file_location("parameters", str(params_file))
params = importlib.util.module_from_spec(params_spec)
params_spec.loader.exec_module(params)


def get_order_wave_nm(wave, order):
    if wave.ndim == 3:
        return wave[order, 0, :]
    if wave.ndim == 2:
        return wave[order, :]
    raise ValueError(f"Unexpected wave shape: {wave.shape}")


def get_order_flux(flux, order):
    if flux is None:
        return None

    if flux.ndim == 3:
        # expected shape: n_orders, n_exp, n_pix OR n_exp, n_orders, n_pix
        if order < flux.shape[0]:
            return flux[order]
        if order < flux.shape[1]:
            return flux[:, order, :]

    return None


def overlap_fraction(order_wave, wmin, wmax):
    finite = np.isfinite(order_wave)
    if np.sum(finite) == 0:
        return 0.0

    usable = finite & (order_wave >= wmin) & (order_wave <= wmax)
    return float(np.sum(usable) / np.sum(finite))


def calculate_noise(kpvsys_map, sigma_cut=3.0):
    x = kpvsys_map - np.nanmedian(kpvsys_map)

    clipped = sigma_clip(
        x,
        sigma_upper=sigma_cut,
        sigma_lower=100,
    )

    return float(np.nanstd(clipped))


def snr_map_from_noise(signal_map, noise):
    x = signal_map - np.nanmedian(signal_map)
    return x / noise


def find_peak(snr_map, RV, Kp):
    max_index = np.nanargmax(snr_map)

    kp_idx, rv_idx = np.unravel_index(
        max_index,
        snr_map.shape,
    )

    return {
        "snr": float(snr_map[kp_idx, rv_idx]),
        "rv": float(RV[rv_idx]),
        "kp": float(Kp[kp_idx]),
        "rv_idx": int(rv_idx),
        "kp_idx": int(kp_idx),
    }


def find_peak_near_expected(
    snr_map,
    RV,
    Kp,
    expected_kp,
    expected_rv,
    kp_window,
    rv_window,
):
    kp_mask = (Kp >= expected_kp - kp_window) & (Kp <= expected_kp + kp_window)
    rv_mask = (RV >= expected_rv - rv_window) & (RV <= expected_rv + rv_window)

    masked = np.full_like(snr_map, np.nan)
    masked[np.ix_(kp_mask, rv_mask)] = snr_map[np.ix_(kp_mask, rv_mask)]

    return find_peak(masked, RV, Kp)


def delta_recovery_map(delta_snr_map, sign):
    if sign == "negative":
        return -1.0 * delta_snr_map
    if sign == "positive":
        return delta_snr_map
    if sign == "absolute":
        return np.abs(delta_snr_map)

    raise ValueError(sign)


def metrics_for_map(
    signal_map,
    noise_map,
    RV,
    Kp,
    expected_kp,
    expected_rv,
    kp_window,
    rv_window,
    sigma_cut,
):
    noise = calculate_noise(noise_map, sigma_cut=sigma_cut)
    snr_map = snr_map_from_noise(signal_map, noise)

    expected = find_peak_near_expected(
        snr_map,
        RV,
        Kp,
        expected_kp,
        expected_rv,
        kp_window,
        rv_window,
    )

    global_peak = find_peak(
        snr_map,
        RV,
        Kp,
    )

    return {
        "noise": noise,
        "expected_snr": expected["snr"],
        "expected_kp": expected["kp"],
        "expected_rv": expected["rv"],
        "global_snr": global_peak["snr"],
        "global_kp": global_peak["kp"],
        "global_rv": global_peak["rv"],
    }


def sum_orders(fmap, positions, map_sign):
    return map_sign * np.nansum(fmap[positions], axis=0)


def main():
    RV = np.asarray(config.RV, dtype=float)
    Kp = np.asarray(config.Kp, dtype=float)

    RV_MIN = config.RV_MIN if hasattr(config, "RV_MIN") else -75
    RV_MAX = config.RV_MAX if hasattr(config, "RV_MAX") else 75
    KP_MIN = config.KP_MIN if hasattr(config, "KP_MIN") else np.nanmin(Kp)
    KP_MAX = config.KP_MAX if hasattr(config, "KP_MAX") else np.nanmax(Kp)

    rv_mask = (RV >= RV_MIN) & (RV <= RV_MAX)
    kp_mask = (Kp >= KP_MIN) & (Kp <= KP_MAX)

    RV_crop = RV[rv_mask]
    Kp_crop = Kp[kp_mask]

    expected_kp = args.expected_kp if args.expected_kp is not None else float(params.K_p)
    expected_rv = args.expected_rv if args.expected_rv is not None else float(params.Vsys)

    base = Path(config.path2reduced)

    obs_file = base / "results" / f"{args.night}_{args.camera}_{args.model}_k{args.k}_iters.npz"
    pos_file = base / "injected" / f"{args.night}_{args.camera}_{args.model}_{args.k}_iters_injected_positive.npz"
    wave_file = base / f"{args.night}_{args.camera}_analysis_ready.npz"

    if not obs_file.exists():
        raise FileNotFoundError(obs_file)

    if not wave_file.exists():
        raise FileNotFoundError(wave_file)

    with np.load(obs_file) as data:
        obs_fmap = data["fmap"]
        saved_orders = data["orders"] if "orders" in data.files else np.arange(obs_fmap.shape[0])

    has_positive_injection = pos_file.exists()

    if has_positive_injection:
        with np.load(pos_file) as data:
            pos_fmap = data["fmap"]
    else:
        pos_fmap = None
        print(f"Warning: positive injection file not found: {pos_file}")
        print("Delta CCF diagnostics will be NaN.")

    with np.load(wave_file) as data:
        wave = data["wave"]

        if "norm_flux" in data.files:
            flux = data["norm_flux"]
        elif "flux" in data.files:
            flux = data["flux"]
        else:
            flux = None

    if args.output_dir is None:
        output_dir = base / "results" / f"order_diagnostics_{args.night}_{args.camera}_{args.model}_k{args.k}"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for row_idx, original_order in enumerate(saved_orders):
        original_order = int(original_order)

        order_wave = get_order_wave_nm(wave, original_order)
        finite_wave = np.isfinite(order_wave)

        wave_min = float(np.nanmin(order_wave)) if np.any(finite_wave) else np.nan
        wave_max = float(np.nanmax(order_wave)) if np.any(finite_wave) else np.nan
        wave_median = float(np.nanmedian(order_wave)) if np.any(finite_wave) else np.nan

        overlap = overlap_fraction(
            order_wave,
            args.wavelength_min_nm,
            args.wavelength_max_nm,
        )

        include_wavelength = overlap >= args.min_overlap_fraction

        order_flux = get_order_flux(flux, original_order)
        if order_flux is not None:
            valid_flux_fraction = float(np.mean(np.isfinite(order_flux)))
        else:
            valid_flux_fraction = np.nan

        include_quality = (
            True
            if not np.isfinite(valid_flux_fraction)
            else valid_flux_fraction >= args.min_valid_flux_fraction
        )

        include_baseline = include_wavelength and include_quality

        obs_order_map = args.map_sign * obs_fmap[row_idx]
        obs_order_crop = obs_order_map[kp_mask][:, rv_mask]

        obs_metrics = metrics_for_map(
            signal_map=obs_order_crop,
            noise_map=obs_order_crop,
            RV=RV_crop,
            Kp=Kp_crop,
            expected_kp=expected_kp,
            expected_rv=expected_rv,
            kp_window=args.kp_window,
            rv_window=args.rv_window,
            sigma_cut=args.sigma_cut,
        )

        if has_positive_injection:
            pos_order_map = args.map_sign * pos_fmap[row_idx]
            pos_order_crop = pos_order_map[kp_mask][:, rv_mask]
            delta_order_crop = pos_order_crop - obs_order_crop

            noise_obs_order = calculate_noise(obs_order_crop, sigma_cut=args.sigma_cut)
            delta_snr_map = snr_map_from_noise(delta_order_crop, noise_obs_order)
            delta_rec_map = delta_recovery_map(delta_snr_map, args.delta_peak_sign)

            delta_expected = find_peak_near_expected(
                delta_rec_map,
                RV_crop,
                Kp_crop,
                expected_kp,
                expected_rv,
                args.kp_window,
                args.rv_window,
            )

            delta_global = find_peak(delta_rec_map, RV_crop, Kp_crop)

            delta_expected_snr = delta_expected["snr"]
            delta_global_snr = delta_global["snr"]
        else:
            delta_expected_snr = np.nan
            delta_global_snr = np.nan

        flag_reasons = []

        if not include_wavelength:
            flag_reasons.append("outside_wavelength_range")
        if not include_quality:
            flag_reasons.append("low_valid_flux_fraction")

        rows.append(
            {
                "row_idx": int(row_idx),
                "original_order": original_order,
                "wave_min_nm": wave_min,
                "wave_max_nm": wave_max,
                "wave_median_nm": wave_median,
                "usable_overlap_fraction": overlap,
                "valid_flux_fraction": valid_flux_fraction,
                "include_wavelength": bool(include_wavelength),
                "include_quality": bool(include_quality),
                "include_baseline": bool(include_baseline),
                "single_obs_expected_snr": obs_metrics["expected_snr"],
                "single_obs_global_snr": obs_metrics["global_snr"],
                "single_obs_global_kp": obs_metrics["global_kp"],
                "single_obs_global_rv": obs_metrics["global_rv"],
                "single_delta_expected_snr": float(delta_expected_snr),
                "single_delta_global_snr": float(delta_global_snr),
                "flag_reasons": ";".join(flag_reasons),
            }
        )

    baseline_positions = np.array(
        [r["row_idx"] for r in rows if r["include_baseline"]],
        dtype=int,
    )

    if len(baseline_positions) == 0:
        raise ValueError("No baseline orders survived the wavelength/quality cuts.")

    obs_baseline = sum_orders(obs_fmap, baseline_positions, args.map_sign)
    obs_baseline_crop = obs_baseline[kp_mask][:, rv_mask]

    obs_baseline_metrics = metrics_for_map(
        signal_map=obs_baseline_crop,
        noise_map=obs_baseline_crop,
        RV=RV_crop,
        Kp=Kp_crop,
        expected_kp=expected_kp,
        expected_rv=expected_rv,
        kp_window=args.kp_window,
        rv_window=args.rv_window,
        sigma_cut=args.sigma_cut,
    )

    if has_positive_injection:
        pos_baseline = sum_orders(pos_fmap, baseline_positions, args.map_sign)
        pos_baseline_crop = pos_baseline[kp_mask][:, rv_mask]

        delta_baseline_crop = pos_baseline_crop - obs_baseline_crop
        noise_obs_baseline = calculate_noise(obs_baseline_crop, sigma_cut=args.sigma_cut)

        delta_baseline_snr_map = snr_map_from_noise(delta_baseline_crop, noise_obs_baseline)
        delta_baseline_rec_map = delta_recovery_map(delta_baseline_snr_map, args.delta_peak_sign)

        delta_baseline_expected = find_peak_near_expected(
            delta_baseline_rec_map,
            RV_crop,
            Kp_crop,
            expected_kp,
            expected_rv,
            args.kp_window,
            args.rv_window,
        )

        delta_baseline_global = find_peak(delta_baseline_rec_map, RV_crop, Kp_crop)

        delta_baseline_metrics = {
            "expected_snr": delta_baseline_expected["snr"],
            "expected_kp": delta_baseline_expected["kp"],
            "expected_rv": delta_baseline_expected["rv"],
            "global_snr": delta_baseline_global["snr"],
            "global_kp": delta_baseline_global["kp"],
            "global_rv": delta_baseline_global["rv"],
        }
    else:
        delta_baseline_metrics = {
            "expected_snr": np.nan,
            "expected_kp": np.nan,
            "expected_rv": np.nan,
            "global_snr": np.nan,
            "global_kp": np.nan,
            "global_rv": np.nan,
        }

    # Leave-one-out diagnostics
    for r in rows:
        if not r["include_baseline"]:
            r["loo_obs_expected_snr"] = np.nan
            r["loo_delta_expected_snr"] = np.nan
            r["loo_obs_minus_baseline"] = np.nan
            r["loo_delta_minus_baseline"] = np.nan
            continue

        leave_out = r["row_idx"]

        loo_positions = np.array(
            [p for p in baseline_positions if p != leave_out],
            dtype=int,
        )

        obs_loo = sum_orders(obs_fmap, loo_positions, args.map_sign)
        obs_loo_crop = obs_loo[kp_mask][:, rv_mask]

        obs_loo_metrics = metrics_for_map(
            signal_map=obs_loo_crop,
            noise_map=obs_loo_crop,
            RV=RV_crop,
            Kp=Kp_crop,
            expected_kp=expected_kp,
            expected_rv=expected_rv,
            kp_window=args.kp_window,
            rv_window=args.rv_window,
            sigma_cut=args.sigma_cut,
        )

        r["loo_obs_expected_snr"] = obs_loo_metrics["expected_snr"]
        r["loo_obs_minus_baseline"] = (
            obs_loo_metrics["expected_snr"] - obs_baseline_metrics["expected_snr"]
        )

        if has_positive_injection:
            pos_loo = sum_orders(pos_fmap, loo_positions, args.map_sign)
            pos_loo_crop = pos_loo[kp_mask][:, rv_mask]

            delta_loo_crop = pos_loo_crop - obs_loo_crop
            noise_obs_loo = calculate_noise(obs_loo_crop, sigma_cut=args.sigma_cut)

            delta_loo_snr_map = snr_map_from_noise(delta_loo_crop, noise_obs_loo)
            delta_loo_rec_map = delta_recovery_map(delta_loo_snr_map, args.delta_peak_sign)

            delta_loo_expected = find_peak_near_expected(
                delta_loo_rec_map,
                RV_crop,
                Kp_crop,
                expected_kp,
                expected_rv,
                args.kp_window,
                args.rv_window,
            )

            r["loo_delta_expected_snr"] = delta_loo_expected["snr"]
            r["loo_delta_minus_baseline"] = (
                delta_loo_expected["snr"] - delta_baseline_metrics["expected_snr"]
            )
        else:
            r["loo_delta_expected_snr"] = np.nan
            r["loo_delta_minus_baseline"] = np.nan

    csv_path = output_dir / "order_diagnostics.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    baseline_orders = [
        int(r["original_order"])
        for r in rows
        if r["include_baseline"]
    ]

    summary = {
        "project_path": str(project_path),
        "model": args.model,
        "night": args.night,
        "camera": args.camera,
        "k": args.k,
        "wavelength_min_nm": args.wavelength_min_nm,
        "wavelength_max_nm": args.wavelength_max_nm,
        "min_overlap_fraction": args.min_overlap_fraction,
        "min_valid_flux_fraction": args.min_valid_flux_fraction,
        "expected_kp": expected_kp,
        "expected_rv": expected_rv,
        "delta_peak_sign": args.delta_peak_sign,
        "baseline_orders": baseline_orders,
        "baseline_obs_expected_snr": obs_baseline_metrics["expected_snr"],
        "baseline_obs_expected_kp": obs_baseline_metrics["expected_kp"],
        "baseline_obs_expected_rv": obs_baseline_metrics["expected_rv"],
        "baseline_obs_global_snr": obs_baseline_metrics["global_snr"],
        "baseline_delta_expected_snr": delta_baseline_metrics["expected_snr"],
        "baseline_delta_expected_kp": delta_baseline_metrics["expected_kp"],
        "baseline_delta_expected_rv": delta_baseline_metrics["expected_rv"],
        "baseline_delta_global_snr": delta_baseline_metrics["global_snr"],
        "obs_file": str(obs_file),
        "pos_file": str(pos_file),
        "wave_file": str(wave_file),
    }

    summary_path = output_dir / "order_diagnostics_summary.json"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print()
    print("=" * 60)
    print("Order diagnostics complete")
    print("=" * 60)
    print(f"Saved CSV     : {csv_path}")
    print(f"Saved summary : {summary_path}")
    print()
    print("Baseline wavelength/quality-selected orders:")
    print(" ".join(str(o) for o in baseline_orders))
    print()
    print("Baseline observed expected-window SNR:")
    print(summary["baseline_obs_expected_snr"])
    print()
    print("Baseline delta expected-window SNR:")
    print(summary["baseline_delta_expected_snr"])
    print("=" * 60)


if __name__ == "__main__":
    main()