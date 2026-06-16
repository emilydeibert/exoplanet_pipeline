"""
Diagnose SYSREM iteration choices from normal, injected, and delta-CCF maps.

This script reads the existing xcorr products:
  results/{night}_{camera}_{model}_k{k}_iters.npz
  injected/{night}_{camera}_{model}_{k}_iters_injected_positive.npz
  injected/{night}_{camera}_{model}_{k}_iters_injected_negative.npz

It writes:
  sysrem_diagnostics.csv
  sysrem_diagnostics.png

Conventions:
  - --orders are original detector/order numbers, not fmap row indices.
  - wavelength cuts are applied using {night}_{camera}_analysis_ready.npz["wave"].
  - map_sign=-1 matches the current getResults.py convention for emission.
"""

from __future__ import annotations

from astropy.stats import sigma_clip
from pathlib import Path
import argparse
import csv
import importlib.util
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_order_wave_nm(wave: np.ndarray, order: int) -> np.ndarray:
    """
    Return 1D wavelength array for an original order.

    Expected wave shapes:
      (n_orders, n_exp, n_pix)
      (n_orders, n_pix)
    """
    if wave.ndim == 3:
        return wave[order, 0, :]
    if wave.ndim == 2:
        return wave[order, :]
    raise ValueError(f"Unexpected wave shape: {wave.shape}")


def order_overlap_fraction(
    order_wave_nm: np.ndarray,
    wavelength_min_nm: float | None,
    wavelength_max_nm: float | None,
) -> float:
    finite = np.isfinite(order_wave_nm)
    if not np.any(finite):
        return 0.0

    ok = finite.copy()
    if wavelength_min_nm is not None:
        ok &= order_wave_nm >= wavelength_min_nm
    if wavelength_max_nm is not None:
        ok &= order_wave_nm <= wavelength_max_nm

    return float(np.sum(ok) / np.sum(finite))


def select_order_positions(
    saved_orders: np.ndarray,
    fmap: np.ndarray,
    wave: np.ndarray | None,
    requested_orders: list[int] | None,
    wavelength_min_nm: float | None,
    wavelength_max_nm: float | None,
    min_overlap_fraction: float,
) -> tuple[np.ndarray, list[int], dict[int, float]]:
    """
    Convert original order numbers into fmap row positions.

    saved_orders[i] is the original detector order number corresponding to fmap[i].
    """
    saved_orders = np.asarray(saved_orders, dtype=int)

    if fmap.shape[0] != len(saved_orders):
        raise ValueError(
            f"fmap has {fmap.shape[0]} order rows but saved_orders has {len(saved_orders)} entries."
        )

    requested_set = None if requested_orders is None else set(int(o) for o in requested_orders)

    positions = []
    used_original_orders = []
    overlap_by_order = {}

    for pos, original_order in enumerate(saved_orders):
        if requested_set is not None and original_order not in requested_set:
            continue

        overlap = np.nan
        if wave is not None and (wavelength_min_nm is not None or wavelength_max_nm is not None):
            order_wave = get_order_wave_nm(wave, original_order)
            overlap = order_overlap_fraction(order_wave, wavelength_min_nm, wavelength_max_nm)
            if overlap < min_overlap_fraction:
                continue

        positions.append(pos)
        used_original_orders.append(int(original_order))
        overlap_by_order[int(original_order)] = float(overlap) if np.isfinite(overlap) else np.nan

    if len(positions) == 0:
        raise ValueError("No orders survived selection.")

    return np.asarray(positions, dtype=int), used_original_orders, overlap_by_order


def xcorr_filename(config, night: str, camera: str, model: str, k: int, kind: str) -> Path:
    base = Path(config.path2reduced)

    if kind == "obs":
        return base / "results" / f"{night}_{camera}_{model}_k{k}_iters.npz"

    if kind in {"positive", "negative"}:
        return base / "injected" / f"{night}_{camera}_{model}_{k}_iters_injected_{kind}.npz"

    raise ValueError(f"Unknown kind: {kind}")


def load_combined_map(
    config,
    model: str,
    k: int,
    nights: list[str],
    cameras: list[str],
    kind: str,
    requested_orders: list[int] | None,
    wavelength_min_nm: float | None,
    wavelength_max_nm: float | None,
    min_overlap_fraction: float,
    map_sign: float,
) -> tuple[np.ndarray, dict]:
    """
    Load and sum maps over nights/cameras/orders.
    """
    combined = []
    metadata = {
        "kind": kind,
        "k": int(k),
        "blocks": [],
    }

    for night in nights:
        for camera in cameras:
            filename = xcorr_filename(config, night, camera, model, k, kind)
            if not filename.exists():
                raise FileNotFoundError(f"Missing {kind} file: {filename}")

            with np.load(filename, allow_pickle=False) as data:
                fmap = data["fmap"]

                if "orders" in data:
                    saved_orders = np.asarray(data["orders"], dtype=int)
                else:
                    saved_orders = np.arange(fmap.shape[0], dtype=int)

            wave_file = Path(config.path2reduced) / f"{night}_{camera}_analysis_ready.npz"
            wave = None
            if wave_file.exists():
                with np.load(wave_file, allow_pickle=False) as data:
                    wave = data["wave"]

            positions, used_orders, overlap_by_order = select_order_positions(
                saved_orders=saved_orders,
                fmap=fmap,
                wave=wave,
                requested_orders=requested_orders,
                wavelength_min_nm=wavelength_min_nm,
                wavelength_max_nm=wavelength_max_nm,
                min_overlap_fraction=min_overlap_fraction,
            )

            order_sum = np.nansum(fmap[positions], axis=0)
            order_sum = map_sign * order_sum

            combined.append(order_sum)

            metadata["blocks"].append(
                {
                    "night": night,
                    "camera": camera,
                    "filename": str(filename),
                    "n_saved_orders": int(len(saved_orders)),
                    "n_used_orders": int(len(used_orders)),
                    "used_orders": used_orders,
                    "overlap_by_order": overlap_by_order,
                }
            )

    return np.nansum(combined, axis=0), metadata


def make_window_mask(Kp, RV, kp0, rv0, kp_half_width, rv_half_width):
    kp_mask = np.abs(Kp - kp0) <= kp_half_width
    rv_mask = np.abs(RV - rv0) <= rv_half_width
    return kp_mask[:, None] & rv_mask[None, :]


def robust_noise(
    noise_reference_map: np.ndarray,
    Kp: np.ndarray,
    RV: np.ndarray,
    exclude_kp: float | None,
    exclude_rv: float | None,
    kp_half_width: float,
    rv_half_width: float,
    sigma_cut: float,
) -> float:
    x = noise_reference_map - np.nanmedian(noise_reference_map)

    good = np.isfinite(x)

    if exclude_kp is not None and exclude_rv is not None:
        signal_window = make_window_mask(Kp, RV, exclude_kp, exclude_rv, kp_half_width, rv_half_width)
        good &= ~signal_window

    clipped = sigma_clip(
        x[good],
        sigma_upper=sigma_cut,
        sigma_lower=100.0,
        masked=True,
    )

    noise = float(np.nanstd(clipped))
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError("Noise estimate failed.")

    return noise


def find_global_peak(snr_map: np.ndarray, Kp: np.ndarray, RV: np.ndarray) -> dict:
    idx = np.nanargmax(snr_map)
    kp_idx, rv_idx = np.unravel_index(idx, snr_map.shape)

    return {
        "snr": float(snr_map[kp_idx, rv_idx]),
        "kp": float(Kp[kp_idx]),
        "rv": float(RV[rv_idx]),
        "kp_idx": int(kp_idx),
        "rv_idx": int(rv_idx),
    }


def find_window_peak(
    snr_map: np.ndarray,
    Kp: np.ndarray,
    RV: np.ndarray,
    target_kp: float,
    target_rv: float,
    kp_half_width: float,
    rv_half_width: float,
) -> dict:
    window = make_window_mask(Kp, RV, target_kp, target_rv, kp_half_width, rv_half_width)

    if not np.any(window):
        raise ValueError("Target window contains no grid points.")

    masked = np.where(window, snr_map, np.nan)
    return find_global_peak(masked, Kp, RV)


def map_metrics(
    signal_map: np.ndarray,
    noise_reference_map: np.ndarray,
    Kp: np.ndarray,
    RV: np.ndarray,
    target_kp: float,
    target_rv: float,
    kp_half_width: float,
    rv_half_width: float,
    sigma_cut: float,
) -> tuple[dict, np.ndarray]:
    signal_centered = signal_map - np.nanmedian(signal_map)
    noise_ref_centered = noise_reference_map - np.nanmedian(noise_reference_map)

    noise = robust_noise(
        noise_reference_map=noise_ref_centered,
        Kp=Kp,
        RV=RV,
        exclude_kp=target_kp,
        exclude_rv=target_rv,
        kp_half_width=kp_half_width,
        rv_half_width=rv_half_width,
        sigma_cut=sigma_cut,
    )

    snr_map = signal_centered / noise

    global_peak = find_global_peak(snr_map, Kp, RV)
    window_peak = find_window_peak(
        snr_map,
        Kp,
        RV,
        target_kp,
        target_rv,
        kp_half_width,
        rv_half_width,
    )

    metrics = {
        "noise": float(noise),
        "global_snr": global_peak["snr"],
        "global_kp": global_peak["kp"],
        "global_rv": global_peak["rv"],
        "window_snr": window_peak["snr"],
        "window_kp": window_peak["kp"],
        "window_rv": window_peak["rv"],
    }

    return metrics, snr_map


def safe_getattr(module, name, default):
    return getattr(module, name) if hasattr(module, name) else default


def plot_sysrem_diagnostics(rows: list[dict], output_png: Path):
    k = np.array([r["k"] for r in rows], dtype=int)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(k, [r["obs_window_snr"] for r in rows], marker="o", label="Observed, expected window")
    axes[0].plot(k, [r["obs_global_snr"] for r in rows], marker="o", linestyle="--", label="Observed, global")
    axes[0].set_ylabel("SNR")
    axes[0].set_title("Observed CCF")
    axes[0].legend()

    axes[1].plot(k, [r["neg_window_snr"] for r in rows], marker="o", label="Negative injection, target window")
    axes[1].plot(k, [r["neg_global_snr"] for r in rows], marker="o", linestyle="--", label="Negative injection, global")
    axes[1].set_ylabel("SNR")
    axes[1].set_title("Negative-injection CCF")
    axes[1].legend()

    axes[2].plot(k, [r["delta_window_snr"] for r in rows], marker="o", label="Delta CCF, expected window")
    axes[2].plot(k, [r["delta_global_snr"] for r in rows], marker="o", linestyle="--", label="Delta CCF, global")
    axes[2].set_xlabel("SYSREM iteration k")
    axes[2].set_ylabel("SNR")
    axes[2].set_title("Delta CCF = positive injection − observed")
    axes[2].legend()

    for ax in axes:
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_png, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("project_path", type=str)
    parser.add_argument("--model", required=True)

    parser.add_argument("--max-k", type=int, default=15)
    parser.add_argument("--ks", nargs="+", type=int, default=None)

    parser.add_argument("--nights", nargs="+", default=None)
    parser.add_argument("--cameras", nargs="+", default=None)

    parser.add_argument(
        "--orders",
        nargs="+",
        type=int,
        default=None,
        help="Original detector/order numbers to include. Default: all saved orders passing wavelength cut.",
    )

    parser.add_argument("--wavelength-min-nm", type=float, default=383.0)
    parser.add_argument("--wavelength-max-nm", type=float, default=1000.0)
    parser.add_argument("--min-overlap-fraction", type=float, default=0.5)

    parser.add_argument("--target-kp", type=float, default=None)
    parser.add_argument("--target-rv", type=float, default=None)
    parser.add_argument("--negative-target-kp", type=float, default=None)
    parser.add_argument("--negative-target-rv", type=float, default=None)

    parser.add_argument("--kp-window", type=float, default=10.0)
    parser.add_argument("--rv-window", type=float, default=5.0)

    parser.add_argument("--sigma-cut", type=float, default=3.0)
    parser.add_argument("--map-sign", type=float, default=-1.0)

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: config.path2reduced/results/sysrem_diagnostics_{model}",
    )

    args = parser.parse_args()

    project_path = Path(args.project_path)
    config = load_module(project_path / "config.py", "config")
    params = load_module(project_path / "parameters.py", "parameters")

    nights = args.nights if args.nights is not None else list(config.nights)
    cameras = args.cameras if args.cameras is not None else list(config.camera)
    ks = args.ks if args.ks is not None else list(range(1, args.max_k + 1))

    RV = np.asarray(config.RV, dtype=float)
    Kp = np.asarray(config.Kp, dtype=float)

    rv_min = safe_getattr(config, "RV_MIN", np.nanmin(RV))
    rv_max = safe_getattr(config, "RV_MAX", np.nanmax(RV))
    kp_min = safe_getattr(config, "KP_MIN", np.nanmin(Kp))
    kp_max = safe_getattr(config, "KP_MAX", np.nanmax(Kp))

    rv_mask = (RV >= rv_min) & (RV <= rv_max)
    kp_mask = (Kp >= kp_min) & (Kp <= kp_max)

    RV_crop = RV[rv_mask]
    Kp_crop = Kp[kp_mask]

    target_kp = args.target_kp
    if target_kp is None:
        target_kp = float(safe_getattr(params, "K_p", 198.0))

    target_rv = args.target_rv
    if target_rv is None:
        target_rv = float(safe_getattr(params, "Vsys", 0.0))

    negative_target_kp = args.negative_target_kp
    if negative_target_kp is None:
        # If the Kp grid contains negative values, use -target_kp.
        # Otherwise use +target_kp and rely also on the global peak.
        if np.nanmin(Kp_crop) < 0:
            negative_target_kp = -target_kp
        else:
            negative_target_kp = target_kp

    negative_target_rv = args.negative_target_rv
    if negative_target_rv is None:
        negative_target_rv = target_rv

    if args.output_dir is None:
        output_dir = Path(config.path2reduced) / "results" / f"sysrem_diagnostics_{args.model}"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    metadata_by_k = {}

    for k in ks:
        print(f"Processing k={k}")

        obs_map_full, obs_meta = load_combined_map(
            config=config,
            model=args.model,
            k=k,
            nights=nights,
            cameras=cameras,
            kind="obs",
            requested_orders=args.orders,
            wavelength_min_nm=args.wavelength_min_nm,
            wavelength_max_nm=args.wavelength_max_nm,
            min_overlap_fraction=args.min_overlap_fraction,
            map_sign=args.map_sign,
        )

        pos_map_full, pos_meta = load_combined_map(
            config=config,
            model=args.model,
            k=k,
            nights=nights,
            cameras=cameras,
            kind="positive",
            requested_orders=args.orders,
            wavelength_min_nm=args.wavelength_min_nm,
            wavelength_max_nm=args.wavelength_max_nm,
            min_overlap_fraction=args.min_overlap_fraction,
            map_sign=args.map_sign,
        )

        neg_map_full, neg_meta = load_combined_map(
            config=config,
            model=args.model,
            k=k,
            nights=nights,
            cameras=cameras,
            kind="negative",
            requested_orders=args.orders,
            wavelength_min_nm=args.wavelength_min_nm,
            wavelength_max_nm=args.wavelength_max_nm,
            min_overlap_fraction=args.min_overlap_fraction,
            map_sign=args.map_sign,
        )

        obs_map = obs_map_full[kp_mask][:, rv_mask]
        pos_map = pos_map_full[kp_mask][:, rv_mask]
        neg_map = neg_map_full[kp_mask][:, rv_mask]

        delta_map = pos_map - obs_map

        obs_metrics, _ = map_metrics(
            signal_map=obs_map,
            noise_reference_map=obs_map,
            Kp=Kp_crop,
            RV=RV_crop,
            target_kp=target_kp,
            target_rv=target_rv,
            kp_half_width=args.kp_window,
            rv_half_width=args.rv_window,
            sigma_cut=args.sigma_cut,
        )

        neg_metrics, _ = map_metrics(
            signal_map=neg_map,
            noise_reference_map=obs_map,
            Kp=Kp_crop,
            RV=RV_crop,
            target_kp=negative_target_kp,
            target_rv=negative_target_rv,
            kp_half_width=args.kp_window,
            rv_half_width=args.rv_window,
            sigma_cut=args.sigma_cut,
        )

        delta_metrics, _ = map_metrics(
            signal_map=delta_map,
            noise_reference_map=obs_map,
            Kp=Kp_crop,
            RV=RV_crop,
            target_kp=target_kp,
            target_rv=target_rv,
            kp_half_width=args.kp_window,
            rv_half_width=args.rv_window,
            sigma_cut=args.sigma_cut,
        )

        row = {
            "k": int(k),
            "model": args.model,
            "nights": " ".join(nights),
            "cameras": " ".join(cameras),
            "target_kp": target_kp,
            "target_rv": target_rv,
            "negative_target_kp": negative_target_kp,
            "negative_target_rv": negative_target_rv,
            "obs_window_snr": obs_metrics["window_snr"],
            "obs_window_kp": obs_metrics["window_kp"],
            "obs_window_rv": obs_metrics["window_rv"],
            "obs_global_snr": obs_metrics["global_snr"],
            "obs_global_kp": obs_metrics["global_kp"],
            "obs_global_rv": obs_metrics["global_rv"],
            "obs_noise": obs_metrics["noise"],
            "neg_window_snr": neg_metrics["window_snr"],
            "neg_window_kp": neg_metrics["window_kp"],
            "neg_window_rv": neg_metrics["window_rv"],
            "neg_global_snr": neg_metrics["global_snr"],
            "neg_global_kp": neg_metrics["global_kp"],
            "neg_global_rv": neg_metrics["global_rv"],
            "neg_noise_from_obs": neg_metrics["noise"],
            "delta_window_snr": delta_metrics["window_snr"],
            "delta_window_kp": delta_metrics["window_kp"],
            "delta_window_rv": delta_metrics["window_rv"],
            "delta_global_snr": delta_metrics["global_snr"],
            "delta_global_kp": delta_metrics["global_kp"],
            "delta_global_rv": delta_metrics["global_rv"],
            "delta_noise_from_obs": delta_metrics["noise"],
        }

        rows.append(row)
        metadata_by_k[str(k)] = {
            "obs": obs_meta,
            "positive": pos_meta,
            "negative": neg_meta,
        }

    csv_path = output_dir / "sysrem_diagnostics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    meta_path = output_dir / "sysrem_diagnostics_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "project_path": str(project_path),
                "model": args.model,
                "nights": nights,
                "cameras": cameras,
                "orders_requested": args.orders,
                "wavelength_min_nm": args.wavelength_min_nm,
                "wavelength_max_nm": args.wavelength_max_nm,
                "min_overlap_fraction": args.min_overlap_fraction,
                "target_kp": target_kp,
                "target_rv": target_rv,
                "negative_target_kp": negative_target_kp,
                "negative_target_rv": negative_target_rv,
                "kp_window": args.kp_window,
                "rv_window": args.rv_window,
                "sigma_cut": args.sigma_cut,
                "map_sign": args.map_sign,
                "metadata_by_k": metadata_by_k,
            },
            f,
            indent=2,
        )

    plot_path = output_dir / "sysrem_diagnostics.png"
    plot_sysrem_diagnostics(rows, plot_path)

    print()
    print("=" * 60)
    print("Saved diagnostics")
    print("=" * 60)
    print(csv_path)
    print(meta_path)
    print(plot_path)

    best_delta = max(rows, key=lambda r: r["delta_window_snr"])
    best_neg = max(rows, key=lambda r: r["neg_window_snr"])
    best_obs = max(rows, key=lambda r: r["obs_window_snr"])

    print()
    print("Best by observed expected-window SNR:")
    print(best_obs)
    print()
    print("Best by negative-injection expected-window SNR:")
    print(best_neg)
    print()
    print("Best by delta-CCF expected-window SNR:")
    print(best_delta)
    print("=" * 60)


if __name__ == "__main__":
    main()