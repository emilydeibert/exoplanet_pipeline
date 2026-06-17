from astropy.stats import sigma_clip
import matplotlib.pyplot as plt
from pathlib import Path
import importlib.util
import numpy as np
import argparse
import json
import csv


parser = argparse.ArgumentParser()

parser.add_argument("project_path", type=str)

parser.add_argument(
    "--model",
    required=True,
)

parser.add_argument(
    "--max-k",
    type=int,
    default=15,
)

parser.add_argument(
    "--ks",
    nargs="+",
    type=int,
    default=None,
)

parser.add_argument(
    "--orders",
    nargs="+",
    type=int,
    default=None,
    help="Original detector order numbers to include. Default: all saved orders.",
)

parser.add_argument(
    "--nights",
    nargs="+",
    default=None,
    help="Nights to include. Default: config.nights.",
)

parser.add_argument(
    "--cameras",
    nargs="+",
    default=None,
    help="Cameras to include. Default: config.camera.",
)

parser.add_argument(
    "--expected-kp",
    type=float,
    default=None,
)

parser.add_argument(
    "--expected-rv",
    type=float,
    default=None,
)

parser.add_argument(
    "--negative-expected-kp",
    type=float,
    default=None,
    help="Expected Kp for the negative injection. Use -Kp if your Kp grid includes negative values.",
)

parser.add_argument(
    "--negative-expected-rv",
    type=float,
    default=None,
)

parser.add_argument(
    "--kp-window",
    type=float,
    default=10.0,
)

parser.add_argument(
    "--rv-window",
    type=float,
    default=5.0,
)

parser.add_argument(
    "--sigma-cut",
    type=float,
    default=3.0,
)

parser.add_argument(
    "--map-sign",
    type=float,
    default=-1.0,
    help="Use -1 to match getResults.py emission convention.",
)

parser.add_argument(
    "--output-dir",
    default=None,
)

parser.add_argument("--rv-min", type=float, default=None)
parser.add_argument("--rv-max", type=float, default=None)
parser.add_argument("--kp-min", type=float, default=None)
parser.add_argument("--kp-max", type=float, default=None)

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


def map_original_orders_to_fmap_rows(fmap, saved_orders, requested_orders):
    """
    In the .npz file, fmap rows are positions in the saved order list.
    The saved 'orders' array gives the original detector order number.

    This function lets --orders mean original detector order numbers,
    not fmap row indices.
    """

    if requested_orders is None:
        return np.arange(fmap.shape[0])

    saved_orders = np.asarray(saved_orders, dtype=int)
    requested_orders = np.asarray(requested_orders, dtype=int)

    positions = []

    for order in requested_orders:
        matches = np.where(saved_orders == order)[0]

        if len(matches) == 0:
            print(f"Warning: requested order {order} not found in saved orders {saved_orders}")
        else:
            positions.append(matches[0])

    if len(positions) == 0:
        raise ValueError("No requested orders were found in this file.")

    return np.asarray(positions, dtype=int)


def load_one_map(
    filename,
    orders=None,
    map_sign=-1.0,
):
    data = np.load(filename)

    fmap = data["fmap"]

    # Use the Kp/RV grid saved in the file if available.
    # Fall back to config for older products.
    RV_file = data["RV"] if "RV" in data.files else config.RV
    Kp_file = data["Kp"] if "Kp" in data.files else config.Kp

    if "orders" in data.files:
        saved_orders = data["orders"]
    else:
        saved_orders = np.arange(fmap.shape[0])

    order_positions = map_original_orders_to_fmap_rows(
        fmap=fmap,
        saved_orders=saved_orders,
        requested_orders=orders,
    )

    order_sum = np.nansum(fmap[order_positions], axis=0)

    order_sum *= map_sign

    return order_sum, saved_orders[order_positions], RV_file, Kp_file


def load_and_combine_maps(
    model,
    iters,
    nights,
    cameras,
    kind="obs",
    orders=None,
    map_sign=-1.0,
):
    combined_maps = []
    used_orders_by_block = {}

    RV_ref = None
    Kp_ref = None

    for camera in cameras:

        for night in nights:

            if kind == "obs":
                filename = (
                    f"{config.path2reduced}/results/"
                    f"{night}_{camera}_{model}_k{iters}_iters.npz"
                )

            elif kind == "positive":
                filename = (
                    f"{config.path2reduced}/injected/"
                    f"{night}_{camera}_{model}_{iters}_iters_injected_positive.npz"
                )

            elif kind == "negative":
                filename = (
                    f"{config.path2reduced}/injected/"
                    f"{night}_{camera}_{model}_{iters}_iters_injected_negative.npz"
                )

            else:
                raise ValueError(f"Unknown kind: {kind}")

            filename = Path(filename)

            if not filename.exists():
                raise FileNotFoundError(filename)

            order_sum, used_orders, RV_file, Kp_file = load_one_map(
                filename,
                orders=orders,
                map_sign=map_sign,
            )

            RV_file = np.asarray(RV_file, dtype=float)
            Kp_file = np.asarray(Kp_file, dtype=float)

            if RV_ref is None:
                RV_ref = RV_file
                Kp_ref = Kp_file
            else:
                if not np.allclose(RV_ref, RV_file):
                    raise ValueError(f"RV grid mismatch for {filename}")
                if not np.allclose(Kp_ref, Kp_file):
                    raise ValueError(f"Kp grid mismatch for {filename}")

            combined_maps.append(order_sum)

            used_orders_by_block[f"{night}_{camera}_{kind}"] = [
                int(o) for o in used_orders
            ]

    final_map = np.nansum(combined_maps, axis=0)

    return final_map, used_orders_by_block, RV_ref, Kp_ref


def crop_map_to_grid(
    signal_map,
    RV,
    Kp,
    rv_min,
    rv_max,
    kp_min=None,
    kp_max=None,
):
    RV = np.asarray(RV, dtype=float)
    Kp = np.asarray(Kp, dtype=float)

    rv_mask = (RV >= rv_min) & (RV <= rv_max)

    if kp_min is None:
        kp_min = np.nanmin(Kp)
    if kp_max is None:
        kp_max = np.nanmax(Kp)

    kp_mask = (Kp >= kp_min) & (Kp <= kp_max)

    return signal_map[kp_mask][:, rv_mask], RV[rv_mask], Kp[kp_mask]

def calculate_snr_map(
    kpvsys_map,
    sigma_cut=3.0,
):
    kpvsys_map = kpvsys_map - np.nanmedian(kpvsys_map)

    clipped = sigma_clip(
        kpvsys_map,
        sigma_upper=sigma_cut,
        sigma_lower=100,
    )

    noise = np.nanstd(clipped)

    snr_map = kpvsys_map / noise

    return snr_map, noise


def calculate_noise_from_observed(
    observed_map,
    sigma_cut=3.0,
):
    """
    This follows the spirit of getResults.py:
    subtract median, sigma-clip, then use std as the map noise.

    This noise is then used for:
      observed SNR = observed / noise_obs
      negative injection SNR = negative_injection / noise_neg
      delta CCF SNR = (positive_injection - observed) / noise_obs
    """

    observed_map = observed_map - np.nanmedian(observed_map)

    clipped = sigma_clip(
        observed_map,
        sigma_upper=sigma_cut,
        sigma_lower=100,
    )

    noise = np.nanstd(clipped)

    return noise


def snr_map_from_noise(
    signal_map,
    noise,
):
    signal_map = signal_map - np.nanmedian(signal_map)

    return signal_map / noise


def find_peak(
    snr_map,
    RV,
    Kp,
):
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

    return find_peak(
        masked,
        RV,
        Kp,
    )


def plot_sysrem_summary(
    rows,
    savefile,
):
    k = np.array([r["k"] for r in rows])

    obs = np.array([r["obs_expected_snr"] for r in rows])
    obs_global = np.array([r["obs_global_snr"] for r in rows])

    neg = np.array([r["neg_expected_snr"] for r in rows])
    neg_global = np.array([r["neg_global_snr"] for r in rows])

    delta = np.array([r["delta_expected_snr"] for r in rows])
    delta_global = np.array([r["delta_global_snr"] for r in rows])

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8, 10),
        sharex=True,
    )

    axes[0].plot(k, obs, marker="o", label="Expected-window peak")
    axes[0].plot(k, obs_global, marker="o", linestyle="--", label="Global peak")
    axes[0].set_ylabel("SNR")
    axes[0].set_title("Observed CCF")
    axes[0].legend()

    axes[1].plot(k, neg, marker="o", label="Expected-window peak")
    axes[1].plot(k, neg_global, marker="o", linestyle="--", label="Global peak")
    axes[1].set_ylabel("SNR")
    axes[1].set_title("Negative injected CCF / negative-map noise")
    axes[1].legend()

    axes[2].plot(k, delta, marker="o", label="Expected-window peak")
    axes[2].plot(k, delta_global, marker="o", linestyle="--", label="Global peak")
    axes[2].set_ylabel("SNR")
    axes[2].set_xlabel("SYSREM iteration k")
    axes[2].set_title("Delta CCF = positive injection − observed, noise from observed")
    axes[2].legend()

    for ax in axes:
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(savefile, dpi=300, bbox_inches="tight")
    plt.close()


def main():

    nights = args.nights if args.nights is not None else config.nights
    cameras = args.cameras if args.cameras is not None else config.camera

    #RV = data["RV"] if "RV" in data.files else config.RV
    #Kp = data["Kp"] if "Kp" in data.files else config.Kp
    RV = np.asarray(config.RV, dtype=float)
    Kp = np.asarray(config.Kp, dtype=float)

    RV_MIN = args.rv_min if args.rv_min is not None else getattr(config, "RV_MIN", -75)
    RV_MAX = args.rv_max if args.rv_max is not None else getattr(config, "RV_MAX", 75)
    KP_MIN = args.kp_min if args.kp_min is not None else getattr(config, "KP_MIN", 50)
    KP_MAX = args.kp_max if args.kp_max is not None else getattr(config, "KP_MAX", 275)

    if RV_MIN is None:
        RV_MIN = -75
    if RV_MAX is None:
        RV_MAX = 75
    if KP_MIN is None:
        KP_MIN = 50
    if KP_MAX is None:
        KP_MAX = 275

    rv_mask = (RV >= RV_MIN) & (RV <= RV_MAX)
    kp_mask = (Kp >= KP_MIN) & (Kp <= KP_MAX)

    RV_crop = RV[rv_mask]
    Kp_crop = Kp[kp_mask]

    expected_kp = args.expected_kp
    if expected_kp is None:
        expected_kp = float(params.K_p)

    expected_rv = args.expected_rv
    if expected_rv is None:
        expected_rv = float(params.Vsys)

    negative_expected_kp = args.negative_expected_kp
    if negative_expected_kp is None:
        negative_expected_kp = -1.0 * expected_kp

    negative_expected_rv = args.negative_expected_rv
    if negative_expected_rv is None:
        negative_expected_rv = expected_rv

    if args.ks is None:
        k_values = np.arange(1, args.max_k + 1)
    else:
        k_values = args.ks

    if args.output_dir is None:
        output_dir = Path(config.path2reduced) / "results" / f"sysrem_diagnostics_{args.model}"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    metadata = {}

    for k in k_values:

        print()
        print("=" * 40)
        print(f"SYSREM k = {k}")
        print("=" * 40)

        obs_map, obs_orders, RV_obs, Kp_obs = load_and_combine_maps(
            model=args.model,
            iters=k,
            nights=nights,
            cameras=cameras,
            kind="obs",
            orders=args.orders,
            map_sign=args.map_sign,
            )

        pos_map, pos_orders, RV_pos, Kp_pos = load_and_combine_maps(
            model=args.model,
            iters=k,
            nights=nights,
            cameras=cameras,
            kind="positive",
            orders=args.orders,
            map_sign=args.map_sign,
            )

        neg_map, neg_orders, RV_neg, Kp_neg = load_and_combine_maps(
            model=args.model,
            iters=k,
            nights=nights,
            cameras=cameras,
            kind="negative",
            orders=args.orders,
            map_sign=args.map_sign,
            )

        if not np.allclose(RV_obs, RV_pos):
            raise ValueError("Observed and positive-injection RV grids do not match.")
        if not np.allclose(Kp_obs, Kp_pos):
            raise ValueError("Observed and positive-injection Kp grids do not match.")

        obs_crop, RV_obs_crop, Kp_obs_crop = crop_map_to_grid(
            obs_map,
            RV_obs,
            Kp_obs,
            rv_min=RV_MIN,
            rv_max=RV_MAX,
            kp_min=KP_MIN,
            kp_max=KP_MAX,
            )

        pos_crop, RV_pos_crop, Kp_pos_crop = crop_map_to_grid(
            pos_map,
            RV_pos,
            Kp_pos,
            rv_min=RV_MIN,
            rv_max=RV_MAX,
            kp_min=KP_MIN,
            kp_max=KP_MAX,
            )

        # Negative-injection Kp crop mirrors the positive Kp crop.
        # For Kp_MIN=50, Kp_MAX=275, this becomes -275 to -50.
        if np.nanmax(Kp_neg) <= 0 and np.nanmin(Kp_neg) < 0:
            NEG_KP_MIN = -1.0 * KP_MAX
            NEG_KP_MAX = -1.0 * KP_MIN
        else:
            NEG_KP_MIN = KP_MIN
            NEG_KP_MAX = KP_MAX

        neg_crop, RV_neg_crop, Kp_neg_crop = crop_map_to_grid(
            signal_map=neg_map,
            RV=RV_neg,
            Kp=Kp_neg,
            rv_min=RV_MIN,
            rv_max=RV_MAX,
            kp_min=NEG_KP_MIN,
            kp_max=NEG_KP_MAX,
        )

        print(
            f"Negative crop: Kp {np.nanmin(Kp_neg_crop):.1f} to {np.nanmax(Kp_neg_crop):.1f}, "
            f"RV {np.nanmin(RV_neg_crop):.1f} to {np.nanmax(RV_neg_crop):.1f}"
        )

        # This is the Cheverall-style / paper-style delta CCF definition:
        #   Delta CCF = CCF_injected - CCF_observed
        #   SNR(Delta CCF) = Delta CCF / noise(CCF_observed)
        delta_crop = pos_crop - obs_crop

        noise_obs = calculate_noise_from_observed(
            obs_crop,
            sigma_cut=args.sigma_cut,
        )

        obs_snr_map = snr_map_from_noise(
            obs_crop,
            noise_obs,
        )

        neg_snr_map_raw, noise_neg = calculate_snr_map(
            neg_crop,
            sigma_cut=args.sigma_cut,
        )

        # The injected signal appears as a negative trough in the current map convention.
        # Flip it so recovered negative injections are positive peaks.
        neg_recovery_snr_map = -1.0 * neg_snr_map_raw

        delta_snr_map = snr_map_from_noise(
            delta_crop,
            noise_obs,
        )

        delta_recovery_snr_map = -1.0 * delta_snr_map

        obs_global = find_peak(
            obs_snr_map,
            RV_obs_crop,
            Kp_obs_crop,
        )

        obs_expected = find_peak_near_expected(
            obs_snr_map,
            RV_obs_crop,
            Kp_obs_crop,
            expected_kp=expected_kp,
            expected_rv=expected_rv,
            kp_window=args.kp_window,
            rv_window=args.rv_window,
        )

        neg_global = find_peak(
            neg_recovery_snr_map,
            RV_neg_crop,
            Kp_neg_crop,
        )

        neg_expected = find_peak_near_expected(
            neg_recovery_snr_map,
            RV_neg_crop,
            Kp_neg_crop,
            expected_kp=negative_expected_kp,
            expected_rv=negative_expected_rv,
            kp_window=args.kp_window,
            rv_window=args.rv_window,
        )

        delta_global = find_peak(
            delta_recovery_snr_map,
            RV_obs_crop,
            Kp_obs_crop,
        )

        delta_expected = find_peak_near_expected(
            delta_recovery_snr_map,
            RV_obs_crop,
            Kp_obs_crop,
            expected_kp=expected_kp,
            expected_rv=expected_rv,
            kp_window=args.kp_window,
            rv_window=args.rv_window,
        )

        row = {
            "k": int(k),
            "model": args.model,
            "nights": " ".join([str(n) for n in nights]),
            "cameras": " ".join([str(c) for c in cameras]),
            "noise_obs": float(noise_obs),
            "noise_neg": noise_neg,

            "obs_expected_snr": obs_expected["snr"],
            "obs_expected_kp": obs_expected["kp"],
            "obs_expected_rv": obs_expected["rv"],
            "obs_global_snr": obs_global["snr"],
            "obs_global_kp": obs_global["kp"],
            "obs_global_rv": obs_global["rv"],

            "neg_expected_snr": neg_expected["snr"],
            "neg_expected_kp": neg_expected["kp"],
            "neg_expected_rv": neg_expected["rv"],
            "neg_global_snr": neg_global["snr"],
            "neg_global_kp": neg_global["kp"],
            "neg_global_rv": neg_global["rv"],

            "delta_expected_snr": delta_expected["snr"],
            "delta_expected_kp": delta_expected["kp"],
            "delta_expected_rv": delta_expected["rv"],
            "delta_global_snr": delta_global["snr"],
            "delta_global_kp": delta_global["kp"],
            "delta_global_rv": delta_global["rv"],
        }

        rows.append(row)

        metadata[str(k)] = {
            "obs_orders": obs_orders,
            "positive_orders": pos_orders,
            "negative_orders": neg_orders,
        }

        print(f"Observed expected-window SNR : {obs_expected['snr']:.2f}")
        print(f"Observed global SNR          : {obs_global['snr']:.2f}")
        print(f"Negative expected-window SNR : {neg_expected['snr']:.2f}")
        print(f"Negative global SNR          : {neg_global['snr']:.2f}")
        print(f"Delta expected-window SNR    : {delta_expected['snr']:.2f}")
        print(f"Delta global SNR             : {delta_global['snr']:.2f}")

    csv_name = output_dir / "sysrem_iteration_diagnostics.csv"

    with open(csv_name, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    json_name = output_dir / "sysrem_iteration_diagnostics_metadata.json"

    with open(json_name, "w") as f:
        json.dump(
            {
                "project_path": str(project_path),
                "model": args.model,
                "nights": list(nights),
                "cameras": list(cameras),
                "orders_requested": args.orders,
                "expected_kp": expected_kp,
                "expected_rv": expected_rv,
                "negative_expected_kp": negative_expected_kp,
                "negative_expected_rv": negative_expected_rv,
                "kp_window": args.kp_window,
                "rv_window": args.rv_window,
                "sigma_cut": args.sigma_cut,
                "map_sign": args.map_sign,
                "metadata": metadata,
                "rv_min": RV_MIN,
                "rv_max": RV_MAX,
                "kp_min": KP_MIN,
                "kp_max": KP_MAX,
            },
            f,
            indent=4,
        )

    plot_name = output_dir / "sysrem_iteration_diagnostics.png"

    plot_sysrem_summary(
        rows,
        savefile=plot_name,
    )

    best_obs = max(rows, key=lambda r: r["obs_expected_snr"])
    best_neg = max(rows, key=lambda r: r["neg_expected_snr"])
    best_delta = max(rows, key=lambda r: r["delta_expected_snr"])

    print()
    print("=" * 60)
    print("Saved:")
    print(csv_name)
    print(json_name)
    print(plot_name)
    print("=" * 60)

    print()
    print("Best observed expected-window SNR:")
    print(best_obs)

    print()
    print("Best negative-injection expected-window SNR:")
    print(best_neg)

    print()
    print("Best delta-CCF expected-window SNR:")
    print(best_delta)


if __name__ == "__main__":
    main()