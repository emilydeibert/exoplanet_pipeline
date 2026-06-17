from astropy.stats import sigma_clip
import matplotlib.pyplot as plt
from pathlib import Path
import importlib.util
import numpy as np
import argparse
import json
import math


parser = argparse.ArgumentParser()

parser.add_argument("project_path", type=str)

parser.add_argument(
    "--model",
    required=True,
)

parser.add_argument(
    "--type",
    choices=["observed", "negative", "delta"],
    required=True,
    help="Map type to plot: observed, negative injected, or delta CCF = positive injected - observed.",
)

parser.add_argument(
    "--max-k",
    type=int,
    default=15,
    help="Maximum SYSREM iteration to plot if --ks is not given.",
)

parser.add_argument(
    "--ks",
    nargs="+",
    type=int,
    default=None,
    help="Specific SYSREM iterations to plot. Default: 1..max-k.",
)

parser.add_argument(
    "--orders",
    nargs="+",
    type=int,
    default=None,
)

parser.add_argument(
    "--nights",
    nargs="+",
    default=None,
    help="Nights to include (default: config.nights)",
)

parser.add_argument(
    "--cameras",
    nargs="+",
    default=None,
    help="Cameras to include (default: config.camera)",
)

parser.add_argument(
    "--sigma-cut",
    type=float,
    default=3.0,
)

parser.add_argument(
    "--rv-min",
    type=float,
    default=None,
    help="Minimum RV/Vsys to plot. Default: config.RV_MIN or -75.",
)

parser.add_argument(
    "--rv-max",
    type=float,
    default=None,
    help="Maximum RV/Vsys to plot. Default: config.RV_MAX or +75.",
)

parser.add_argument(
    "--kp-min",
    type=float,
    default=None,
    help="Minimum positive Kp to plot. Default: config.KP_MIN or 1.",
)

parser.add_argument(
    "--kp-max",
    type=float,
    default=None,
    help="Maximum positive Kp to plot. Default: config.KP_MAX or 300.",
)

parser.add_argument(
    "--expected-kp",
    type=float,
    default=None,
    help="Optional expected positive Kp to mark on observed/delta plots. Negative of this is marked for negative plots.",
)

parser.add_argument(
    "--expected-rv",
    type=float,
    default=0.0,
    help="Optional expected RV/Vsys to mark.",
)

parser.add_argument(
    "--map-sign",
    type=float,
    default=-1.0,
    help="Multiplicative sign applied to loaded fmap/order sums. Default matches getResults.py.",
)

parser.add_argument(
    "--vmax",
    type=float,
    default=None,
    help="Optional fixed absolute color scale maximum.",
)

parser.add_argument(
    "--output-dir",
    default=None,
    help="Directory for saved plot/summary. Default: config.path2reduced/results/sysrem_iteration_maps",
)

parser.add_argument(
    "--dpi",
    type=int,
    default=250,
)

parser.add_argument(
    "--kp-window",
    type=float,
    default=15.0,
    help="Kp half-width used to decide whether a peak is near the expected location.",
)

parser.add_argument(
    "--rv-window",
    type=float,
    default=10.0,
    help="RV half-width used to decide whether a peak is near the expected location.",
)

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


def get_crop_limits():
    rv_min = args.rv_min
    rv_max = args.rv_max
    kp_min = args.kp_min
    kp_max = args.kp_max

    if rv_min is None:
        rv_min = getattr(config, "RV_MIN", -75)
    if rv_max is None:
        rv_max = getattr(config, "RV_MAX", 75)
    if kp_min is None:
        kp_min = getattr(config, "KP_MIN", 50)
    if kp_max is None:
        kp_max = getattr(config, "KP_MAX", 275)

    if rv_min is None:
        rv_min = -75
    if rv_max is None:
        rv_max = 75
    if kp_min is None:
        kp_min = 50
    if kp_max is None:
        kp_max = 275

    return float(rv_min), float(rv_max), float(kp_min), float(kp_max)


def get_filename(
    model,
    iters,
    night,
    camera,
    kind,
):
    if kind == "observed":
        return Path(
            f"{config.path2reduced}/results/"
            f"{night}_{camera}_{model}_k{iters}_iters.npz"
        )

    if kind == "positive":
        return Path(
            f"{config.path2reduced}/injected/"
            f"{night}_{camera}_{model}_{iters}_iters_injected_positive.npz"
        )

    if kind == "negative":
        return Path(
            f"{config.path2reduced}/injected/"
            f"{night}_{camera}_{model}_{iters}_iters_injected_negative.npz"
        )

    raise ValueError(f"Unknown kind: {kind}")


def select_order_indices(
    fmap,
    data,
    requested_orders=None,
):
    """
    Select orders robustly.

    If the file contains an 'orders' array, requested_orders are interpreted
    as original order numbers and mapped onto the fmap axis.

    If no 'orders' array exists, requested_orders are interpreted as direct
    fmap indices, matching old getResults.py behavior.
    """
    n_file_orders = fmap.shape[0]

    if requested_orders is None:
        return np.arange(n_file_orders), (
            data["orders"].astype(int)
            if "orders" in data.files
            else np.arange(n_file_orders)
        )

    requested_orders = np.asarray(requested_orders, dtype=int)

    if "orders" in data.files:
        file_orders = np.asarray(data["orders"], dtype=int)

        keep = []
        used_orders = []

        for od in requested_orders:
            matches = np.where(file_orders == od)[0]
            if len(matches) == 0:
                continue
            keep.append(int(matches[0]))
            used_orders.append(int(od))

        if len(keep) == 0:
            raise ValueError(
                "None of the requested orders were found in file orders. "
                f"Requested={requested_orders.tolist()}, file_orders={file_orders.tolist()}"
            )

        return np.asarray(keep, dtype=int), np.asarray(used_orders, dtype=int)

    # Fallback: old behavior.
    keep = requested_orders
    keep = keep[(keep >= 0) & (keep < n_file_orders)]

    if len(keep) == 0:
        raise ValueError(
            "Requested order indices are outside fmap axis. "
            f"Requested={requested_orders.tolist()}, n_file_orders={n_file_orders}"
        )

    return keep.astype(int), keep.astype(int)


def load_one_map(
    filename,
    orders=None,
    map_sign=-1.0,
):
    if not filename.exists():
        raise FileNotFoundError(filename)

    with np.load(filename, allow_pickle=True) as data:
        fmap = data["fmap"]

        order_indices, used_orders = select_order_indices(
            fmap=fmap,
            data=data,
            requested_orders=orders,
        )

        order_sum = np.nansum(fmap[order_indices], axis=0)
        order_sum *= map_sign

        RV_file = (
            np.asarray(data["RV"], dtype=float)
            if "RV" in data.files
            else np.asarray(config.RV, dtype=float)
        )

        Kp_file = (
            np.asarray(data["Kp"], dtype=float)
            if "Kp" in data.files
            else np.asarray(config.Kp, dtype=float)
        )

    return order_sum, used_orders, RV_file, Kp_file


def load_and_combine_maps(
    model,
    iters,
    nights,
    cameras,
    kind,
    orders=None,
    map_sign=-1.0,
):
    combined_maps = []

    RV_ref = None
    Kp_ref = None
    used_orders_by_block = {}

    for camera in cameras:

        for night in nights:

            filename = get_filename(
                model=model,
                iters=iters,
                night=night,
                camera=camera,
                kind=kind,
            )

            order_sum, used_orders, RV_file, Kp_file = load_one_map(
                filename=filename,
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
    kp_min,
    kp_max,
):
    RV = np.asarray(RV, dtype=float)
    Kp = np.asarray(Kp, dtype=float)

    rv_mask = (RV >= rv_min) & (RV <= rv_max)
    kp_mask = (Kp >= kp_min) & (Kp <= kp_max)

    if not np.any(rv_mask):
        raise ValueError(f"No RV values inside crop range {rv_min} to {rv_max}")

    if not np.any(kp_mask):
        raise ValueError(f"No Kp values inside crop range {kp_min} to {kp_max}")

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
    obs_map,
    sigma_cut=3.0,
):
    obs_map = obs_map - np.nanmedian(obs_map)

    clipped = sigma_clip(
        obs_map,
        sigma_upper=sigma_cut,
        sigma_lower=100,
    )

    noise = np.nanstd(clipped)

    return noise


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
    kp_mask = (
        (Kp >= expected_kp - kp_window)
        & (Kp <= expected_kp + kp_window)
    )

    rv_mask = (
        (RV >= expected_rv - rv_window)
        & (RV <= expected_rv + rv_window)
    )

    if not np.any(kp_mask) or not np.any(rv_mask):
        return None

    masked = np.full_like(
        snr_map,
        np.nan,
        dtype=float,
    )

    masked[np.ix_(kp_mask, rv_mask)] = snr_map[np.ix_(kp_mask, rv_mask)]

    if not np.any(np.isfinite(masked)):
        return None

    return find_peak(
        masked,
        RV,
        Kp,
    )


def get_expected_location_for_type(
    map_type,
    Kp,
):
    if args.expected_kp is None:
        return None, None

    expected_kp = float(args.expected_kp)
    expected_rv = float(args.expected_rv)

    if map_type == "negative":
        # Negative-injection products may have a negative Kp grid.
        # If so, mirror the expected positive Kp location.
        if np.nanmax(Kp) <= 0 and np.nanmin(Kp) < 0:
            expected_kp = -1.0 * abs(expected_kp)

    return expected_kp, expected_rv


def peak_is_near_expected(
    peak,
    expected_kp,
    expected_rv,
    kp_window,
    rv_window,
):
    if expected_kp is None or expected_rv is None:
        return True

    return (
        abs(peak["kp"] - expected_kp) <= kp_window
        and abs(peak["rv"] - expected_rv) <= rv_window
    )


def peak_is_near_crop_edge(
    peak,
    RV,
    Kp,
):
    rv_step = np.nanmedian(np.abs(np.diff(RV))) if len(RV) > 1 else 0.0
    kp_step = np.nanmedian(np.abs(np.diff(Kp))) if len(Kp) > 1 else 0.0

    near_rv_edge = (
        abs(peak["rv"] - np.nanmin(RV)) <= rv_step
        or abs(peak["rv"] - np.nanmax(RV)) <= rv_step
    )

    near_kp_edge = (
        abs(peak["kp"] - np.nanmin(Kp)) <= kp_step
        or abs(peak["kp"] - np.nanmax(Kp)) <= kp_step
    )

    return bool(near_rv_edge or near_kp_edge)


def build_snr_map_for_type(
    model,
    iters,
    nights,
    cameras,
    map_type,
    orders,
    rv_min,
    rv_max,
    kp_min,
    kp_max,
):
    obs_map, obs_orders, RV_obs, Kp_obs = load_and_combine_maps(
        model=model,
        iters=iters,
        nights=nights,
        cameras=cameras,
        kind="observed",
        orders=orders,
        map_sign=args.map_sign,
    )

    obs_crop, RV_obs_crop, Kp_obs_crop = crop_map_to_grid(
        signal_map=obs_map,
        RV=RV_obs,
        Kp=Kp_obs,
        rv_min=rv_min,
        rv_max=rv_max,
        kp_min=kp_min,
        kp_max=kp_max,
    )

    obs_noise = calculate_noise_from_observed(
        obs_crop,
        sigma_cut=args.sigma_cut,
    )

    if map_type == "observed":
        snr_map, noise = calculate_snr_map(
            obs_crop,
            sigma_cut=args.sigma_cut,
        )

        peak = find_peak(
            snr_map,
            RV_obs_crop,
            Kp_obs_crop,
        )

        return {
            "snr_map": snr_map,
            "RV": RV_obs_crop,
            "Kp": Kp_obs_crop,
            "peak": peak,
            "noise": noise,
            "orders": obs_orders,
        }

    if map_type == "negative":
        neg_map, neg_orders, RV_neg, Kp_neg = load_and_combine_maps(
            model=model,
            iters=iters,
            nights=nights,
            cameras=cameras,
            kind="negative",
            orders=orders,
            map_sign=args.map_sign,
        )

        if np.nanmax(Kp_neg) <= 0 and np.nanmin(Kp_neg) < 0:
            neg_kp_min = -1.0 * kp_max
            neg_kp_max = -1.0 * kp_min
        else:
            neg_kp_min = kp_min
            neg_kp_max = kp_max

        neg_crop, RV_neg_crop, Kp_neg_crop = crop_map_to_grid(
            signal_map=neg_map,
            RV=RV_neg,
            Kp=Kp_neg,
            rv_min=rv_min,
            rv_max=rv_max,
            kp_min=neg_kp_min,
            kp_max=neg_kp_max,
        )

        # Use the negative-injection map's own noise estimate.
        # Then flip the sign so the recovered negative injection appears positive.
        neg_snr_map, neg_noise = calculate_snr_map(
            neg_crop,
            sigma_cut=args.sigma_cut,
        )

        # Show recovered negative injection as positive.
        neg_snr_map *= -1.0

        peak = find_peak(
            neg_snr_map,
            RV_neg_crop,
            Kp_neg_crop,
        )

        return {
            "snr_map": neg_snr_map,
            "RV": RV_neg_crop,
            "Kp": Kp_neg_crop,
            "peak": peak,
            "noise": neg_noise,
            "orders": neg_orders,
        }

    if map_type == "delta":
        pos_map, pos_orders, RV_pos, Kp_pos = load_and_combine_maps(
            model=model,
            iters=iters,
            nights=nights,
            cameras=cameras,
            kind="positive",
            orders=orders,
            map_sign=args.map_sign,
        )

        if not np.allclose(RV_obs, RV_pos):
            raise ValueError("Observed and positive-injection RV grids do not match.")
        if not np.allclose(Kp_obs, Kp_pos):
            raise ValueError("Observed and positive-injection Kp grids do not match.")

        pos_crop, RV_pos_crop, Kp_pos_crop = crop_map_to_grid(
            signal_map=pos_map,
            RV=RV_pos,
            Kp=Kp_pos,
            rv_min=rv_min,
            rv_max=rv_max,
            kp_min=kp_min,
            kp_max=kp_max,
        )

        # In raw CCF terms, delta is often described as injected - observed.
        # Here the loaded maps have already been multiplied by map_sign.
        # With the current processed-map sign convention, obs_crop - pos_crop
        # makes the recovered injected signal positive in the plotted SNR map.
        delta_map = obs_crop - pos_crop
        delta_map = delta_map - np.nanmedian(delta_map)
        delta_snr_map = delta_map / obs_noise

        peak = find_peak(
            delta_snr_map,
            RV_pos_crop,
            Kp_pos_crop,
        )

        return {
            "snr_map": delta_snr_map,
            "RV": RV_pos_crop,
            "Kp": Kp_pos_crop,
            "peak": peak,
            "noise": obs_noise,
            "orders": pos_orders,
        }

    raise ValueError(f"Unknown map_type: {map_type}")


def plot_iteration_grid(
    results,
    ks,
    model,
    map_type,
    nights,
    cameras,
    orders,
    rv_min,
    rv_max,
    kp_min,
    kp_max,
    savefile,
    vmax=None,
):
    n_panels = len(ks)
    ncols = 5
    nrows = int(math.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.0 * ncols, 3.4 * nrows),
        squeeze=False,
    )

    if vmax is None:
        vmax = np.nanmax(
            [
                np.nanmax(np.abs(result["snr_map"]))
                for result in results
            ]
        )

    # Avoid pathological all-zero/all-NaN weirdness.
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    last_im = None

    for idx, (k, result) in enumerate(zip(ks, results)):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        RV = result["RV"]
        Kp = result["Kp"]
        snr_map = result["snr_map"]
        peak = result["peak"]

        last_im = ax.pcolormesh(
            RV,
            Kp,
            snr_map,
            shading="auto",
            vmin=-vmax,
            vmax=vmax,
        )

        ax.scatter(
            peak["rv"],
            peak["kp"],
            marker="x",
            s=45,
            linewidths=1.5,
            color="black",
        )

        expected_kp, expected_rv = get_expected_location_for_type(
            map_type=map_type,
            Kp=Kp,
        )

        if expected_kp is not None:
            ax.scatter(
                expected_rv,
                expected_kp,
                marker="+",
                s=55,
                linewidths=1.5,
                color="black",
            )

        ax.set_title(
            f"k={k} | peak={peak['snr']:.2f}\n"
            f"Kp={peak['kp']:.0f}, RV={peak['rv']:.0f}",
            fontsize=10,
        )

        ax.set_xlim(rv_min, rv_max)
        ax.set_ylim(np.nanmin(Kp), np.nanmax(Kp))

        if row == nrows - 1:
            ax.set_xlabel("Vsys [km/s]")
        if col == 0:
            ax.set_ylabel("Kp [km/s]")

    # Hide unused axes.
    for idx in range(n_panels, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis("off")

    night_str = "_".join([str(n) for n in nights])
    cam_str = "_".join([str(c) for c in cameras])

    orders_str = (
        "all orders"
        if orders is None
        else f"orders {min(orders)}-{max(orders)}"
    )

    fig.suptitle(
        f"{model} {map_type} SYSREM maps | nights={night_str} | cameras={cam_str} | {orders_str}\n"
        f"RV=[{rv_min}, {rv_max}], Kp=[{kp_min}, {kp_max}] "
        f"(negative plots mirror Kp limits)",
        fontsize=14,
        y=0.995,
    )

    cbar = fig.colorbar(
        last_im,
        ax=axes,
        fraction=0.025,
        pad=0.02,
    )
    cbar.set_label("SNR", rotation=270, labelpad=15)

    fig.savefig(
        savefile,
        dpi=args.dpi,
        bbox_inches="tight",
    )

    plt.close(fig)



def plot_iteration_peaks(
    results,
    ks,
    model,
    map_type,
    nights,
    cameras,
    orders,
    rv_min,
    rv_max,
    kp_min,
    kp_max,
    savefile,
):
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(7.2, 4.4),
    )

    global_snrs = []
    expected_snrs = []

    good_global_ks = []
    good_global_snrs = []

    off_expected_ks = []
    off_expected_snrs = []

    edge_ks = []
    edge_snrs = []

    peak_summary_rows = []

    for k, result in zip(ks, results):
        RV = result["RV"]
        Kp = result["Kp"]
        snr_map = result["snr_map"]
        global_peak = result["peak"]

        expected_kp, expected_rv = get_expected_location_for_type(
            map_type=map_type,
            Kp=Kp,
        )

        expected_peak = None
        if expected_kp is not None:
            expected_peak = find_peak_near_expected(
                snr_map=snr_map,
                RV=RV,
                Kp=Kp,
                expected_kp=expected_kp,
                expected_rv=expected_rv,
                kp_window=args.kp_window,
                rv_window=args.rv_window,
            )

        global_near_expected = peak_is_near_expected(
            peak=global_peak,
            expected_kp=expected_kp,
            expected_rv=expected_rv,
            kp_window=args.kp_window,
            rv_window=args.rv_window,
        )

        global_near_edge = peak_is_near_crop_edge(
            peak=global_peak,
            RV=RV,
            Kp=Kp,
        )

        global_snrs.append(global_peak["snr"])

        if expected_peak is None:
            expected_snrs.append(np.nan)
        else:
            expected_snrs.append(expected_peak["snr"])

        if global_near_edge:
            edge_ks.append(k)
            edge_snrs.append(global_peak["snr"])
        elif global_near_expected:
            good_global_ks.append(k)
            good_global_snrs.append(global_peak["snr"])
        else:
            off_expected_ks.append(k)
            off_expected_snrs.append(global_peak["snr"])

        peak_summary_rows.append(
            {
                "k": int(k),
                "global_snr": float(global_peak["snr"]),
                "global_kp": float(global_peak["kp"]),
                "global_rv": float(global_peak["rv"]),
                "expected_snr": (
                    None
                    if expected_peak is None
                    else float(expected_peak["snr"])
                ),
                "expected_peak_kp": (
                    None
                    if expected_peak is None
                    else float(expected_peak["kp"])
                ),
                "expected_peak_rv": (
                    None
                    if expected_peak is None
                    else float(expected_peak["rv"])
                ),
                "target_expected_kp": (
                    None
                    if expected_kp is None
                    else float(expected_kp)
                ),
                "target_expected_rv": (
                    None
                    if expected_rv is None
                    else float(expected_rv)
                ),
                "global_near_expected": bool(global_near_expected),
                "global_near_crop_edge": bool(global_near_edge),
            }
        )

    ax.plot(
        ks,
        global_snrs,
        linestyle="-",
        linewidth=1.2,
        alpha=0.5,
        label="global peak",
    )

    if np.any(np.isfinite(expected_snrs)):
        ax.plot(
            ks,
            expected_snrs,
            marker="o",
            linewidth=1.8,
            label=(
                "expected-window peak "
                f"(±{args.kp_window:g} Kp, ±{args.rv_window:g} RV)"
            ),
        )

    if args.expected_kp is None:
        good_label = "global peak"
    else:
        good_label = "global peak near expected"

    ax.scatter(
        good_global_ks,
        good_global_snrs,
        marker="o",
        s=45,
        color="black",
        label=good_label,
    )

    if len(off_expected_ks) > 0:
        ax.scatter(
            off_expected_ks,
            off_expected_snrs,
            marker="o",
            s=75,
            facecolors="none",
            edgecolors="black",
            linewidths=1.6,
            label="global peak off expected",
        )

    if len(edge_ks) > 0:
        ax.scatter(
            edge_ks,
            edge_snrs,
            marker="s",
            s=75,
            facecolors="none",
            edgecolors="black",
            linewidths=1.6,
            label="global peak near crop edge",
        )

    for k, snr in zip(off_expected_ks, off_expected_snrs):
        ax.annotate(
            "off",
            xy=(k, snr),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    for k, snr in zip(edge_ks, edge_snrs):
        ax.annotate(
            "edge",
            xy=(k, snr),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    ax.set_xlabel("SYSREM iteration k")
    ax.set_ylabel("Peak SNR")
    ax.set_xticks(list(ks))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    night_str = "_".join([str(n) for n in nights])
    cam_str = "_".join([str(c) for c in cameras])

    orders_str = (
        "all orders"
        if orders is None
        else f"orders {min(orders)}-{max(orders)}"
    )

    fig.suptitle(
        f"{model} {map_type} peak detections | nights={night_str} | cameras={cam_str} | {orders_str}\n"
        f"RV=[{rv_min}, {rv_max}], Kp=[{kp_min}, {kp_max}] "
        f"(negative plots mirror Kp limits)",
        fontsize=12,
        y=1.02,
    )

    fig.savefig(
        savefile,
        dpi=args.dpi,
        bbox_inches="tight",
    )

    plt.close(fig)

    return peak_summary_rows



def main():
    nights = args.nights if args.nights is not None else config.nights
    cameras = args.cameras if args.cameras is not None else config.camera

    ks = args.ks if args.ks is not None else list(range(1, args.max_k + 1))

    rv_min, rv_max, kp_min, kp_max = get_crop_limits()

    if args.output_dir is None:
        output_dir = Path(f"{config.path2reduced}/results/sysrem_iteration_maps")
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    print()
    print("=" * 60)
    print("Building SYSREM iteration maps")
    print("=" * 60)
    print(f"Model      : {args.model}")
    print(f"Type       : {args.type}")
    print(f"Nights     : {nights}")
    print(f"Cameras    : {cameras}")
    print(f"Orders     : {args.orders}")
    print(f"SYSREM ks  : {ks}")
    print(f"RV crop    : {rv_min} to {rv_max}")
    print(f"Kp crop    : {kp_min} to {kp_max}")
    print("=" * 60)

    summary_rows = []

    for k in ks:
        result = build_snr_map_for_type(
            model=args.model,
            iters=k,
            nights=nights,
            cameras=cameras,
            map_type=args.type,
            orders=args.orders,
            rv_min=rv_min,
            rv_max=rv_max,
            kp_min=kp_min,
            kp_max=kp_max,
        )

        results.append(result)

        peak = result["peak"]

        summary_rows.append(
            {
                "k": int(k),
                "peak_snr": peak["snr"],
                "peak_kp": peak["kp"],
                "peak_rv": peak["rv"],
                "noise": result["noise"],
            }
        )

        print(
            f"k={k:2d} | peak SNR={peak['snr']:.2f} "
            f"at Kp={peak['kp']:.1f}, RV={peak['rv']:.1f}"
        )

    night_str = "_".join([str(n) for n in nights])
    cam_str = "_".join([str(c) for c in cameras])
    order_str = (
        "allorders"
        if args.orders is None
        else f"orders{min(args.orders)}to{max(args.orders)}"
    )

    basename = (
        f"sysrem_maps_{night_str}_{cam_str}_{args.model}_"
        f"{args.type}_{order_str}_k{min(ks)}to{max(ks)}"
        f"_RV{rv_min:g}to{rv_max:g}_Kp{kp_min:g}to{kp_max:g}"
    )

    plot_file = output_dir / f"{basename}.png"
    summary_file = output_dir / f"{basename}_summary.json"

    plot_iteration_grid(
        results=results,
        ks=ks,
        model=args.model,
        map_type=args.type,
        nights=nights,
        cameras=cameras,
        orders=args.orders,
        rv_min=rv_min,
        rv_max=rv_max,
        kp_min=kp_min,
        kp_max=kp_max,
        savefile=plot_file,
        vmax=args.vmax,
    )

    peak_plot_file = output_dir / f"{basename}_peaks.png"
    peak_summary_file = output_dir / f"{basename}_peaks_summary.json"

    peak_summary_rows = plot_iteration_peaks(
        results=results,
        ks=ks,
        model=args.model,
        map_type=args.type,
        nights=nights,
        cameras=cameras,
        orders=args.orders,
        rv_min=rv_min,
        rv_max=rv_max,
        kp_min=kp_min,
        kp_max=kp_max,
        savefile=peak_plot_file,
    )

    with open(peak_summary_file, "w") as f:
        json.dump(
            {
                "model": args.model,
                "type": args.type,
                "nights": list(nights),
                "cameras": list(cameras),
                "orders": None if args.orders is None else list(args.orders),
                "ks": [int(k) for k in ks],
                "rv_min": rv_min,
                "rv_max": rv_max,
                "kp_min": kp_min,
                "kp_max": kp_max,
                "sigma_cut": args.sigma_cut,
                "map_sign": args.map_sign,
                "expected_kp": args.expected_kp,
                "expected_rv": args.expected_rv,
                "kp_window": args.kp_window,
                "rv_window": args.rv_window,
                "plot_file": str(peak_plot_file),
                "summary_rows": peak_summary_rows,
            },
            f,
            indent=4,
        )

    summary = {
        "model": args.model,
        "type": args.type,
        "nights": list(nights),
        "cameras": list(cameras),
        "orders": None if args.orders is None else list(args.orders),
        "ks": [int(k) for k in ks],
        "rv_min": rv_min,
        "rv_max": rv_max,
        "kp_min": kp_min,
        "kp_max": kp_max,
        "sigma_cut": args.sigma_cut,
        "map_sign": args.map_sign,
        "expected_kp": args.expected_kp,
        "expected_rv": args.expected_rv,
        "plot_file": str(plot_file),
        "summary_rows": summary_rows,
    }

    with open(summary_file, "w") as f:
        json.dump(
            summary,
            f,
            indent=4,
        )

    print()
    print("=" * 60)
    print("Saved")
    print("=" * 60)
    print(f"Map plot      : {plot_file}")
    print(f"Map summary   : {summary_file}")
    print(f"Peak plot     : {peak_plot_file}")
    print(f"Peak summary  : {peak_summary_file}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()