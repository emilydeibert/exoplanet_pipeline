from astropy.stats import sigma_clip
import matplotlib.pyplot as plt
from pathlib import Path
import importlib.util
import numpy as np
import argparse
import json
import sys

parser = argparse.ArgumentParser()

parser.add_argument("project_path", type=str)

parser.add_argument(
    "--model",
    required=True,
)

parser.add_argument(
    "--iters",
    type=int,
    required=False,
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
    help="Nights to include (default: config.nights)"
)

parser.add_argument(
    "--cameras",
    nargs="+",
    default=None,
    help="Cameras to include (default: config.camera)"
)

parser.add_argument(
    "--sigma-cut",
    type=float,
    default=3.0,
)

parser.add_argument(
    "--expected-kp",
    type=float,
    default=None,
    help="Expected Kp for local peak search / exclusion box."
)

parser.add_argument(
    "--expected-rv",
    type=float,
    default=None,
    help="Expected Vsys/RV for local peak search / exclusion box."
)

parser.add_argument(
    "--local-kp-half-width",
    type=float,
    default=20.0,
)

parser.add_argument(
    "--local-rv-half-width",
    type=float,
    default=10.0,
)

parser.add_argument(
    "--exclude-kp-half-width",
    type=float,
    default=20.0,
)

parser.add_argument(
    "--exclude-rv-half-width",
    type=float,
    default=10.0,
)

parser.add_argument(
    "--plot-snr-method",
    choices=["clip", "outside_box"],
    default="clip",
)

parser.add_argument(
    "--save-output",
    default=True,
)

# parser.add_argument(
#     "--save-plot",
#     default=None,
# )

# parser.add_argument(
#     "--save-map",
#     default=None,
# )

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

def load_and_combine_maps(
    model,
    nights,
    cameras,
    orders=None,
):
    combined_maps = []

    for camera in cameras:

        if camera == "red":
            cameraDict = config.redCameraDict
        elif camera == "blue":
            cameraDict = config.blueCameraDict
        else:
            raise ValueError(f"Unknown camera: {camera}")

        for night in nights:

            iters = config.optimumSysremIters[f"{night}_{camera}"]

            filename = (
                f"{config.path2reduced}/results/"
                f"{night}_{camera}_{model}_k{iters}_iters.npz"
            )

            data = np.load(filename)

            fmap = data["fmap"]

            # These are the original order numbers represented by fmap rows.
            # For species like Fe, this may be all orders.
            # For species like Al, this may be only selected orders, e.g. [0, 5, 13].
            if "orders" in data.files:
                file_orders = np.asarray(data["orders"], dtype=int)
            else:
                # Backward-compatible fallback: assume fmap rows correspond directly
                # to original order numbers 0, 1, 2, ...
                file_orders = np.arange(fmap.shape[0], dtype=int)

            if fmap.shape[0] != len(file_orders):
                raise ValueError(
                    f"Order mapping mismatch for {filename}: "
                    f"fmap has {fmap.shape[0]} order rows but "
                    f"data['orders'] has length {len(file_orders)}"
                )

            if orders is None:
                desired_orders = np.asarray(cameraDict["goodOrders"], dtype=int)
            else:
                desired_orders = np.asarray(orders, dtype=int)

            # Convert desired original order numbers into fmap row indices.
            keep_idx = np.where(np.isin(file_orders, desired_orders))[0]

            if len(keep_idx) == 0:
                raise ValueError(
                    f"No overlapping orders for {night} {camera} {model}. "
                    f"File orders: {file_orders.tolist()} | "
                    f"Desired orders: {desired_orders.tolist()}"
                )

            selected_original_orders = file_orders[keep_idx]
            print(
                f"{night} {camera} {model}: summing fmap rows {keep_idx.tolist()} "
                f"corresponding to original orders {selected_original_orders.tolist()}"
            )

            order_sum = np.nansum(fmap[keep_idx], axis=0)
            order_sum *= -1.0

            combined_maps.append(order_sum)

    final_map = np.nansum(combined_maps, axis=0)

    return final_map


def calculate_snr_map(
    kpvsys_map,
    sigma_cut=3.0):

    kpvsys_map = kpvsys_map - np.nanmedian(kpvsys_map)

    clipped = sigma_clip(
        kpvsys_map,
        sigma_upper=sigma_cut,
        sigma_lower=100,
    )

    noise = np.nanstd(clipped)

    snr_map = kpvsys_map / noise

    return snr_map, noise

def calculate_snr_map_outside_box(
    kpvsys_map,
    RV,
    Kp,
    expected_kp,
    expected_rv,
    kp_half_width=20.0,
    rv_half_width=10.0,
):
    kpvsys_map = kpvsys_map - np.nanmedian(kpvsys_map)

    kp_box = np.abs(Kp - expected_kp) <= kp_half_width
    rv_box = np.abs(RV - expected_rv) <= rv_half_width
    box_mask = kp_box[:, None] & rv_box[None, :]

    noise = np.nanstd(kpvsys_map[~box_mask])
    snr_map = kpvsys_map / noise

    return snr_map, noise


def find_peak_in_box(
    snr_map,
    RV,
    Kp,
    expected_kp,
    expected_rv,
    kp_half_width=20.0,
    rv_half_width=10.0,
):
    kp_box = np.abs(Kp - expected_kp) <= kp_half_width
    rv_box = np.abs(RV - expected_rv) <= rv_half_width
    box_mask = kp_box[:, None] & rv_box[None, :]

    masked_map = np.where(box_mask, snr_map, np.nan)

    max_index = np.nanargmax(masked_map)

    kp_idx, rv_idx = np.unravel_index(
        max_index,
        masked_map.shape,
    )

    return {
        "snr": float(snr_map[kp_idx, rv_idx]),
        "rv": float(RV[rv_idx]),
        "kp": float(Kp[kp_idx]),
        "rv_idx": int(rv_idx),
        "kp_idx": int(kp_idx),
    }


def find_peak(
    snr_map,
    RV,
    Kp):

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


# def plot_detection(RV, Kp, snr_map, peak, savefile = None):

#     fig = plt.figure(figsize=(8, 7))

#     # -----------------------------------------
#     # Main map
#     # -----------------------------------------
#     ax_map = fig.add_subplot(223)

#     vmax = np.nanmax(np.abs(snr_map))

#     im = ax_map.pcolormesh(
#         RV,
#         Kp,
#         snr_map,
#         shading="auto",
#         vmin=-vmax,
#         vmax=vmax,
#     )

#     ax_map.set_xlabel("Vsys [km/s]")
#     ax_map.set_ylabel("Kp [km/s]")

#     ax_map.tick_params(axis='both', which='both', labelbottom=True, labelleft=True)

#     # -----------------------------------------
#     # RV slice (top)
#     # -----------------------------------------
#     ax_rv = fig.add_subplot(221)

#     ax_rv.plot(RV, snr_map[peak["kp_idx"], :])

#     ax_rv.set_ylabel("SNR")
#     ax_rv.set_xticks([])
#     ax_rv.set_xlim(min(RV), max(RV))

#     # -----------------------------------------
#     # Kp slice (right)
#     # -----------------------------------------
#     ax_kp = fig.add_subplot(224)

#     ax_kp.plot(snr_map[:, peak["rv_idx"]], Kp)

#     ax_kp.set_xlabel("SNR")
#     ax_kp.set_yticks([])
#     ax_kp.set_ylim(min(Kp), max(Kp))

#     # -----------------------------------------
#     # Colorbar (normal, right of map)
#     # -----------------------------------------
#     cbar = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
#     cbar.set_label("SNR", rotation=270, labelpad=2)

#     # -----------------------------------------
#     # Titles
#     # -----------------------------------------
#     ax_rv.set_title(f"Kp = {peak['kp']:.1f} km/s")
#     ax_kp.set_title(f"Vsys = {peak['rv']:.1f} km/s")

#     if savefile is not None:
#         plt.savefig(savefile, dpi=300, bbox_inches="tight")

#     plt.show()
def plot_detection(
    RV,
    Kp,
    snr_map,
    peak,
    savefile=None,
    expected_kp=None,
    expected_rv=None,
    local_peak=None,
    title_suffix=None,
):

    fig = plt.figure(figsize=(8, 7))

    ax_map = fig.add_subplot(223)

    vmax = np.nanmax(np.abs(snr_map))

    im = ax_map.pcolormesh(
        RV,
        Kp,
        snr_map,
        shading="auto",
        vmin=-vmax,
        vmax=vmax,
    )

    ax_map.set_xlabel("Vsys [km/s]")
    ax_map.set_ylabel("Kp [km/s]")
    ax_map.tick_params(axis="both", which="both", labelbottom=True, labelleft=True)

    if expected_kp is not None and expected_rv is not None:
        ax_map.axhline(expected_kp, color="k", linestyle=":", linewidth=1)
        ax_map.axvline(expected_rv, color="k", linestyle=":", linewidth=1)

    ax_map.scatter(
        peak["rv"],
        peak["kp"],
        marker="x",
        s=60,
        color="k",
        linewidths=1.5,
        label="Global peak",
    )

    if local_peak is not None:
        ax_map.scatter(
            local_peak["rv"],
            local_peak["kp"],
            marker="+",
            s=70,
            color="k",
            linewidths=1.5,
            label="Local peak",
        )

    ax_rv = fig.add_subplot(221)
    ax_rv.plot(RV, snr_map[peak["kp_idx"], :])
    ax_rv.set_ylabel("SNR")
    ax_rv.set_xticks([])
    ax_rv.set_xlim(min(RV), max(RV))

    ax_kp = fig.add_subplot(224)
    ax_kp.plot(snr_map[:, peak["rv_idx"]], Kp)
    ax_kp.set_xlabel("SNR")
    ax_kp.set_yticks([])
    ax_kp.set_ylim(min(Kp), max(Kp))

    cbar = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
    cbar.set_label("SNR", rotation=270, labelpad=2)

    ax_rv.set_title(f"Kp = {peak['kp']:.1f} km/s")
    ax_kp.set_title(f"Vsys = {peak['rv']:.1f} km/s")

    if title_suffix is not None:
        fig.suptitle(title_suffix)

    if savefile is not None:
        plt.savefig(savefile, dpi=300, bbox_inches="tight")

    plt.show()


# def save_results(
#     filename,
#     combined_map,
#     snr_map,
#     peak,
#     noise,
#     model,
#     iters,
#     orders,
#     nights,
#     cameras,
#     RV,
#     Kp):

#     np.savez_compressed(
#         filename,
#         combined_map=combined_map,
#         snr_map=snr_map,
#         RV=RV,
#         Kp=Kp,
#         peak_snr=peak["snr"],
#         peak_rv=peak["rv"],
#         peak_kp=peak["kp"],
#         noise=noise,
#         model=model,
#         iters=iters,
#         orders=orders,
#         nights=np.array(nights, dtype='str'),
#         cameras=np.array(cameras, dtype='str'))
def save_results(
    filename,
    combined_map,
    snr_map,
    snr_map_clip,
    snr_map_outside_box,
    peaks,
    noises,
    model,
    iters,
    orders,
    nights,
    cameras,
    RV,
    Kp,
):

    np.savez_compressed(
        filename,
        combined_map=combined_map,
        snr_map=snr_map,  # backwards-compatible: the plotted/default SNR map
        snr_map_clip=snr_map_clip,
        snr_map_outside_box=snr_map_outside_box,
        RV=RV,
        Kp=Kp,
        peaks=json.dumps(peaks),
        noises=json.dumps(noises),
        model=model,
        iters=iters,
        orders=orders,
        nights=np.array(nights, dtype="str"),
        cameras=np.array(cameras, dtype="str"),
    )


# def save_summary(
#     filename,
#     peak,
#     noise,
#     model,
#     iters,
#     orders,
#     nights,
#     cameras):

#     summary = {
#         "model": model,
#         "iters": iters,
#         "orders": (
#             None
#             if orders is None
#             else list(orders)
#         ),
#         "peak_snr": peak["snr"],
#         "peak_rv": peak["rv"],
#         "peak_kp": peak["kp"],
#         "noise": noise,
#         "nights": list(nights),
#         "cameras": list(cameras),
#     }

#     with open(filename, "w") as f:
#         json.dump(
#             summary,
#             f,
#             indent=4,
#         )
def save_summary(
    filename,
    peaks,
    noises,
    model,
    iters,
    orders,
    nights,
    cameras,
    expected_kp,
    expected_rv,
    local_kp_half_width,
    local_rv_half_width,
    exclude_kp_half_width,
    exclude_rv_half_width,
):

    summary = {
        "model": model,
        "iters": iters,
        "orders": (
            None
            if orders is None
            else list(orders)
        ),
        "nights": list(nights),
        "cameras": list(cameras),
        "expected_kp": expected_kp,
        "expected_rv": expected_rv,
        "local_box": {
            "kp_half_width": local_kp_half_width,
            "rv_half_width": local_rv_half_width,
        },
        "exclude_box": {
            "kp_half_width": exclude_kp_half_width,
            "rv_half_width": exclude_rv_half_width,
        },
        "noises": noises,
        "peaks": peaks,

        # Backwards-compatible top-level entries using current/default method.
        "peak_snr": peaks["clip_global"]["snr"],
        "peak_rv": peaks["clip_global"]["rv"],
        "peak_kp": peaks["clip_global"]["kp"],
        "noise": noises["clip"],
    }

    with open(filename, "w") as f:
        json.dump(
            summary,
            f,
            indent=4,
        )


def main():

    nights = args.nights if args.nights is not None else config.nights
    cameras = args.cameras if args.cameras is not None else config.camera

    RV = config.RV
    Kp = config.Kp

    RV_MIN = config.RV_MIN if hasattr(config, "RV_MIN") else -75
    RV_MAX = config.RV_MAX if hasattr(config, "RV_MAX") else 75

    KP_MIN = config.KP_MIN if hasattr(config, "KP_MIN") else 50
    KP_MAX = config.KP_MAX if hasattr(config, "KP_MAX") else 275


    rv_mask = (RV >= RV_MIN) & (RV <= RV_MAX)
    kp_mask = (Kp >= KP_MIN) & (Kp <= KP_MAX)

    RV_crop = RV[rv_mask]
    Kp_crop = Kp[kp_mask]

    combined_map = load_and_combine_maps(
        model=args.model,
        #iters=args.iters,
        nights=nights,
        cameras=cameras,
        orders=args.orders,
    )

    # snr_map, noise = calculate_snr_map(
    #     combined_map[kp_mask][:, rv_mask],
    #     sigma_cut=args.sigma_cut,
    # )

    # peak = find_peak(
    #     snr_map,
    #     RV_crop,
    #     Kp_crop,
    # )
combined_map_crop = combined_map[kp_mask][:, rv_mask]

snr_map_clip, noise_clip = calculate_snr_map(
    combined_map_crop,
    sigma_cut=args.sigma_cut,
)

snr_map_outside_box = None
noise_outside_box = None

peaks = {
    "clip_global": find_peak(
        snr_map_clip,
        RV_crop,
        Kp_crop,
    )
}

noises = {
    "clip": float(noise_clip),
}

if args.expected_kp is not None and args.expected_rv is not None:

    peaks["clip_local"] = find_peak_in_box(
        snr_map_clip,
        RV_crop,
        Kp_crop,
        expected_kp=args.expected_kp,
        expected_rv=args.expected_rv,
        kp_half_width=args.local_kp_half_width,
        rv_half_width=args.local_rv_half_width,
    )

    snr_map_outside_box, noise_outside_box = calculate_snr_map_outside_box(
        combined_map_crop,
        RV_crop,
        Kp_crop,
        expected_kp=args.expected_kp,
        expected_rv=args.expected_rv,
        kp_half_width=args.exclude_kp_half_width,
        rv_half_width=args.exclude_rv_half_width,
    )

    noises["outside_box"] = float(noise_outside_box)

    peaks["outside_box_global"] = find_peak(
        snr_map_outside_box,
        RV_crop,
        Kp_crop,
    )

    peaks["outside_box_local"] = find_peak_in_box(
        snr_map_outside_box,
        RV_crop,
        Kp_crop,
        expected_kp=args.expected_kp,
        expected_rv=args.expected_rv,
        kp_half_width=args.local_kp_half_width,
        rv_half_width=args.local_rv_half_width,
    )

else:
    peaks["clip_local"] = None
    peaks["outside_box_global"] = None
    peaks["outside_box_local"] = None
    noises["outside_box"] = None

if args.plot_snr_method == "outside_box" and snr_map_outside_box is not None:
    snr_map = snr_map_outside_box
    peak = peaks["outside_box_global"]
    local_peak = peaks["outside_box_local"]
    plot_suffix = "outside-box SNR"
else:
    snr_map = snr_map_clip
    peak = peaks["clip_global"]
    local_peak = peaks["clip_local"]
    plot_suffix = "sigma-clipped SNR"

    # print()
    # print("=" * 40)
    # print("Detection Summary")
    # print("=" * 40)
    # print(f"Model     : {args.model}")
    # print(f"Peak SNR  : {peak['snr']:.2f}")
    # print(f"Peak Vsys : {peak['rv']:.2f} km/s")
    # print(f"Peak Kp   : {peak['kp']:.2f} km/s")
    # print("=" * 40)
    # print()
print()
print("=" * 40)
print("Detection Summary")
print("=" * 40)
print(f"Model     : {args.model}")
print()
print("Sigma-clipped map:")
print(f"  Global peak SNR  : {peaks['clip_global']['snr']:.2f}")
print(f"  Global peak Vsys : {peaks['clip_global']['rv']:.2f} km/s")
print(f"  Global peak Kp   : {peaks['clip_global']['kp']:.2f} km/s")

if peaks["clip_local"] is not None:
    print(f"  Local peak SNR   : {peaks['clip_local']['snr']:.2f}")
    print(f"  Local peak Vsys  : {peaks['clip_local']['rv']:.2f} km/s")
    print(f"  Local peak Kp    : {peaks['clip_local']['kp']:.2f} km/s")

if peaks["outside_box_global"] is not None:
    print()
    print("Outside-box map:")
    print(f"  Global peak SNR  : {peaks['outside_box_global']['snr']:.2f}")
    print(f"  Global peak Vsys : {peaks['outside_box_global']['rv']:.2f} km/s")
    print(f"  Global peak Kp   : {peaks['outside_box_global']['kp']:.2f} km/s")
    print(f"  Local peak SNR   : {peaks['outside_box_local']['snr']:.2f}")
    print(f"  Local peak Vsys  : {peaks['outside_box_local']['rv']:.2f} km/s")
    print(f"  Local peak Kp    : {peaks['outside_box_local']['kp']:.2f} km/s")

print("=" * 40)
print()

    if args.save_output:

        night_str = "_".join([str(n) for n in nights])
        cam_str = "_".join([str(c) for c in cameras])
        savename = f"{config.path2reduced}/results/{args.model}_{night_str}_{cam_str}_final_result"

    # plot_detection(
    #     RV_crop,
    #     Kp_crop,
    #     snr_map,
    #     peak,
    #     savefile=savename+'.png',
    # )
    plot_detection(
        RV_crop,
        Kp_crop,
        snr_map,
        peak,
        savefile=savename + ".png",
        expected_kp=args.expected_kp,
        expected_rv=args.expected_rv,
        local_peak=local_peak,
        title_suffix=plot_suffix,
    )

    if args.save_output:

        # save_results(
        #     savename+'.npz',
        #     combined_map,
        #     snr_map,
        #     peak,
        #     noise,
        #     args.model,
        #     args.iters,
        #     args.orders,
        #     nights,
        #     cameras,
        #     RV_crop,
        #     Kp_crop
        # )
        save_results(
            savename + ".npz",
            combined_map_crop,
            snr_map,
            snr_map_clip,
            snr_map_outside_box,
            peaks,
            noises,
            args.model,
            args.iters,
            args.orders,
            nights,
            cameras,
            RV_crop,
            Kp_crop,
        )

        summary_name = (
            savename+"_summary.json"
            )

        # save_summary(
        #     summary_name,
        #     peak,
        #     noise,
        #     args.model,
        #     args.iters,
        #     args.orders,
        #     nights,
        #     cameras
        # )
        save_summary(
            summary_name,
            peaks,
            noises,
            args.model,
            args.iters,
            args.orders,
            nights,
            cameras,
            args.expected_kp,
            args.expected_rv,
            args.local_kp_half_width,
            args.local_rv_half_width,
            args.exclude_kp_half_width,
            args.exclude_rv_half_width,
        )


if __name__ == "__main__":
    main()