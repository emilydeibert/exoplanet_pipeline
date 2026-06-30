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
    nargs="+",
    required=True,
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
    "--cmap",
    type=str,
    default="viridis",
)

parser.add_argument(
    "--sigma-cut",
    type=float,
    default=3.0,
)

parser.add_argument(
    "--save-output",
    default=False,
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
    model: str,
    iters: list,
    nights: list,
    cameras: list,
    orders=None):
    
    combined_maps = []
    
    if len(iters) != len(nights):
        iters*= len(nights)

    for camera in cameras:
        
        for night, iter in zip(nights, iters):

            filename = (
                f"{config.path2reduced}/results/"
                f"{night}_{camera}_{model}_k{iter}_iters.npz"
            )

            data = np.load(filename)

            fmap = data["fmap"]

            if orders is None:
                order_sum = np.nansum(fmap, axis=0)
            else:
                order_sum = np.nansum(fmap[orders], axis=0)

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


def plot_detection(RV, Kp, snr_map, peak, savefile = None):

    fig = plt.figure(figsize=(8, 7))
    # -----------------------------------------
    # Main map
    # -----------------------------------------
    ax_map = fig.add_subplot(223)

    vmax = np.nanmax(np.abs(snr_map))

    im = ax_map.pcolormesh(
        RV,
        Kp,
        snr_map,
        shading="auto",
        cmap=args.cmap,
        vmin=-vmax,
        vmax=vmax,
    )

    ax_map.set_xlabel("Vsys [km/s]")
    ax_map.set_ylabel("Kp [km/s]")

    ax_map.tick_params(axis='both', which='both', labelbottom=True, labelleft=True)

    # -----------------------------------------
    # RV slice (top)
    # -----------------------------------------
    ax_rv = fig.add_subplot(221)

    ax_rv.plot(RV, snr_map[peak["kp_idx"], :])

    ax_rv.set_ylabel("SNR")
    ax_rv.set_xticks([])
    ax_rv.set_xlim(min(RV), max(RV))

    # -----------------------------------------
    # Kp slice (right)
    # -----------------------------------------
    ax_kp = fig.add_subplot(224)

    ax_kp.plot(snr_map[:, peak["rv_idx"]], Kp)

    ax_kp.set_xlabel("SNR")
    ax_kp.set_yticks([])
    ax_kp.set_ylim(min(Kp), max(Kp))

    # -----------------------------------------
    # Colorbar (normal, right of map)
    # -----------------------------------------
    cbar = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
    cbar.set_label("SNR", rotation=270, labelpad=2)

    # -----------------------------------------
    # Titles
    # -----------------------------------------
    ax_rv.set_title(f"Kp = {peak['kp']:.1f} km/s")
    ax_kp.set_title(f"Vsys = {peak['rv']:.1f} km/s")

    if savefile is not None:
        plt.savefig(savefile, dpi=300, bbox_inches="tight")

    plt.show()


def save_results(
    filename,
    combined_map,
    snr_map,
    peak,
    noise,
    model,
    iters,
    orders,
    nights,
    cameras,
    RV,
    Kp):

    np.savez_compressed(
        filename,
        combined_map=combined_map,
        snr_map=snr_map,
        RV=RV,
        Kp=Kp,
        peak_snr=peak["snr"],
        peak_rv=peak["rv"],
        peak_kp=peak["kp"],
        noise=noise,
        model=model,
        iters=iters,
        orders=orders,
        nights=np.array(nights, dtype='str'),
        cameras=np.array(cameras, dtype='str'))


def save_summary(
    filename,
    peak,
    noise,
    model,
    iters,
    orders,
    nights,
    cameras):

    summary = {
        "model": model,
        "iters": iters,
        "orders": (
            None
            if orders is None
            else list(orders)
        ),
        "peak_snr": peak["snr"],
        "peak_rv": peak["rv"],
        "peak_kp": peak["kp"],
        "noise": noise,
        "nights": list(nights),
        "cameras": list(cameras),
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

    KP_MIN = config.KP_MIN if hasattr(config, "KP_MIN") else 1
    KP_MAX = config.KP_MAX if hasattr(config, "KP_MAX") else 300


    rv_mask = (RV >= RV_MIN) & (RV <= RV_MAX)
    kp_mask = (Kp >= KP_MIN) & (Kp <= KP_MAX)

    RV_crop = RV[rv_mask]
    Kp_crop = Kp[kp_mask]

    combined_map = load_and_combine_maps(
        model=args.model,
        iters=args.iters,
        nights=nights,
        cameras=cameras,
        orders=args.orders,
    )

    snr_map, noise = calculate_snr_map(
        combined_map[kp_mask][:, rv_mask],
        sigma_cut=args.sigma_cut,
    )

    peak = find_peak(
        snr_map,
        RV_crop,
        Kp_crop,
    )

    print()
    print("=" * 40)
    print("Detection Summary")
    print("=" * 40)
    print(f"Model     : {args.model}")
    print(f"Peak SNR  : {peak['snr']:.2f}")
    print(f"Peak Vsys : {peak['rv']:.2f} km/s")
    print(f"Peak Kp   : {peak['kp']:.2f} km/s")
    print("=" * 40)
    print()

    if args.save_output:

        night_str = "_".join([str(n) for n in nights])
        cam_str = "_".join([str(c) for c in cameras])
        savename = f"{config.path2reduced}/results/{args.model}_{night_str}_{cam_str}_final_result"

    plot_detection(
        RV_crop,
        Kp_crop,
        snr_map,
        peak,
        savefile=savename+'.png',
    )

    if args.save_output:

        save_results(
            savename+'.npz',
            combined_map,
            snr_map,
            peak,
            noise,
            args.model,
            args.iters,
            args.orders,
            nights,
            cameras,
            RV_crop,
            Kp_crop
        )

        summary_name = (
            savename+"_summary.json"
            )

        save_summary(
            summary_name,
            peak,
            noise,
            args.model,
            args.iters,
            args.orders,
            nights,
            cameras
        )


if __name__ == "__main__":
    main()