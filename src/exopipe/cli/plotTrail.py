from scipy import interpolate
import matplotlib.pyplot as plt
from pathlib import Path
import importlib.util
import numpy as np
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("project_path", type=str)

parser.add_argument("--model", required=True)

parser.add_argument(
    "--iters",
    type=int,
    default=None,
    help="SYSREM iteration to plot. Default: config.optimumSysremIters[night_camera].",
)

parser.add_argument(
    "--ks",
    type=str,
    default=None,
    help="Comma-separated list of SYSREM iterations to plot, e.g. 1,3,5,10.",
)

parser.add_argument(
    "--orders",
    nargs="+",
    type=int,
    default=None,
    help="Original order numbers to sum. Default: cameraDict['goodOrders'].",
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
    "--result-tag",
    default="",
    help="Optional tag in xcorr result filename, e.g. transmission_T2700.",
)

parser.add_argument(
    "--map-sign",
    type=float,
    default=-1.0,
    help="Factor applied after summing orders. Default -1 matches current emission getResults behavior.",
)

parser.add_argument(
    "--frame",
    choices=["stellar", "planet", "both"],
    default="both",
    help="Which trail frame to plot.",
)

parser.add_argument(
    "--kp-planet",
    type=float,
    default=None,
    help="Kp used to shift to the planet frame. If omitted, tries params.Kp or params.K_p.",
)

parser.add_argument(
    "--rv-offset",
    type=float,
    default=0.0,
    help="Extra velocity offset added when shifting to planet frame.",
)

parser.add_argument(
    "--use-centered-phase",
    action="store_true",
    help="Plot y-axis as centered phase instead of raw phase.",
)

parser.add_argument(
    "--vpercentile",
    type=float,
    default=99.0,
    help="Percentile for symmetric color scaling.",
)

parser.add_argument(
    "--save-output",
    action="store_true",
    help="Save PNGs instead of only showing plots.",
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


def get_camera_dict(camera):
    if camera == "red":
        return config.redCameraDict
    if camera == "blue":
        return config.blueCameraDict
    raise ValueError(f"Unknown camera: {camera}")


def get_k_list(nights, cameras):
    if args.ks is not None:
        return [int(x) for x in args.ks.split(",")]

    if args.iters is not None:
        return [args.iters]

    # None means use optimum per night/camera inside the loop.
    return [None]


def get_planet_kp():
    if args.kp_planet is not None:
        return args.kp_planet

    if hasattr(params, "Kp"):
        val = params.Kp
    elif hasattr(params, "K_p"):
        val = params.K_p
    else:
        raise ValueError(
            "Need --kp-planet for planet-frame plot, because params.Kp/K_p was not found."
        )

    if hasattr(val, "to_value"):
        return float(val.to_value("km/s"))

    return float(val)


def load_phase(night, camera, xcorr_data):
    if "phase" in xcorr_data.files:
        phase = np.asarray(xcorr_data["phase"], dtype=float)
    else:
        data_array = np.load(f"{config.path2reduced}/{night}_{camera}_analysis_ready.npz")
        phase = np.asarray(data_array["phase"], dtype=float)

    if "phase_centered" in xcorr_data.files:
        phase_centered = np.asarray(xcorr_data["phase_centered"], dtype=float)
    else:
        phase_centered = (phase + 0.5) % 1.0 - 0.5

    return phase, phase_centered


def load_summed_cmap(model, night, camera, k, orders=None, result_tag="", map_sign=-1.0):
    camera_dict = get_camera_dict(camera)

    if k is None:
        k = config.optimumSysremIters[f"{night}_{camera}"]

    tag = f"_{result_tag}" if result_tag else ""

    filename = (
        f"{config.path2reduced}/results/"
        f"{night}_{camera}_{model}_k{k}_iters{tag}.npz"
    )

    print(f"Loading {filename}")
    data = np.load(filename)

    cmap = data["cmap"]  # (n_order_rows, n_exp, n_rv)

    if "RV" in data.files:
        RV = np.asarray(data["RV"], dtype=float)
    else:
        RV = np.asarray(config.RV, dtype=float)

    if "orders" in data.files:
        file_orders = np.asarray(data["orders"], dtype=int)
    else:
        file_orders = np.arange(cmap.shape[0], dtype=int)

    if cmap.shape[0] != len(file_orders):
        raise ValueError(
            f"Order mapping mismatch for {filename}: "
            f"cmap has {cmap.shape[0]} order rows but "
            f"data['orders'] has length {len(file_orders)}"
        )

    if orders is None:
        desired_orders = np.asarray(camera_dict["goodOrders"], dtype=int)
    else:
        desired_orders = np.asarray(orders, dtype=int)

    keep_idx = np.where(np.isin(file_orders, desired_orders))[0]

    if len(keep_idx) == 0:
        raise ValueError(
            f"No overlapping orders for {night} {camera} {model}. "
            f"File orders: {file_orders.tolist()} | "
            f"Desired orders: {desired_orders.tolist()}"
        )

    selected_original_orders = file_orders[keep_idx]
    print(
        f"{night} {camera} {model} k={k}: summing cmap rows {keep_idx.tolist()} "
        f"corresponding to original orders {selected_original_orders.tolist()}"
    )

    trail = np.nansum(cmap[keep_idx], axis=0)
    trail *= map_sign

    phase, phase_centered = load_phase(night, camera, data)

    return trail, RV, phase, phase_centered, k, selected_original_orders


def shift_trail_to_planet_frame(trail, RV, phase, kp, rv_offset=0.0):
    shifted = np.full_like(trail, np.nan, dtype=float)

    for edx in range(trail.shape[0]):
        vp = kp * np.sin(2.0 * np.pi * phase[edx]) + rv_offset

        # Same convention as finalCorr_stack: shifted RV axis = RV - vp
        x = RV - vp
        y = trail[edx, :]

        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 5:
            continue

        xs = x[m]
        ys = y[m]
        s = np.argsort(xs)

        f = interpolate.interp1d(
            xs[s],
            ys[s],
            bounds_error=False,
            fill_value=np.nan,
        )

        shifted[edx, :] = f(RV)

    return shifted


def plot_trail(
    trail,
    RV,
    phase,
    title,
    savefile=None,
    vpercentile=99.0,
):
    vmax = np.nanpercentile(np.abs(trail), vpercentile)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    im = ax.pcolormesh(
        RV,
        phase,
        trail,
        shading="auto",
        vmin=-vmax,
        vmax=vmax,
    )

    ax.set_xlabel("RV [km/s]")
    ax.set_ylabel("Orbital phase")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Summed CCF")

    fig.tight_layout()

    if savefile is not None:
        fig.savefig(savefile, dpi=300, bbox_inches="tight")
        print(f"Saved {savefile}")

    plt.show()


def main():
    nights = args.nights if args.nights is not None else config.nights
    cameras = args.cameras if args.cameras is not None else config.camera
    k_list = get_k_list(nights, cameras)

    for night in nights:
        for camera in cameras:
            for k_in in k_list:

                trail, RV, phase, phase_centered, k, selected_orders = load_summed_cmap(
                    model=args.model,
                    night=night,
                    camera=camera,
                    k=k_in,
                    orders=args.orders,
                    result_tag=args.result_tag,
                    map_sign=args.map_sign,
                )

                yphase = phase_centered if args.use_centered_phase else phase
                phase_label = "centered phase" if args.use_centered_phase else "phase"

                base = (
                    f"{args.model}_{night}_{camera}_k{k}"
                    f"{'_' + args.result_tag if args.result_tag else ''}"
                )

                if args.frame in ["stellar", "both"]:
                    title = (
                        f"{args.model} {night} {camera} k={k} "
                        f"stellar frame, y={phase_label}"
                    )

                    savefile = None
                    if args.save_output:
                        savefile = f"{config.path2reduced}/results/{base}_trail_stellar.png"

                    plot_trail(
                        trail,
                        RV,
                        yphase,
                        title,
                        savefile=savefile,
                        vpercentile=args.vpercentile,
                    )

                if args.frame in ["planet", "both"]:
                    kp_planet = get_planet_kp()

                    trail_planet = shift_trail_to_planet_frame(
                        trail,
                        RV,
                        phase,
                        kp=kp_planet,
                        rv_offset=args.rv_offset,
                    )

                    title = (
                        f"{args.model} {night} {camera} k={k} "
                        f"planet frame, Kp={kp_planet:.1f} km/s, "
                        f"offset={args.rv_offset:.1f} km/s"
                    )

                    savefile = None
                    if args.save_output:
                        savefile = f"{config.path2reduced}/results/{base}_trail_planet.png"

                    plot_trail(
                        trail_planet,
                        RV,
                        yphase,
                        title,
                        savefile=savefile,
                        vpercentile=args.vpercentile,
                    )


if __name__ == "__main__":
    main()
