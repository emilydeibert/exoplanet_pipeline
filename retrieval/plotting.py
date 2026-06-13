"""Diagnostic plotting for the pRT emission retrieval smoke tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Union

from .prt_emission_model import require_numpy, wavelengths_cm_to_micron


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise RuntimeError("Matplotlib is required for retrieval diagnostic plots.") from exc
    return plt


def save_raw_spectrum_plot(wavelengths_cm: Any, flux: Any, filename: Union[str, Path]) -> None:
    """Save the raw pRT emission spectrum."""

    plt = _plt()
    wave_um = wavelengths_cm_to_micron(wavelengths_cm)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wave_um, flux, lw=0.7)
    ax.set_xlabel("Wavelength [micron]")
    ax.set_ylabel("pRT emission flux")
    ax.set_title("Raw pRT emission spectrum")
    fig.tight_layout()
    fig.savefig(filename, dpi=250)
    plt.close(fig)


def save_prepared_model_plot(wavelengths_cm: Any, prepared_model: Any, filename: Union[str, Path]) -> None:
    """Save a continuum-normalized/high-pass model for the first exposure."""

    np = require_numpy()
    plt = _plt()
    wave_um = wavelengths_cm_to_micron(wavelengths_cm)
    model = np.asarray(prepared_model, dtype=float)

    if model.ndim == 3:
        y = model[0, 0]
        x = wave_um[0]
    elif model.ndim == 2:
        y = model[0]
        x = wave_um[0] if wave_um.ndim == 2 else wave_um
    else:
        y = model
        x = wave_um

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, lw=0.7)
    ax.set_xlabel("Wavelength [micron]")
    ax.set_ylabel("Prepared model")
    ax.set_title("Continuum-normalized/high-pass model")
    fig.tight_layout()
    fig.savefig(filename, dpi=250)
    plt.close(fig)


def save_shifted_model_plot(
    wavelengths_cm: Any,
    shifted_cube: Any,
    phases: Any,
    filename: Union[str, Path],
    order_index: int = 0,
) -> None:
    """Save a phase-wavelength image of the shifted model for one order."""

    np = require_numpy()
    plt = _plt()
    wave_um = wavelengths_cm_to_micron(wavelengths_cm)
    cube = np.asarray(shifted_cube, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.pcolormesh(
        wave_um[order_index],
        phases,
        cube[order_index],
        shading="auto",
    )
    ax.set_xlabel("Wavelength [micron]")
    ax.set_ylabel("Orbital phase")
    ax.set_title(f"Shifted model cube, order {order_index}")
    fig.colorbar(image, ax=ax, label="Model")
    fig.tight_layout()
    fig.savefig(filename, dpi=250)
    plt.close(fig)


def save_data_mask_plot(data: Any, model_cube: Optional[Any], filename: Union[str, Path]) -> None:
    """Save a compact data/mask dimensionality diagnostic."""

    np = require_numpy()
    plt = _plt()

    n_orders, n_exp, n_pix = data.flux.shape
    valid_fraction_by_order = np.mean(data.good_mask, axis=(1, 2))
    valid_fraction_by_exp = np.mean(data.good_mask, axis=(0, 2))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(np.arange(n_orders), valid_fraction_by_order)
    axes[0].set_xlabel("Order")
    axes[0].set_ylabel("Valid fraction")
    axes[0].set_ylim(0, 1)

    axes[1].plot(np.arange(n_exp), valid_fraction_by_exp, marker=".", lw=0.8)
    axes[1].set_xlabel("Exposure")
    axes[1].set_ylabel("Valid fraction")
    axes[1].set_ylim(0, 1)

    title = f"Data shape {data.flux.shape}"
    if model_cube is not None:
        title += f"; model shape {model_cube.shape}"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=250)
    plt.close(fig)


def save_kp_vsys_likelihood_plot(
    Kp_grid: Any,
    Vsys_grid: Any,
    log_likelihood: Any,
    filename: Union[str, Path],
    expected: Optional[Mapping[str, float]] = None,
) -> None:
    """Save the Kp-Vsys likelihood map."""

    np = require_numpy()
    plt = _plt()
    Kp_grid = np.asarray(Kp_grid, dtype=float)
    Vsys_grid = np.asarray(Vsys_grid, dtype=float)
    ll = np.asarray(log_likelihood, dtype=float)
    delta = ll - np.nanmax(ll)

    best_idx = np.nanargmax(ll)
    best_k, best_v = np.unravel_index(best_idx, ll.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.pcolormesh(Vsys_grid, Kp_grid, delta, shading="auto")
    ax.scatter([Vsys_grid[best_v]], [Kp_grid[best_k]], color="white", edgecolor="black", s=45, label="best")
    if expected is not None:
        ax.scatter(
            [float(expected["Vsys"])],
            [float(expected["Kp"])],
            marker="x",
            color="red",
            s=55,
            label="expected",
        )
    ax.set_xlabel("Vsys/rest-frame velocity [km/s]")
    ax.set_ylabel("Kp [km/s]")
    ax.set_title("Fe-only Kp-Vsys log-likelihood")
    ax.legend(loc="best")
    fig.colorbar(image, ax=ax, label="Delta log likelihood")
    fig.tight_layout()
    fig.savefig(filename, dpi=250)
    plt.close(fig)


def save_best_fit_model_plot(data: Any, prepared_model: Any, amplitude: float, filename: Union[str, Path]) -> None:
    """Save a first-order best-fit model/data comparison for one order/exposure."""

    np = require_numpy()
    plt = _plt()
    wave_um = wavelengths_cm_to_micron(data.wavelengths_cm)
    valid = data.good_mask[0, 0]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wave_um[0, valid], data.flux[0, 0, valid], lw=0.7, label="data")
    ax.plot(
        wave_um[0, valid],
        float(amplitude) * np.asarray(prepared_model)[0, 0, valid],
        lw=0.7,
        label="scaled model",
    )
    ax.set_xlabel("Wavelength [micron]")
    ax.set_ylabel("Prepared units")
    ax.set_title("Best grid model vs data, order 0 exposure 0")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(filename, dpi=250)
    plt.close(fig)
