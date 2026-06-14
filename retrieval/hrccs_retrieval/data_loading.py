"""Data loading helpers for the HRCCS/CCF retrieval path.

These functions intentionally follow the working xcorr scripts:

* load ``project_path/config.py`` and ``project_path/parameters.py``
* read ``{night}_{camera}_sysrem.npz`` and ``{night}_{camera}_analysis_ready.npz``
* select ``sysrem[k-1]``
* shift each order to the same BERV/system frame with ``tools.shift2rest``

The returned data are cached in memory and reused for every pRT model
evaluation.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence


@dataclass
class ObservationBlock:
    """One night/camera data block after the trusted BERV/rest-frame shift."""

    night: str
    camera: str
    orders: list[int]
    wave: Any
    data_rest: Any
    error_rest: Any
    phase: Any
    berv: Any
    sysrem_iteration: int


def require_numpy():
    """Import NumPy lazily so ``--help`` works in bare environments."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise RuntimeError("NumPy is required for HRCCS retrieval runs.") from exc
    return np


def _repo_src_path() -> Path:
    return Path(__file__).resolve().parents[2] / "src"


def ensure_exopipe_importable() -> None:
    """Make local ``src/exopipe`` importable when the package is not installed."""

    try:
        import exopipe  # noqa: F401
        return
    except ImportError:
        src_path = _repo_src_path()
        if src_path.exists() and str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))


def load_python_module(path: Path, module_name: str) -> Any:
    """Load a small project config module from an explicit path."""

    if not path.exists():
        raise FileNotFoundError(f"Required project file does not exist: {path}")

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_project_modules(project_path: str | Path) -> tuple[Any, Any]:
    """Load ``config.py`` and ``parameters.py`` from a project directory."""

    project_path = Path(project_path)
    config = load_python_module(project_path / "config.py", "hrccs_project_config")
    params = load_python_module(project_path / "parameters.py", "hrccs_project_parameters")
    return config, params


def split_cli_list(value: Optional[Any]) -> Optional[list[str]]:
    """Parse comma-separated CLI lists while also accepting repeated strings."""

    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)):
        raw_chunks = [str(item) for item in value]
    else:
        raw_chunks = [str(value)]

    pieces: list[str] = []
    for raw_chunk in raw_chunks:
        for chunk in raw_chunk.replace(";", ",").split(","):
            chunk = chunk.strip()
            if chunk:
                pieces.append(chunk)
    return pieces or None


def parse_int_list(value: Optional[Any]) -> Optional[list[int]]:
    """Parse comma/semicolon-separated integer order lists."""

    pieces = split_cli_list(value)
    if pieces is None:
        return None
    return [int(piece) for piece in pieces]


def get_nights_cameras(config: Any, nights: Optional[Sequence[str]], cameras: Optional[Sequence[str]]) -> tuple[list[str], list[str]]:
    """Resolve night/camera selections against the project config."""

    selected_nights = list(nights) if nights is not None else list(config.nights)
    selected_cameras = list(cameras) if cameras is not None else list(config.camera)
    if not selected_nights:
        raise ValueError("No nights selected.")
    if not selected_cameras:
        raise ValueError("No cameras selected.")
    return selected_nights, selected_cameras


def orders_from_model(wave: Any, model_array: Any, order_threshold: float = 1.0e-6) -> list[int]:
    """Use the existing ``tools.orders2keep`` model-overlap rule."""

    ensure_exopipe_importable()
    from exopipe import tools

    orders = tools.orders2keep(wave, float(order_threshold), model_array)
    return [int(order) for order in orders]


def _select_orders(wave: Any, model_array: Optional[Any], orders: Optional[Sequence[int]], order_threshold: float) -> list[int]:
    np = require_numpy()
    n_orders = int(np.asarray(wave).shape[0])

    if orders is not None:
        selected = [int(order) for order in orders]
    elif model_array is not None:
        selected = orders_from_model(wave, model_array, order_threshold=order_threshold)
    else:
        selected = list(range(n_orders))

    bad = [order for order in selected if order < 0 or order >= n_orders]
    if bad:
        raise ValueError(f"Selected orders outside available range 0..{n_orders - 1}: {bad}")
    if not selected:
        raise ValueError("No orders selected for HRCCS retrieval.")
    return selected


def _load_npz(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required data file does not exist: {path}")
    return require_numpy().load(path, allow_pickle=False)


def load_one_night_camera(
    config: Any,
    params: Any,
    night: str,
    camera: str,
    k: int,
    model_array: Optional[Any] = None,
    orders: Optional[Sequence[int]] = None,
    order_threshold: float = 1.0e-6,
    logger: Optional[logging.Logger] = None,
) -> ObservationBlock:
    """Load and BERV-shift one night/camera block using the old xcorr convention."""

    np = require_numpy()
    ensure_exopipe_importable()
    from astropy import units as u
    from exopipe import tools

    start = time.perf_counter()
    reduced = Path(str(config.path2reduced))
    sysrem_path = reduced / f"{night}_{camera}_sysrem.npz"
    analysis_path = reduced / f"{night}_{camera}_analysis_ready.npz"

    with _load_npz(sysrem_path) as sysr_array:
        sysrem = np.asarray(sysr_array["sysrem"], dtype=float)
        magerr = np.asarray(sysr_array["magerr"], dtype=float)

    with _load_npz(analysis_path) as data_array:
        wave = np.asarray(data_array["wave"], dtype=float)
        berv = np.asarray(data_array["berv"], dtype=float)
        phase = np.asarray(data_array["phase"], dtype=float)

    if k < 1 or k > sysrem.shape[0]:
        raise ValueError(f"Requested SYSREM k={k}, but sysrem has {sysrem.shape[0]} iterations.")

    sysr_k = np.asarray(sysrem[k - 1], dtype=float)
    if magerr.shape != sysr_k.shape:
        raise ValueError(f"magerr shape {magerr.shape} does not match sysrem[k-1] shape {sysr_k.shape}.")
    if wave.shape[0] != sysr_k.shape[0] or wave.shape[2] != sysr_k.shape[2]:
        raise ValueError(
            "wave must have shape (n_orders, n_exposures, n_pixels) compatible "
            f"with sysrem[k-1]; got wave={wave.shape}, sysr_k={sysr_k.shape}."
        )

    selected_orders = _select_orders(wave, model_array, orders, order_threshold)
    data_rest = np.full((len(selected_orders), sysr_k.shape[1], sysr_k.shape[2]), np.nan, dtype=float)
    error_rest = np.full_like(data_rest, np.nan)
    wave_selected = np.full((len(selected_orders), sysr_k.shape[2]), np.nan, dtype=float)

    vsys = params.Vsys * np.ones_like(berv)
    for idx, original_order in enumerate(selected_orders):
        # This line intentionally mirrors the working xcorr script exactly.
        shifted, shifted_error = tools.shift2rest(
            sysr_k[original_order],
            wave[original_order][0],
            -1.0 * berv * u.km / u.s + vsys,
            magerr[original_order],
        )
        data_rest[idx] = shifted
        error_rest[idx] = shifted_error
        wave_selected[idx] = wave[original_order, 0, :]

    if logger is not None:
        logger.info(
            "Loaded %s %s k=%d: %d orders, %d exposures, %d pixels in %.2fs",
            night,
            camera,
            k,
            len(selected_orders),
            data_rest.shape[1],
            data_rest.shape[2],
            time.perf_counter() - start,
        )
        logger.info("%s %s selected original orders: %s", night, camera, selected_orders)

    return ObservationBlock(
        night=str(night),
        camera=str(camera),
        orders=selected_orders,
        wave=wave_selected,
        data_rest=data_rest,
        error_rest=error_rest,
        phase=phase,
        berv=berv,
        sysrem_iteration=int(k),
    )


def load_hrccs_data(
    config: Any,
    params: Any,
    k: int,
    nights: Optional[Sequence[str]] = None,
    cameras: Optional[Sequence[str]] = None,
    model_array: Optional[Any] = None,
    orders: Optional[Sequence[int]] = None,
    order_threshold: float = 1.0e-6,
    logger: Optional[logging.Logger] = None,
) -> list[ObservationBlock]:
    """Load all requested night/camera blocks once for repeated model evaluation."""

    selected_nights, selected_cameras = get_nights_cameras(config, nights, cameras)
    blocks: list[ObservationBlock] = []
    for night in selected_nights:
        for camera in selected_cameras:
            blocks.append(
                load_one_night_camera(
                    config=config,
                    params=params,
                    night=str(night),
                    camera=str(camera),
                    k=int(k),
                    model_array=model_array,
                    orders=orders,
                    order_threshold=order_threshold,
                    logger=logger,
                )
            )
    return blocks


def block_summary(blocks: Sequence[ObservationBlock]) -> dict[str, Any]:
    """Return a compact JSON-friendly data summary."""

    np = require_numpy()
    n_blocks = len(blocks)
    n_orders = int(sum(len(block.orders) for block in blocks))
    n_valid = 0
    n_total = 0
    details = []
    for block in blocks:
        valid = np.isfinite(block.data_rest) & np.isfinite(block.error_rest) & (block.error_rest > 0)
        n_valid += int(np.sum(valid))
        n_total += int(valid.size)
        details.append(
            {
                "night": block.night,
                "camera": block.camera,
                "orders": list(block.orders),
                "n_exposures": int(block.data_rest.shape[1]),
                "n_pixels": int(block.data_rest.shape[2]),
                "finite_fraction": float(np.sum(valid) / valid.size),
            }
        )

    return {
        "n_blocks": n_blocks,
        "n_order_blocks": n_orders,
        "valid_fraction": float(n_valid / n_total) if n_total else 0.0,
        "blocks": details,
    }
