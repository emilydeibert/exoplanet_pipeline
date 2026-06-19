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
from typing import Any, Mapping, Optional, Sequence

from retrieval.prt_emission_model import C_KM_S


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


def parse_k_per_night(value: Optional[Any]) -> dict[str, int]:
    """Parse ``--k-per-night`` entries of the form ``NIGHT:K``.

    The map is intentionally keyed by night only for now.  Keeping this as a
    helper makes it straightforward to extend to night+camera keys later
    without changing the sampler entry points again.
    """

    pieces = split_cli_list(value)
    if pieces is None:
        return {}

    parsed: dict[str, int] = {}
    for piece in pieces:
        if ":" not in piece:
            raise ValueError(
                "--k-per-night entries must have the form NIGHT:K, "
                f"for example 20240528:4; got {piece!r}."
            )
        night, raw_k = piece.split(":", 1)
        night = night.strip()
        raw_k = raw_k.strip()
        if not night:
            raise ValueError(f"--k-per-night entry has an empty night: {piece!r}.")
        try:
            k = int(raw_k)
        except ValueError as exc:
            raise ValueError(f"SYSREM k for night {night!r} must be an integer; got {raw_k!r}.") from exc
        if k < 1:
            raise ValueError(f"SYSREM k for night {night!r} must be >= 1; got {k}.")
        if night in parsed and parsed[night] != k:
            raise ValueError(f"Conflicting SYSREM k values for night {night!r}: {parsed[night]} and {k}.")
        parsed[night] = k
    return parsed


def resolve_sysrem_iteration_for_night(
    night: str,
    global_k: Optional[int],
    k_per_night: Optional[Mapping[str, int]] = None,
) -> int:
    """Return the SYSREM iteration for one selected night."""

    night_key = str(night)
    night_map = {str(key): int(value) for key, value in dict(k_per_night or {}).items()}
    if night_key in night_map:
        return int(night_map[night_key])
    if global_k is not None:
        k = int(global_k)
        if k < 1:
            raise ValueError(f"--k must be >= 1 when supplied; got {k}.")
        return k
    raise ValueError(
        f"No SYSREM iteration was provided for selected night {night_key!r}. "
        "Use --k for a global fallback or --k-per-night NIGHT:K."
    )


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
    k: Optional[int],
    k_per_night: Optional[Mapping[str, int]] = None,
    nights: Optional[Sequence[str]] = None,
    cameras: Optional[Sequence[str]] = None,
    model_array: Optional[Any] = None,
    orders: Optional[Sequence[int]] = None,
    order_threshold: float = 1.0e-6,
    logger: Optional[logging.Logger] = None,
) -> list[ObservationBlock]:
    """Load all requested night/camera blocks once for repeated model evaluation."""

    selected_nights, selected_cameras = get_nights_cameras(config, nights, cameras)
    if logger is not None:
        logger.info("Selected nights: %s", selected_nights)
        logger.info("Selected cameras: %s", selected_cameras)
        logger.info(
            "SYSREM selection: global k=%s, k_per_night=%s",
            None if k is None else int(k),
            {str(key): int(value) for key, value in dict(k_per_night or {}).items()},
        )
    blocks: list[ObservationBlock] = []
    for night in selected_nights:
        night_k = resolve_sysrem_iteration_for_night(str(night), k, k_per_night)
        for camera in selected_cameras:
            blocks.append(
                load_one_night_camera(
                    config=config,
                    params=params,
                    night=str(night),
                    camera=str(camera),
                    k=int(night_k),
                    model_array=model_array,
                    orders=orders,
                    order_threshold=order_threshold,
                    logger=logger,
                )
            )
    return blocks


def sysrem_iterations_by_night(blocks: Sequence[ObservationBlock]) -> dict[str, int]:
    """Return the selected SYSREM iteration per night and reject mixed cameras."""

    by_night: dict[str, int] = {}
    for block in blocks:
        night = str(block.night)
        k = int(block.sysrem_iteration)
        if night in by_night and by_night[night] != k:
            raise ValueError(
                "Mixed SYSREM iterations for the same night are not supported yet: "
                f"night={night!r}, values={by_night[night]} and {k}. "
                "The current --k-per-night map is keyed by night only."
            )
        by_night[night] = k
    return by_night


def scalar_sysrem_iteration_if_uniform(blocks: Sequence[ObservationBlock]) -> Optional[int]:
    """Return the single k value only when every block used the same SYSREM k."""

    values = {int(block.sysrem_iteration) for block in blocks}
    if len(values) == 1:
        return int(next(iter(values)))
    return None


def _infer_wave_unit_and_convert_to_micron(wavelengths: Any) -> tuple[Any, str]:
    """Infer HRCCS wavelength units and return microns for logging.

    The existing xcorr path uses nanometers for GHOST order wavelengths, while
    retrieval YAML model boundaries are in microns.  This helper keeps the
    logging explicit without changing any data arrays.
    """

    np = require_numpy()
    wave = np.asarray(wavelengths, dtype=float)
    finite = wave[np.isfinite(wave)]
    if finite.size == 0:
        raise ValueError("No finite wavelengths available for padding diagnostics.")
    median = float(np.nanmedian(finite))
    if median > 10.0:
        return wave / 1000.0, "nm"
    return wave, "micron"


def _max_prior_abs(priors: Mapping[str, Any], name: str) -> Optional[float]:
    values = priors.get(name)
    if values is None:
        return None
    try:
        lo, hi = values
        return max(abs(float(lo)), abs(float(hi)))
    except Exception:
        return None


def log_model_data_wavelength_padding(
    blocks: Sequence[ObservationBlock],
    retrieval_config: Mapping[str, Any],
    logger: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    """Log pRT model-boundary padding relative to selected HRCCS data.

    This diagnostic is intentionally read-only: it does not affect data/order
    selection or pRT wavelength boundaries.  It warns when model-generation
    boundaries are close enough to the selected data wavelengths that Doppler
    shifts may create velocity-dependent finite-overlap artifacts.
    """

    np = require_numpy()
    if logger is None:
        logger = logging.getLogger("retrieval")

    if not blocks:
        raise ValueError("No HRCCS data blocks supplied for wavelength padding diagnostics.")

    all_wave = np.concatenate([np.asarray(block.wave, dtype=float).reshape(-1) for block in blocks])
    all_wave_micron, inferred_unit = _infer_wave_unit_and_convert_to_micron(all_wave)
    finite_wave = all_wave_micron[np.isfinite(all_wave_micron)]
    if finite_wave.size == 0:
        raise ValueError("Selected HRCCS data wavelengths have no finite values.")

    data_min = float(np.nanmin(finite_wave))
    data_max = float(np.nanmax(finite_wave))
    model_bounds = retrieval_config.get("model", {}).get("wavelength_boundaries_micron", None)
    if model_bounds is None:
        logger.info(
            "No model.wavelength_boundaries_micron configured; cannot log pRT/data wavelength padding."
        )
        return {
            "data_wavelength_min_micron": data_min,
            "data_wavelength_max_micron": data_max,
            "data_wavelength_unit_inferred": inferred_unit,
            "model_wavelength_boundaries_micron": None,
        }

    model_min = float(model_bounds[0])
    model_max = float(model_bounds[1])
    blue_padding = data_min - model_min
    red_padding = model_max - data_max

    max_abs_berv = 0.0
    for block in blocks:
        berv = np.asarray(block.berv, dtype=float)
        finite_berv = berv[np.isfinite(berv)]
        if finite_berv.size:
            max_abs_berv = max(max_abs_berv, float(np.nanmax(np.abs(finite_berv))))

    priors = retrieval_config.get("priors", {})
    max_abs_kp = _max_prior_abs(priors, "Kp")
    max_abs_vsys = _max_prior_abs(priors, "Vsys")
    velocity_cfg = retrieval_config.get("velocity", {})
    if isinstance(velocity_cfg, Mapping) and str(velocity_cfg.get("mode", "")) == "per_night_offsets":
        mapping = velocity_cfg.get("per_night_offsets", {})
        if isinstance(mapping, Mapping):
            per_night_maxima = [
                value
                for value in (_max_prior_abs(priors, str(parameter_name)) for parameter_name in mapping.values())
                if value is not None
            ]
            max_abs_vsys = max(per_night_maxima) if per_night_maxima else None
    max_abs_velocity = None
    required_padding_blue = None
    required_padding_red = None
    if max_abs_kp is not None or max_abs_vsys is not None:
        max_abs_velocity = float(max_abs_kp or 0.0) + float(max_abs_vsys or 0.0) + max_abs_berv
        required_padding_blue = data_min * max_abs_velocity / C_KM_S
        required_padding_red = data_max * max_abs_velocity / C_KM_S

    summary = {
        "data_wavelength_min_micron": data_min,
        "data_wavelength_max_micron": data_max,
        "data_wavelength_unit_inferred": inferred_unit,
        "model_wavelength_boundaries_micron": [model_min, model_max],
        "blue_padding_micron": float(blue_padding),
        "red_padding_micron": float(red_padding),
        "blue_padding_nm": float(blue_padding * 1000.0),
        "red_padding_nm": float(red_padding * 1000.0),
        "max_abs_berv_kms": float(max_abs_berv),
        "max_abs_velocity_kms": max_abs_velocity,
        "required_blue_padding_micron_approx": required_padding_blue,
        "required_red_padding_micron_approx": required_padding_red,
    }

    logger.info(
        "Selected HRCCS data wavelength range: %.6f-%.6f micron "
        "(input unit inferred as %s)",
        data_min,
        data_max,
        inferred_unit,
    )
    logger.info(
        "pRT model wavelength_boundaries_micron: %.6f-%.6f",
        model_min,
        model_max,
    )
    logger.info(
        "pRT/data wavelength padding: blueward %.6f micron (%.3f nm), "
        "redward %.6f micron (%.3f nm)",
        blue_padding,
        blue_padding * 1000.0,
        red_padding,
        red_padding * 1000.0,
    )
    if max_abs_velocity is not None:
        logger.info(
            "Approx max |velocity| from Kp/residual-velocity priors plus BERV: %.3f km/s; "
            "Doppler padding estimate blue/red: %.6f/%.6f micron",
            max_abs_velocity,
            required_padding_blue,
            required_padding_red,
        )

    warn = blue_padding <= 0.0 or red_padding <= 0.0
    if required_padding_blue is not None and required_padding_red is not None:
        warn = warn or blue_padding < required_padding_blue or red_padding < required_padding_red
    else:
        warn = warn or min(blue_padding, red_padding) < 0.005

    if warn:
        logger.warning(
            "pRT model wavelength boundaries are close to selected data wavelengths; "
            "velocity-dependent edge overlap may bias matched_filter_loglike. "
            "Consider padded model boundaries. Data/order selection is separate "
            "from pRT model-generation boundaries."
        )

    return summary


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
                "sysrem_iteration": int(block.sysrem_iteration),
            }
        )

    return {
        "n_blocks": n_blocks,
        "n_order_blocks": n_orders,
        "valid_fraction": float(n_valid / n_total) if n_total else 0.0,
        "blocks": details,
    }
