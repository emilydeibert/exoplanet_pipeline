"""petitRADTRANS emission model utilities for high-resolution retrieval tests.

Units are intentionally explicit:

* pRT wavelengths are handled in cm.
* User-facing wavelengths in config files are in micron unless stated otherwise.
* pRT pressures are supplied in bar at initialization.
* Velocities in the public wrapper are km/s; pRT internals remain cgs.

The first-pass retrieval uses direct ``Radtrans.calculate_flux`` rather than
``SpectralModel.with_velocity_range``.  The SpectralModel workflow is the
reference architecture for high-resolution pRT retrievals, but this project
starts from already prepared/SYSREM-cleaned emission residuals and a custom
two-point inversion profile.  Generating one rest-frame emission model and
doing the Doppler shifting/rebinning here keeps the smoke test auditable.
"""

from __future__ import annotations

import importlib.metadata
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

C_KM_S = 299792.458
MICRON_TO_CM = 1.0e-4
NM_TO_CM = 1.0e-7
ANGSTROM_TO_CM = 1.0e-8
R_JUP_CM = 7.1492e9
SUPPORTED_GAS_CONTINUUM_CONTRIBUTORS = {"H2-H2", "H2-He", "H-"}
GAS_CONTINUUM_ALIASES = {
    "H2--H2": "H2-H2",
    "H2--He": "H2-He",
}
CONTINUUM_OPACITY_SEARCH_TOKENS = {
    "H2-H2": ["H2-H2", "H2--H2", "H2H2"],
    "H2-He": ["H2-He", "H2--He", "H2He"],
}


def _continuum_generic_name(value: str) -> Optional[str]:
    """Return the generic CIA/H- contributor family for a YAML or pRT name."""

    value = str(value)
    normalized = GAS_CONTINUUM_ALIASES.get(value, value)
    if normalized in SUPPORTED_GAS_CONTINUUM_CONTRIBUTORS:
        return normalized
    if value.startswith("H2--H2") or value.startswith("H2-H2"):
        return "H2-H2"
    if value.startswith("H2--He") or value.startswith("H2-He"):
        return "H2-He"
    if value == "H-":
        return "H-"
    return None


@dataclass
class ConvolvedModel:
    """Instrumentally convolved rest-frame model cached for grid evaluation."""

    wavelengths_cm: Any
    flux: Any
    resolving_power: float


def require_numpy():
    """Import numpy lazily so syntax checks work in minimal environments."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise RuntimeError(
            "NumPy is required for the retrieval module. Install the science "
            "environment before running this script."
        ) from exc
    return np


def load_yaml_config(path: Union[str, Path]) -> dict[str, Any]:
    """Load a YAML config file and return a mutable dictionary."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise RuntimeError(
            "PyYAML is required to read retrieval config files. Install pyyaml."
        ) from exc

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping at top level.")

    config.setdefault("_config_path", str(path))
    return config


def setup_logging(output_dir: Union[str, Path], log_name: str = "retrieval.log") -> logging.Logger:
    """Create a file and stream logger for a retrieval run."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("retrieval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(output_dir / log_name)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_prt_version() -> str:
    """Return the installed petitRADTRANS version, or a clear placeholder."""

    try:
        return importlib.metadata.version("petitRADTRANS")
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def configured_prt_input_data_path(config: Mapping[str, Any]) -> Optional[Path]:
    """Return a configured pRT input_data path from YAML or environment.

    The YAML key ``prt.input_data_path`` takes precedence.  Otherwise the code
    checks ``prt.input_data_env_var`` and then the common environment variables
    ``PETITRADTRANS_INPUT_DATA`` and ``PRT_INPUT_DATA_PATH``.
    """

    prt_cfg = config.get("prt", {})
    path_value = prt_cfg.get("input_data_path", None)
    if path_value in {"", None}:
        env_names = [
            str(prt_cfg.get("input_data_env_var", "PETITRADTRANS_INPUT_DATA")),
            "PETITRADTRANS_INPUT_DATA",
            "PRT_INPUT_DATA_PATH",
        ]
        for env_name in env_names:
            value = os.environ.get(env_name)
            if value:
                path_value = value
                break

    if path_value in {"", None}:
        return None
    return Path(str(path_value)).expanduser()


def configure_prt_input_data_path(config: Mapping[str, Any], logger: Optional[logging.Logger] = None) -> Optional[Path]:
    """Set pRT's input_data path using pRT's config parser when configured."""

    path = configured_prt_input_data_path(config)
    prt_cfg = config.get("prt", {})
    require_path = bool(prt_cfg.get("require_input_data_path", False))

    if path is None:
        if require_path:
            raise RuntimeError(
                "No pRT input_data path was configured. Set prt.input_data_path "
                "in the YAML file or export PETITRADTRANS_INPUT_DATA."
            )
        return None

    if path.name != "input_data":
        message = (
            "pRT expects the opacity/data directory itself to be named input_data. "
            f"Configured path is {path}."
        )
        if require_path:
            raise RuntimeError(message)
        if logger is not None:
            logger.warning(message)

    if not path.exists():
        raise RuntimeError(
            f"Configured pRT input_data path does not exist: {path}. "
            "Create it or point PETITRADTRANS_INPUT_DATA to the correct folder."
        )

    try:
        from petitRADTRANS.config import petitradtrans_config_parser
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise RuntimeError("petitRADTRANS is not installed, so the pRT input_data path cannot be configured.") from exc

    petitradtrans_config_parser.set_input_data_path(str(path))
    if logger is not None:
        logger.info("Configured pRT input_data path: %s", path)
    return path


def current_prt_input_data_path() -> Optional[str]:
    """Return pRT's current input_data path if pRT is importable."""

    try:
        from petitRADTRANS.config import petitradtrans_config_parser
    except ImportError:
        return None
    try:
        return str(petitradtrans_config_parser.get_input_data_path())
    except Exception:
        return None


def standardize_wavelengths_to_cm(wavelengths: Any, unit: str) -> Any:
    """Convert a wavelength array to cm.

    Parameters
    ----------
    wavelengths
        Numeric array in the unit given by ``unit``.
    unit
        One of ``cm``, ``micron``/``um``, ``nm``, or ``angstrom``.
    """

    np = require_numpy()
    wave = np.asarray(wavelengths, dtype=float)
    unit_norm = unit.strip().lower()

    if unit_norm in {"cm", "centimeter", "centimeters"}:
        factor = 1.0
    elif unit_norm in {"micron", "microns", "um", "micrometer", "micrometers"}:
        factor = MICRON_TO_CM
    elif unit_norm in {"nm", "nanometer", "nanometers"}:
        factor = NM_TO_CM
    elif unit_norm in {"angstrom", "angstroms", "aa"}:
        factor = ANGSTROM_TO_CM
    else:
        raise ValueError(
            "Unknown wavelength_unit. Use one of: cm, micron, nm, angstrom. "
            f"Got {unit!r}."
        )

    wave_cm = wave * factor
    finite = np.isfinite(wave_cm)
    if not finite.any():
        raise ValueError("Wavelength array has no finite values after unit conversion.")
    if np.nanmin(wave_cm) <= 0:
        raise ValueError("Wavelengths must be positive after conversion to cm.")

    return wave_cm


def wavelengths_cm_to_micron(wavelengths_cm: Any) -> Any:
    """Convert cm wavelengths to micron for plots and logs."""

    np = require_numpy()
    return np.asarray(wavelengths_cm, dtype=float) / MICRON_TO_CM


def build_pressure_grid(config: Mapping[str, Any]) -> Any:
    """Build the pRT pressure grid in bar.

    Defaults match the requested first-pass model: logspace from 1e-6 to
    1e2 bar with 100 layers.
    """

    np = require_numpy()
    grid = config.get("pressure_grid", {})
    p_min = float(grid.get("min_bar", 1.0e-6))
    p_max = float(grid.get("max_bar", 1.0e2))
    n_layers = int(grid.get("n_layers", 100))

    if p_min <= 0 or p_max <= 0 or p_min >= p_max:
        raise ValueError(
            "Pressure grid must satisfy 0 < min_bar < max_bar; "
            f"got min_bar={p_min}, max_bar={p_max}."
        )
    if n_layers < 3:
        raise ValueError("Pressure grid must have at least 3 layers.")

    return np.logspace(math.log10(p_min), math.log10(p_max), n_layers)


def pressure_grid_log10_bounds(config: Mapping[str, Any]) -> tuple[float, float]:
    """Return the actual configured pRT pressure-grid bounds in log10(bar)."""

    np = require_numpy()
    pressures_bar = np.asarray(build_pressure_grid(config), dtype=float)
    finite = np.isfinite(pressures_bar) & (pressures_bar > 0)
    if not finite.any():
        raise ValueError("Configured pressure grid has no finite positive pressures.")
    log_pressures = np.log10(pressures_bar[finite])
    return float(np.nanmin(log_pressures)), float(np.nanmax(log_pressures))


def _validate_log_pressure_inside_grid(
    name: str,
    log_pressure_bar: float,
    grid_log_min: float,
    grid_log_max: float,
) -> None:
    """Raise when a log10(bar) pressure point falls outside the pRT grid."""

    value = float(log_pressure_bar)
    tol = 1.0e-12
    if value < float(grid_log_min) - tol or value > float(grid_log_max) + tol:
        raise ValueError(
            f"{name}={value:.6g} is outside the configured pRT pressure grid "
            f"[{float(grid_log_min):.6g}, {float(grid_log_max):.6g}] log10(bar). "
            "Adjust the prior/fixed value or extend pressure_grid explicitly."
        )


def two_point_inversion_profile(
    pressures_bar: Any,
    T_deep: float,
    delta_T_inv: float,
    P_upper: float = 1.0e-4,
    P_deep: float = 1.0e-1,
    allow_negative_delta_T: bool = False,
) -> Any:
    """Return a two-point thermal inversion profile.

    ``T_upper = T_deep + delta_T_inv``.  The interpolation is linear in
    log10(pressure) between ``P_upper`` and ``P_deep`` and constant outside
    that interval.
    """

    np = require_numpy()
    pressures_bar = np.asarray(pressures_bar, dtype=float)

    if P_upper <= 0 or P_deep <= 0:
        raise ValueError("P_upper and P_deep must be positive bar values.")
    if P_upper >= P_deep:
        raise ValueError(
            f"Expected P_upper < P_deep for an atmosphere; got {P_upper} >= {P_deep}."
        )
    if delta_T_inv < 0 and not allow_negative_delta_T:
        raise ValueError(
            "delta_T_inv is negative but the baseline prior requires delta_T_inv >= 0. "
            "Set priors.allow_negative_delta_T_inv_validation=true for validation runs."
        )

    T_upper = float(T_deep) + float(delta_T_inv)
    log_p = np.log10(pressures_bar)
    log_upper = math.log10(P_upper)
    log_deep = math.log10(P_deep)

    temperatures = np.interp(
        log_p,
        [log_upper, log_deep],
        [T_upper, float(T_deep)],
        left=T_upper,
        right=float(T_deep),
    )
    return temperatures


def free_two_point_inversion_profile(
    pressures_bar: Any,
    T_deep: float,
    delta_T_inv: float,
    logP_deep: float,
    logP_upper: float,
    min_delta_logP: float = 0.25,
    allow_negative_delta_T: bool = False,
    T_upper_bounds: Optional[Sequence[float]] = None,
) -> Any:
    """Return a two-point inversion with both pressure points sampled.

    ``logP_deep`` and ``logP_upper`` are log10 pressures in bar.  Higher
    log-pressure is deeper, so valid samples must satisfy
    ``logP_deep > logP_upper + min_delta_logP``.  The points are not sorted:
    invalid ordering is rejected to preserve parameter meaning.

    The sampled temperatures are ``T_deep`` and ``delta_T_inv`` with derived
    ``T_upper = T_deep + delta_T_inv``.  Temperature is linearly interpolated as
    a function of log10 pressure between the two sampled points.  Outside the
    interval, the profile is held constant at the nearest endpoint.
    """

    np = require_numpy()
    pressures_bar = np.asarray(pressures_bar, dtype=float)
    logP_deep = float(logP_deep)
    logP_upper = float(logP_upper)
    min_delta_logP = float(min_delta_logP)

    if min_delta_logP < 0:
        raise ValueError("min_delta_logP must be non-negative.")
    if not logP_deep > logP_upper + min_delta_logP:
        raise ValueError(
            "Invalid free two-point T-P ordering: require "
            f"logP_deep > logP_upper + min_delta_logP, got "
            f"logP_deep={logP_deep}, logP_upper={logP_upper}, "
            f"min_delta_logP={min_delta_logP}."
        )
    if delta_T_inv < 0 and not allow_negative_delta_T:
        raise ValueError(
            "delta_T_inv is negative but the inversion profile requires "
            "delta_T_inv >= 0."
        )

    T_deep = float(T_deep)
    T_upper = T_deep + float(delta_T_inv)
    if T_upper_bounds is not None:
        lower, upper = [float(value) for value in T_upper_bounds]
        if not lower <= T_upper <= upper:
            raise ValueError(
                f"Derived T_upper={T_upper} is outside configured "
                f"T_upper_bounds=[{lower}, {upper}]."
            )

    log_p = np.log10(pressures_bar)
    return np.interp(
        log_p,
        [logP_upper, logP_deep],
        [T_upper, T_deep],
        left=T_upper,
        right=T_deep,
    )


def free_two_point_inversion_direct_profile(
    pressures_bar: Any,
    T_lower: float,
    T_upper: float,
    logP_lower: float,
    logP_upper: float,
    min_delta_logP: float = 0.25,
    min_delta_T: float = 100.0,
) -> Any:
    """Return a direct two-point inversion profile with sampled endpoint T.

    ``lower`` means deeper atmosphere and therefore higher pressure.  The
    profile is rejected unless ``logP_lower > logP_upper + min_delta_logP`` and
    ``T_upper > T_lower + min_delta_T``.  Points are not sorted internally.
    Temperature is linear in log10 pressure between the two points and constant
    outside the interval.
    """

    np = require_numpy()
    pressures_bar = np.asarray(pressures_bar, dtype=float)
    T_lower = float(T_lower)
    T_upper = float(T_upper)
    logP_lower = float(logP_lower)
    logP_upper = float(logP_upper)
    min_delta_logP = float(min_delta_logP)
    min_delta_T = float(min_delta_T)

    if min_delta_logP < 0:
        raise ValueError("min_delta_logP must be non-negative.")
    if min_delta_T < 0:
        raise ValueError("min_delta_T must be non-negative.")
    if not logP_lower > logP_upper + min_delta_logP:
        raise ValueError(
            "Invalid direct two-point T-P ordering: require "
            f"logP_lower > logP_upper + min_delta_logP, got "
            f"logP_lower={logP_lower}, logP_upper={logP_upper}, "
            f"min_delta_logP={min_delta_logP}."
        )
    if not T_upper > T_lower + min_delta_T:
        raise ValueError(
            "Invalid direct two-point inversion: require "
            f"T_upper > T_lower + min_delta_T, got T_lower={T_lower}, "
            f"T_upper={T_upper}, min_delta_T={min_delta_T}."
        )

    log_p = np.log10(pressures_bar)
    return np.interp(
        log_p,
        [logP_upper, logP_lower],
        [T_upper, T_lower],
        left=T_upper,
        right=T_lower,
    )


def free_two_point_inversion_delta_parameters(
    parameters: Mapping[str, float],
    config: Mapping[str, Any],
) -> dict[str, float]:
    """Validate and derive direct two-point T-P parameters from delta form.

    The Guo-like delta parameterization samples ``T_lower``,
    ``delta_T_inv``, ``logP_upper``, and ``delta_logP``.  This helper derives
    ``T_upper = T_lower + delta_T_inv`` and
    ``logP_lower = logP_upper + delta_logP`` while enforcing that both pressure
    points lie inside the configured pRT pressure grid.
    """

    tp_cfg = config.get("tp_profile", {})
    priors = config.get("priors", {})
    allow_negative_delta_T = bool(priors.get("allow_negative_delta_T_inv_validation", False))

    T_lower = float(parameters["T_lower"])
    delta_T_inv = float(parameters["delta_T_inv"])
    logP_upper = float(parameters["logP_upper"])
    delta_logP = float(parameters["delta_logP"])
    min_delta_T = float(tp_cfg.get("min_delta_T", 0.0))
    min_delta_logP = float(tp_cfg.get("min_delta_logP", 0.0))

    if min_delta_T < 0:
        raise ValueError("min_delta_T must be non-negative.")
    if min_delta_logP < 0:
        raise ValueError("min_delta_logP must be non-negative.")
    if not allow_negative_delta_T and not delta_T_inv > min_delta_T:
        raise ValueError(
            "Invalid delta two-point inversion: require "
            f"delta_T_inv > min_delta_T, got delta_T_inv={delta_T_inv}, "
            f"min_delta_T={min_delta_T}."
        )
    if not delta_logP > min_delta_logP:
        raise ValueError(
            "Invalid delta two-point pressure separation: require "
            f"delta_logP > min_delta_logP, got delta_logP={delta_logP}, "
            f"min_delta_logP={min_delta_logP}."
        )

    T_upper = T_lower + delta_T_inv
    logP_lower = logP_upper + delta_logP
    if not logP_upper < logP_lower:
        raise ValueError(
            "Invalid delta two-point T-P ordering: require "
            f"logP_upper < logP_lower, got logP_upper={logP_upper}, "
            f"logP_lower={logP_lower}."
        )

    T_upper_bounds = tp_cfg.get("T_upper_bounds", None)
    if T_upper_bounds is None and "T_upper_max" in tp_cfg:
        T_upper_bounds = [float("-inf"), float(tp_cfg["T_upper_max"])]
    if T_upper_bounds is not None:
        lower, upper = [float(value) for value in T_upper_bounds]
        if not lower <= T_upper <= upper:
            raise ValueError(
                f"Derived T_upper={T_upper} is outside configured "
                f"T_upper_bounds=[{lower}, {upper}]."
            )

    grid_log_min, grid_log_max = pressure_grid_log10_bounds(config)
    _validate_log_pressure_inside_grid("logP_upper", logP_upper, grid_log_min, grid_log_max)
    _validate_log_pressure_inside_grid("logP_lower", logP_lower, grid_log_min, grid_log_max)

    return {
        "T_lower": T_lower,
        "delta_T_inv": delta_T_inv,
        "T_upper": T_upper,
        "logP_upper": logP_upper,
        "delta_logP": delta_logP,
        "logP_lower": logP_lower,
        "min_delta_T": min_delta_T,
        "min_delta_logP": min_delta_logP,
        "pressure_grid_log10_min": grid_log_min,
        "pressure_grid_log10_max": grid_log_max,
    }


def fixed_pressure_nodes_parameters(
    parameters: Mapping[str, float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate fixed-pressure T-P nodes and return JSON-friendly metadata.

    ``logP_nodes`` are fixed log10 pressures in bar.  The node order may be
    deep-to-high or high-to-deep, but it must be strictly monotonic by default.
    Non-monotonic nodes are only sorted internally when
    ``require_monotonic_logP_nodes`` is explicitly false.
    """

    np = require_numpy()
    tp_cfg = config.get("tp_profile", {})
    if "logP_nodes" not in tp_cfg:
        raise ValueError("tp_profile.logP_nodes is required for fixed_pressure_nodes.")
    if "temperature_parameters" not in tp_cfg:
        raise ValueError("tp_profile.temperature_parameters is required for fixed_pressure_nodes.")

    logP_nodes = [float(value) for value in tp_cfg.get("logP_nodes", [])]
    temperature_parameters = [str(name) for name in tp_cfg.get("temperature_parameters", [])]
    if len(logP_nodes) != len(temperature_parameters):
        raise ValueError(
            "tp_profile.logP_nodes and tp_profile.temperature_parameters must have "
            f"the same length; got {len(logP_nodes)} and {len(temperature_parameters)}."
        )
    if len(logP_nodes) < 2:
        raise ValueError("fixed_pressure_nodes requires at least two pressure nodes.")
    missing = [name for name in temperature_parameters if name not in parameters]
    if missing:
        raise ValueError(f"Missing fixed-pressure-node temperature parameters: {missing}.")

    node_temperatures = [float(parameters[name]) for name in temperature_parameters]
    if not np.all(np.isfinite(np.asarray(logP_nodes, dtype=float))):
        raise ValueError("tp_profile.logP_nodes must all be finite.")
    if not np.all(np.isfinite(np.asarray(node_temperatures, dtype=float))):
        raise ValueError("fixed-pressure-node temperatures must all be finite.")

    require_monotonic = bool(tp_cfg.get("require_monotonic_logP_nodes", True))
    diffs = np.diff(np.asarray(logP_nodes, dtype=float))
    strictly_increasing = bool(np.all(diffs > 0))
    strictly_decreasing = bool(np.all(diffs < 0))
    monotonic_direction = "increasing" if strictly_increasing else "decreasing" if strictly_decreasing else "non_monotonic"
    if require_monotonic and not (strictly_increasing or strictly_decreasing):
        raise ValueError(
            "tp_profile.logP_nodes must be strictly monotonic for fixed_pressure_nodes; "
            f"got {logP_nodes}."
        )

    grid_log_min, grid_log_max = pressure_grid_log10_bounds(config)
    for idx, logP_node in enumerate(logP_nodes):
        _validate_log_pressure_inside_grid(
            f"logP_nodes[{idx}]",
            float(logP_node),
            grid_log_min,
            grid_log_max,
        )

    enforce_temperature_bounds = bool(tp_cfg.get("enforce_temperature_bounds", False))
    if enforce_temperature_bounds:
        priors = config.get("priors", {})
        for name, temperature in zip(temperature_parameters, node_temperatures):
            if name not in priors:
                raise ValueError(
                    "tp_profile.enforce_temperature_bounds=true requires prior bounds "
                    f"for temperature parameter {name!r}."
                )
            lo, hi = [float(value) for value in priors[name]]
            if not lo <= float(temperature) <= hi:
                raise ValueError(
                    f"Temperature parameter {name}={temperature:.6g} is outside "
                    f"configured prior bounds [{lo:.6g}, {hi:.6g}]."
                )

    order = np.argsort(np.asarray(logP_nodes, dtype=float))
    sorted_logP = [float(logP_nodes[int(idx)]) for idx in order]
    sorted_temperatures = [float(node_temperatures[int(idx)]) for idx in order]
    sorted_names = [str(temperature_parameters[int(idx)]) for idx in order]

    deepest_idx = int(np.nanargmax(np.asarray(logP_nodes, dtype=float)))
    highest_idx = int(np.nanargmin(np.asarray(logP_nodes, dtype=float)))
    T_deep = float(node_temperatures[deepest_idx])
    T_high = float(node_temperatures[highest_idx])

    interpolation = str(tp_cfg.get("interpolation", "linear")).lower()
    if interpolation not in {"linear", "pchip"}:
        raise ValueError(
            "tp_profile.interpolation for fixed_pressure_nodes must be 'linear' "
            f"or 'pchip'; got {interpolation!r}."
        )

    return {
        "profile_type": "fixed_pressure_nodes",
        "logP_nodes": logP_nodes,
        "temperature_parameters": temperature_parameters,
        "node_temperatures": {
            name: float(value) for name, value in zip(temperature_parameters, node_temperatures)
        },
        "interpolation": interpolation,
        "require_monotonic_logP_nodes": require_monotonic,
        "monotonic_direction": monotonic_direction,
        "enforce_temperature_bounds": enforce_temperature_bounds,
        "sorted_logP_nodes": sorted_logP,
        "sorted_temperature_parameters": sorted_names,
        "sorted_node_temperatures": sorted_temperatures,
        "T_deep": T_deep,
        "T_high": T_high,
        "T_high_minus_T_deep": float(T_high - T_deep),
        "pressure_grid_log10_min": grid_log_min,
        "pressure_grid_log10_max": grid_log_max,
    }


def fixed_pressure_nodes_profile(
    pressures_bar: Any,
    parameters: Mapping[str, float],
    config: Mapping[str, Any],
) -> Any:
    """Interpolate fixed-pressure node temperatures onto the pRT grid."""

    np = require_numpy()
    pressures_bar = np.asarray(pressures_bar, dtype=float)
    node_info = fixed_pressure_nodes_parameters(parameters, config)
    log_p = np.log10(pressures_bar)
    xp = np.asarray(node_info["sorted_logP_nodes"], dtype=float)
    fp = np.asarray(node_info["sorted_node_temperatures"], dtype=float)

    if node_info["interpolation"] == "linear":
        return np.interp(log_p, xp, fp, left=float(fp[0]), right=float(fp[-1]))

    try:
        from scipy.interpolate import PchipInterpolator
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise RuntimeError(
            "tp_profile.interpolation='pchip' requires scipy. "
            "Use interpolation: linear or install scipy."
        ) from exc
    interpolator = PchipInterpolator(xp, fp, extrapolate=False)
    temperatures = np.asarray(interpolator(log_p), dtype=float)
    temperatures[log_p < xp[0]] = float(fp[0])
    temperatures[log_p > xp[-1]] = float(fp[-1])
    return temperatures


def derived_temperature_pressure_parameters(
    parameters: Mapping[str, float],
    config: Mapping[str, Any],
) -> dict[str, float]:
    """Return derived T-P parameters that are not independently sampled."""

    profile_type = temperature_profile_type(config)
    if profile_type == "free_two_point_inversion_delta":
        derived = free_two_point_inversion_delta_parameters(parameters, config)
        return {
            "T_upper": float(derived["T_upper"]),
            "logP_lower": float(derived["logP_lower"]),
        }
    if profile_type == "fixed_pressure_nodes":
        node_info = fixed_pressure_nodes_parameters(parameters, config)
        return {
            "T_high_minus_T_deep": float(node_info["T_high_minus_T_deep"]),
        }
    return {}


def temperature_pressure_parameter_report(
    parameters: Mapping[str, float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return sampled/derived T-P quantities for logs and JSON summaries."""

    profile_type = temperature_profile_type(config)
    grid_log_min, grid_log_max = pressure_grid_log10_bounds(config)
    report: dict[str, Any] = {
        "profile_type": profile_type,
        "pressure_grid_log10_min": grid_log_min,
        "pressure_grid_log10_max": grid_log_max,
        "sampled": {},
        "derived": {},
    }

    if profile_type == "free_two_point_inversion_delta":
        derived = free_two_point_inversion_delta_parameters(parameters, config)
        report["sampled"] = {
            "T_lower": float(derived["T_lower"]),
            "delta_T_inv": float(derived["delta_T_inv"]),
            "logP_upper": float(derived["logP_upper"]),
            "delta_logP": float(derived["delta_logP"]),
        }
        report["derived"] = {
            "T_upper": float(derived["T_upper"]),
            "logP_lower": float(derived["logP_lower"]),
        }
    elif profile_type == "free_two_point_inversion_direct":
        report["sampled"] = {
            key: float(parameters[key])
            for key in ("T_lower", "T_upper", "logP_lower", "logP_upper")
            if key in parameters
        }
    elif profile_type == "free_two_point_inversion":
        T_deep = float(parameters["T_deep"])
        delta_T_inv = float(parameters["delta_T_inv"])
        report["sampled"] = {
            "T_deep": T_deep,
            "delta_T_inv": delta_T_inv,
            "logP_deep": float(parameters["logP_deep"]),
            "logP_upper": float(parameters["logP_upper"]),
        }
        report["derived"] = {"T_upper": T_deep + delta_T_inv}
    elif profile_type == "fixed_pressure_nodes":
        node_info = fixed_pressure_nodes_parameters(parameters, config)
        report["logP_nodes"] = list(node_info["logP_nodes"])
        report["temperature_node_parameters"] = list(node_info["temperature_parameters"])
        report["node_temperatures"] = dict(node_info["node_temperatures"])
        report["interpolation"] = str(node_info["interpolation"])
        report["require_monotonic_logP_nodes"] = bool(node_info["require_monotonic_logP_nodes"])
        report["monotonic_direction"] = str(node_info["monotonic_direction"])
        report["sampled"] = dict(node_info["node_temperatures"])
        report["derived"] = {
            "T_high_minus_T_deep": float(node_info["T_high_minus_T_deep"]),
        }
    else:
        report["sampled"] = {
            key: float(parameters[key])
            for key in ("T_deep", "delta_T_inv")
            if key in parameters
        }
    return report


def temperature_profile_type(config: Mapping[str, Any]) -> str:
    """Return the configured T-P profile type with backward-compatible default."""

    if "tp_profile" in config:
        tp_cfg = config.get("tp_profile", {})
        return str(tp_cfg.get("profile_type", tp_cfg.get("type", "fixed_two_point_inversion")))
    return str(config.get("temperature_profile", {}).get("type", "fixed_two_point_inversion"))


def temperature_profile_from_parameters(
    pressures_bar: Any,
    parameters: Mapping[str, float],
    config: Mapping[str, Any],
) -> Any:
    """Build the configured temperature profile from sampled parameters."""

    profile_type = temperature_profile_type(config)
    priors = config.get("priors", {})
    allow_negative_delta_T = bool(priors.get("allow_negative_delta_T_inv_validation", False))

    if profile_type in {"fixed_two_point_inversion", "two_point_inversion"}:
        tp_cfg = config.get("temperature_profile", {})
        return two_point_inversion_profile(
            pressures_bar=pressures_bar,
            T_deep=float(parameters["T_deep"]),
            delta_T_inv=float(parameters["delta_T_inv"]),
            P_upper=float(tp_cfg.get("P_upper_bar", 1.0e-4)),
            P_deep=float(tp_cfg.get("P_deep_bar", 1.0e-1)),
            allow_negative_delta_T=allow_negative_delta_T,
        )

    if profile_type == "free_two_point_inversion":
        tp_cfg = config.get("tp_profile", {})
        return free_two_point_inversion_profile(
            pressures_bar=pressures_bar,
            T_deep=float(parameters["T_deep"]),
            delta_T_inv=float(parameters["delta_T_inv"]),
            logP_deep=float(parameters["logP_deep"]),
            logP_upper=float(parameters["logP_upper"]),
            min_delta_logP=float(tp_cfg.get("min_delta_logP", 0.25)),
            allow_negative_delta_T=allow_negative_delta_T,
            T_upper_bounds=tp_cfg.get("T_upper_bounds", None),
        )

    if profile_type == "free_two_point_inversion_direct":
        tp_cfg = config.get("tp_profile", {})
        return free_two_point_inversion_direct_profile(
            pressures_bar=pressures_bar,
            T_lower=float(parameters["T_lower"]),
            T_upper=float(parameters["T_upper"]),
            logP_lower=float(parameters["logP_lower"]),
            logP_upper=float(parameters["logP_upper"]),
            min_delta_logP=float(tp_cfg.get("min_delta_logP", 0.25)),
            min_delta_T=float(tp_cfg.get("min_delta_T", 100.0)),
        )

    if profile_type == "free_two_point_inversion_delta":
        derived = free_two_point_inversion_delta_parameters(parameters, config)
        return free_two_point_inversion_direct_profile(
            pressures_bar=pressures_bar,
            T_lower=derived["T_lower"],
            T_upper=derived["T_upper"],
            logP_lower=derived["logP_lower"],
            logP_upper=derived["logP_upper"],
            min_delta_logP=derived["min_delta_logP"],
            min_delta_T=derived["min_delta_T"],
        )

    if profile_type == "fixed_pressure_nodes":
        return fixed_pressure_nodes_profile(
            pressures_bar=pressures_bar,
            parameters=parameters,
            config=config,
        )

    raise ValueError(
        "Unknown T-P profile type. Use fixed_two_point_inversion, "
        "two_point_inversion, free_two_point_inversion, or "
        "free_two_point_inversion_direct, or "
        "free_two_point_inversion_delta, or fixed_pressure_nodes; "
        f"got {profile_type!r}."
    )


def validate_temperature_profile_parameters(
    parameters: Mapping[str, float],
    config: Mapping[str, Any],
) -> None:
    """Validate T-P parameters without building the full pRT model."""

    # Use the actual configured pRT pressure grid so free-pressure modes cannot
    # quietly propose points outside the model grid.
    temperature_profile_from_parameters(
        pressures_bar=build_pressure_grid(config),
        parameters=parameters,
        config=config,
    )


def validate_model_parameters(parameters: Mapping[str, float], config: Mapping[str, Any]) -> None:
    """Validate lightweight parameter constraints before expensive pRT calls."""

    validate_temperature_profile_parameters(parameters, config)


def _constant_profile(value: float, pressures_bar: Any) -> Any:
    np = require_numpy()
    return float(value) * np.ones_like(pressures_bar, dtype=float)


def _species_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config.get("species", {})


def resolve_prt_species_names(config: Mapping[str, Any]) -> list[str]:
    """Map configured shorthand species to pRT opacity names."""

    names = [entry["prt_name"] for entry in active_species_entries(config)]

    if not names:
        raise ValueError("At least one line species must be requested.")
    return names


def _abundance_key_for_species(species: str) -> str:
    clean = species.replace("+", "plus").replace("-", "_").replace(" ", "_")
    return f"log10_{clean}"


def active_species_entries(config: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return active line species with pRT names and abundance parameters.

    Backward-compatible YAML styles supported:

    * old: ``species.line_species`` plus optional ``species.prt_names``;
    * new: ``species.active_species`` plus per-label mappings such as
      ``species.FeII.prt_name`` and ``species.FeII.abundance_parameter``.
    """

    species_cfg = _species_config(config)
    requested = species_cfg.get("active_species", species_cfg.get("line_species", ["Fe"]))
    prt_mapping = species_cfg.get("prt_names", {})

    entries: list[dict[str, str]] = []
    for species in requested:
        label = str(species)
        per_species = species_cfg.get(label, {})
        if not isinstance(per_species, Mapping):
            per_species = {}
        prt_name = str(per_species.get("prt_name", prt_mapping.get(label, label)))
        abundance_parameter = str(
            per_species.get("abundance_parameter", _abundance_key_for_species(label))
        )
        entries.append(
            {
                "label": label,
                "prt_name": prt_name,
                "abundance_parameter": abundance_parameter,
            }
        )

    if not entries:
        raise ValueError("At least one active line species must be configured.")
    return entries


def build_mass_fractions(
    pressures_bar: Any,
    parameters: Mapping[str, float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build pRT mass fractions for trace species plus H2/He filling gas.

    The retrieved ``log10_X`` parameters are interpreted as log10 mass
    fractions.  H2 and He fill the remaining mass at a fixed ratio.
    """

    np = require_numpy()
    pressures_bar = np.asarray(pressures_bar, dtype=float)
    species_cfg = _species_config(config)
    filling = species_cfg.get("filling_gas", {})

    mass_fractions: dict[str, Any] = {}
    trace_total = np.zeros_like(pressures_bar, dtype=float)

    for entry in active_species_entries(config):
        species = entry["label"]
        parameter_name = entry["abundance_parameter"]
        if parameter_name not in parameters:
            raise ValueError(
                f"Missing abundance parameter {parameter_name!r} for species {species!r}."
            )
        abundance = 10.0 ** float(parameters[parameter_name])
        if abundance < 0 or abundance >= 1:
            raise ValueError(
                f"Mass fraction for {species!r} must be in [0, 1); got {abundance}."
            )

        prt_name = entry["prt_name"]
        profile = abundance * np.ones_like(pressures_bar, dtype=float)
        mass_fractions[prt_name] = profile
        trace_total += profile

    continuum = config.get("continuum", {})
    hminus = continuum.get("hminus", {})
    continuum_contributors = gas_continuum_contributors(config)
    if bool(hminus.get("enabled", False)) or "H-" in continuum_contributors:
        required = {"H-", "H", "e-"}
        fixed = hminus.get("fixed_mass_fractions", {})
        missing = sorted(required.difference(fixed))
        if missing:
            raise ValueError(
                "H- continuum is enabled, so fixed mass fractions for H-, H, "
                f"and e- are required. Missing: {missing}."
            )
        for name in sorted(required):
            profile = _constant_profile(float(fixed[name]), pressures_bar)
            mass_fractions[name] = profile
            trace_total += profile

    if np.nanmax(trace_total) >= 1.0:
        raise ValueError(
            "Trace plus continuum mass fractions exceed unity; lower abundances."
        )

    h2_weight = float(filling.get("H2_weight", 0.74))
    he_weight = float(filling.get("He_weight", 0.24))
    if h2_weight <= 0 or he_weight <= 0:
        raise ValueError("H2_weight and He_weight must both be positive.")

    fill_total = h2_weight + he_weight
    leftover = 1.0 - trace_total
    mass_fractions["H2"] = leftover * h2_weight / fill_total
    mass_fractions["He"] = leftover * he_weight / fill_total

    return mass_fractions


def build_mean_molar_masses(pressures_bar: Any, config: Mapping[str, Any]) -> Any:
    """Return a fixed mean molar mass profile in amu."""

    species_cfg = _species_config(config)
    mean_molar_mass = float(species_cfg.get("mean_molar_mass", 2.33))
    return _constant_profile(mean_molar_mass, pressures_bar)


def _raw_continuum_contributors(config: Mapping[str, Any]) -> Any:
    continuum = config.get("continuum", {})
    if "continuum_contributors" in config:
        return config.get("continuum_contributors", [])
    return continuum.get(
        "continuum_contributors",
        continuum.get("gas_continuum_contributors", []),
    )


def continuum_contributor_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return normalized continuum contributor specs from YAML.

    Accepted YAML entries are either strings, preserving the old behavior, or
    mappings with ``yaml_name``, ``prt_name``, and optional absolute ``file``.
    ``prt_name`` is what will be passed to ``Radtrans``.
    """

    continuum = config.get("continuum", {})
    raw_contributors = _raw_continuum_contributors(config)
    if isinstance(raw_contributors, str):
        raw_items = [raw_contributors]
    else:
        raw_items = list(raw_contributors or [])

    specs: list[dict[str, Any]] = []
    for item in raw_items:
        if isinstance(item, Mapping):
            yaml_name = str(item.get("yaml_name", item.get("name", item.get("contributor", ""))))
            prt_name = str(item.get("prt_name", "") or GAS_CONTINUUM_ALIASES.get(yaml_name, yaml_name))
            file_value = item.get("file", None)
        else:
            yaml_name = str(item)
            prt_name = GAS_CONTINUUM_ALIASES.get(yaml_name, yaml_name)
            file_value = None

        generic = _continuum_generic_name(prt_name) or _continuum_generic_name(yaml_name)
        if generic is None:
            raise ValueError(
                "Unsupported pRT gas continuum contributor requested: "
                f"yaml_name={yaml_name!r}, prt_name={prt_name!r}. Supported "
                "families in this wrapper are H2-H2, H2-He, and H-."
            )
        specs.append(
            {
                "yaml_name": yaml_name,
                "prt_name": prt_name,
                "generic_name": generic,
                "file": str(file_value) if file_value not in {None, ""} else None,
                "explicit": isinstance(item, Mapping),
            }
        )

    contributors = [spec["prt_name"] for spec in specs]
    if bool(continuum.get("hminus", {}).get("enabled", False)) and "H-" not in contributors:
        specs.append(
            {
                "yaml_name": "H-",
                "prt_name": "H-",
                "generic_name": "H-",
                "file": None,
                "explicit": False,
            }
        )

    return specs


def gas_continuum_contributors(config: Mapping[str, Any]) -> list[str]:
    """Return pRT continuum opacity contributors passed to Radtrans."""

    return [spec["prt_name"] for spec in continuum_contributor_specs(config)]


def gas_continuum_contributor_families(config: Mapping[str, Any]) -> list[str]:
    """Return generic continuum contributor families used for mass fractions."""

    return [spec["generic_name"] for spec in continuum_contributor_specs(config)]


def requested_gas_continuum_contributors(config: Mapping[str, Any]) -> list[str]:
    """Return YAML-requested continuum contributor labels before alias mapping."""

    return [spec["yaml_name"] for spec in continuum_contributor_specs(config)]


def _continuum_file_matches_prt_name(path: Path, prt_name: str) -> bool:
    """Return true when an explicit file appears to be the requested opacity."""

    return str(prt_name) in path.name


def _unique_sorted_paths(paths: list[Path]) -> list[Path]:
    return sorted(set(paths))


def find_local_continuum_opacity_files(
    input_data_path: Path,
    contributor: str,
    prt_name: Optional[str] = None,
    max_matches: int = 100,
) -> list[Path]:
    """Best-effort local file search for requested pRT CIA continuum opacities."""

    if prt_name and prt_name not in SUPPORTED_GAS_CONTINUUM_CONTRIBUTORS:
        tokens = [str(prt_name)]
    else:
        tokens = CONTINUUM_OPACITY_SEARCH_TOKENS.get(str(contributor), [])
    if not tokens:
        return []

    roots = [
        input_data_path / "opacities" / "continuum",
        input_data_path / "opacities" / "continuum" / "collision_induced_absorptions",
        input_data_path / "opacities" / "continuum" / "collision_induced_absorption",
    ]
    search_roots = [root for root in roots if root.exists()]
    if not search_roots and (input_data_path / "opacities").exists():
        search_roots = [input_data_path / "opacities"]

    matches: list[Path] = []
    for root in search_roots:
        for token in tokens:
            for path in root.rglob(f"*{token}*"):
                if path.is_file():
                    matches.append(path)
                    if len(matches) >= max_matches:
                        return _unique_sorted_paths(matches)
    return _unique_sorted_paths(matches)


def resolved_continuum_opacity_specs(
    config: Mapping[str, Any],
    input_data_path: Optional[Path],
) -> list[dict[str, Any]]:
    """Resolve explicit/non-interactive local CIA opacity selections."""

    specs = continuum_contributor_specs(config)
    resolved: list[dict[str, Any]] = []
    for spec in specs:
        spec_out = dict(spec)
        generic = spec["generic_name"]
        if generic not in CONTINUUM_OPACITY_SEARCH_TOKENS:
            spec_out["matched_files"] = []
            spec_out["unique"] = True
            resolved.append(spec_out)
            continue

        if input_data_path is None:
            raise RuntimeError(
                "Continuum contributors were requested, but no pRT input_data path "
                "is configured. Set prt.input_data_path or PETITRADTRANS_INPUT_DATA "
                "so local CIA opacities can be checked before pRT initialization."
            )

        explicit_file = spec.get("file")
        if explicit_file is not None:
            path = Path(str(explicit_file)).expanduser()
            if not path.exists():
                raise RuntimeError(
                    f"Explicit CIA opacity file for {spec['yaml_name']} does not exist: {path}"
                )
            if not _continuum_file_matches_prt_name(path, spec["prt_name"]):
                raise RuntimeError(
                    f"Explicit CIA opacity file {path} does not appear to match "
                    f"prt_name={spec['prt_name']!r}. Use a pRT identifier contained "
                    "in the selected file name."
                )
            spec_out["matched_files"] = [str(path)]
            spec_out["unique"] = True
            resolved.append(spec_out)
            continue

        matches = find_local_continuum_opacity_files(
            input_data_path=input_data_path,
            contributor=generic,
            prt_name=spec["prt_name"],
        )
        if not matches:
            raise RuntimeError(
                "Missing local pRT continuum opacity file for contributor "
                f"{spec['yaml_name']} (pRT name {spec['prt_name']}). pRT may "
                "otherwise try to auto-download via Keeper/Selenium. Use a "
                "no-continuum config or install/select an explicit CIA file."
            )
        if len(matches) > 1:
            raise RuntimeError(
                "Ambiguous local pRT continuum opacity files for contributor "
                f"{spec['yaml_name']} (pRT name {spec['prt_name']}): "
                f"{[str(path) for path in matches]}. Specify an explicit "
                "continuum_contributors entry with exact prt_name and file so "
                "pRT cannot prompt interactively in SLURM."
            )

        spec_out["matched_files"] = [str(matches[0])]
        spec_out["unique"] = True
        resolved.append(spec_out)
    return resolved


def preflight_continuum_opacity_files(
    config: Mapping[str, Any],
    contributors: Sequence[str],
    input_data_path: Optional[Path],
    logger: Optional[logging.Logger] = None,
) -> dict[str, list[str]]:
    """Verify local CIA opacity files before pRT can try auto-downloads."""

    if not contributors:
        return {}

    prt_cfg = config.get("prt", {})
    require_local = bool(prt_cfg.get("require_continuum_opacity_files", True))
    if not require_local:
        if logger is not None:
            logger.warning(
                "Skipping local continuum opacity preflight because "
                "prt.require_continuum_opacity_files=false."
            )
        return {}

    resolved = resolved_continuum_opacity_specs(config, input_data_path)
    found: dict[str, list[str]] = {}
    for spec in resolved:
        files = list(spec.get("matched_files", []))
        if files:
            found[spec["prt_name"]] = files
            if logger is not None:
                logger.info(
                    "Continuum opacity selection %s -> %s is non-interactive: %s",
                    spec["yaml_name"],
                    spec["prt_name"],
                    files[0],
                )
    return found


def reference_gravity_cgs(config: Mapping[str, Any]) -> float:
    """Return the configured planetary reference gravity in cm/s2."""

    planet = config.get("planet", {})
    if "reference_gravity_cgs" in planet:
        gravity = float(planet["reference_gravity_cgs"])
    elif "log_g_cgs" in planet:
        gravity = 10.0 ** float(planet["log_g_cgs"])
    else:
        gravity = 4.1e3

    if gravity <= 0:
        raise ValueError(f"reference_gravity_cgs must be positive; got {gravity}.")
    return gravity


def _wavelength_boundaries_micron(config: Mapping[str, Any], override: Optional[Sequence[float]]) -> list[float]:
    if override is not None:
        boundaries = [float(override[0]), float(override[1])]
    else:
        model_cfg = config.get("model", {})
        boundaries = list(model_cfg.get("wavelength_boundaries_micron", [0.383, 1.0]))
        boundaries = [float(boundaries[0]), float(boundaries[1])]

    if boundaries[0] <= 0 or boundaries[1] <= boundaries[0]:
        raise ValueError(
            "wavelength_boundaries_micron must be [positive_min, larger_max]."
        )
    return boundaries


def _line_by_line_sampling(config: Mapping[str, Any]) -> Optional[int]:
    value = config.get("model", {}).get("line_by_line_opacity_sampling", None)
    if value in {None, "none", "None"}:
        return None
    value_int = int(value)
    if value_int < 1:
        raise ValueError("line_by_line_opacity_sampling must be >= 1 when set.")
    return value_int


def initialize_prt_atmosphere(
    config: Mapping[str, Any],
    wavelength_boundaries_micron: Optional[Sequence[float]] = None,
    logger: Optional[logging.Logger] = None,
) -> Any:
    """Initialize a pRT Radtrans object and load requested opacities."""

    try:
        from petitRADTRANS.radtrans import Radtrans
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise RuntimeError(
            "petitRADTRANS is required to initialize Radtrans. Install "
            "petitRADTRANS and run retrieval/check_retrieval_environment.py."
        ) from exc

    input_data_path = configure_prt_input_data_path(config, logger=logger)
    if logger is not None:
        logger.info("pRT active input_data path: %s", current_prt_input_data_path())
    pressures_bar = build_pressure_grid(config)
    line_species = resolve_prt_species_names(config)
    continuum_species = gas_continuum_contributors(config)
    requested_continuum_species = requested_gas_continuum_contributors(config)
    wavelength_bounds = _wavelength_boundaries_micron(config, wavelength_boundaries_micron)
    if logger is not None:
        logger.info("Requested pRT line species for Radtrans: %s", line_species)
        logger.info("YAML-requested pRT continuum contributors: %s", requested_continuum_species)
        logger.info("Requested pRT continuum contributors for Radtrans: %s", continuum_species)
    preflight_continuum_opacity_files(
        config=config,
        contributors=continuum_species,
        input_data_path=input_data_path,
        logger=logger,
    )

    kwargs: dict[str, Any] = {
        "pressures": pressures_bar,
        "line_species": line_species,
        "rayleigh_species": list(config.get("continuum", {}).get("rayleigh_species", [])),
        "gas_continuum_contributors": continuum_species,
        "wavelength_boundaries": wavelength_bounds,
        "line_opacity_mode": str(config.get("model", {}).get("line_opacity_mode", "lbl")),
    }
    sampling = _line_by_line_sampling(config)
    if sampling is not None:
        kwargs["line_by_line_opacity_sampling"] = sampling

    try:
        return Radtrans(**kwargs)
    except EOFError as exc:  # pragma: no cover - pRT/opacity dependent
        raise RuntimeError(
            "pRT attempted interactive default opacity selection while loading "
            f"CIA continuum contributor(s) {continuum_species}. Use explicit "
            "continuum_contributors entries with prt_name/file, or configure "
            "pRT default CIA files interactively before SLURM production."
        ) from exc
    except Exception as exc:  # pragma: no cover - pRT/opacity dependent
        raise RuntimeError(
            "Failed to initialize petitRADTRANS Radtrans and load opacities. "
            f"Requested line_species={line_species}, continuum={continuum_species}, "
            f"wavelength_boundaries_micron={wavelength_bounds}, "
            f"pRT input_data={current_prt_input_data_path()}. "
            "Check that the lbl/CIA opacity files exist locally on the cluster. "
            "Production jobs should not rely on pRT auto-downloads."
        ) from exc


def generate_prt_emission_model(
    config: Mapping[str, Any],
    parameters: Mapping[str, float],
    wavelength_boundaries_micron: Optional[Sequence[float]] = None,
    atmosphere: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Any, Any, dict[str, Any]]:
    """Generate a rest-frame high-resolution pRT emission spectrum.

    Returns
    -------
    wavelength_cm, flux_lambda, metadata
        pRT3 ``calculate_flux`` returns wavelength in cm and F_lambda in cgs
        by default.
    """

    np = require_numpy()

    try:
        import petitRADTRANS
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise RuntimeError(
            "petitRADTRANS is required to generate pRT emission models. "
            "Install petitRADTRANS and make sure the requested lbl opacities "
            "are available locally or downloadable by pRT."
        ) from exc

    pressures_bar = build_pressure_grid(config)

    temperatures = temperature_profile_from_parameters(
        pressures_bar=pressures_bar,
        parameters=parameters,
        config=config,
    )
    mass_fractions = build_mass_fractions(pressures_bar, parameters, config)
    mean_molar_masses = build_mean_molar_masses(pressures_bar, config)
    line_species = resolve_prt_species_names(config)
    continuum_species = gas_continuum_contributors(config)
    requested_continuum_species = requested_gas_continuum_contributors(config)
    wavelength_bounds = _wavelength_boundaries_micron(config, wavelength_boundaries_micron)

    if logger is not None:
        logger.info("pRT version: %s", getattr(petitRADTRANS, "__version__", get_prt_version()))
        logger.info("Requested pRT line species: %s", line_species)
        logger.info("YAML-requested pRT continuum contributors: %s", requested_continuum_species)
        logger.info("Requested pRT continuum contributors: %s", continuum_species)
        logger.info("pRT wavelength boundaries: %.6f-%.6f micron", *wavelength_bounds)
        tp_report = temperature_pressure_parameter_report(parameters, config)
        logger.info(
            "T-P pressure grid log10(bar): %.6g to %.6g",
            tp_report["pressure_grid_log10_min"],
            tp_report["pressure_grid_log10_max"],
        )
        logger.info("T-P sampled parameters: %s", tp_report["sampled"])
        if tp_report["derived"]:
            logger.info("T-P derived parameters: %s", tp_report["derived"])

    if atmosphere is None:
        atmosphere = initialize_prt_atmosphere(
            config,
            wavelength_boundaries_micron=wavelength_boundaries_micron,
            logger=logger,
        )

    try:
        wavelengths_cm, flux_lambda, _ = atmosphere.calculate_flux(
            temperatures=temperatures,
            mass_fractions=mass_fractions,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=reference_gravity_cgs(config),
        )
    except Exception as exc:  # pragma: no cover - pRT dependent
        raise RuntimeError(
            "pRT calculate_flux failed for the current emission model. "
            f"Species={line_species}, mass_fraction_keys={sorted(mass_fractions)}, "
            f"T_deep={parameters.get('T_deep')}, "
            f"delta_T_inv={parameters.get('delta_T_inv')}. "
            "Check pRT opacity coverage, pressure/temperature grid coverage, "
            "and required continuum mass fractions."
        ) from exc

    wavelengths_cm = np.asarray(wavelengths_cm, dtype=float)
    flux_lambda = np.asarray(flux_lambda, dtype=float)
    order = np.argsort(wavelengths_cm)
    wavelengths_cm = wavelengths_cm[order]
    flux_lambda = flux_lambda[order]

    opacity_paths = collect_prt_opacity_file_paths(atmosphere) if logger is not None else []
    if logger is not None:
        if opacity_paths:
            for path in opacity_paths:
                logger.info("pRT opacity file: %s", path)
        else:
            logger.info("pRT opacity file paths were not discoverable from the Radtrans object.")

    metadata = {
        "pressures_bar": pressures_bar,
        "temperatures": temperatures,
        "temperature_profile_type": temperature_profile_type(config),
        "temperature_pressure_report": temperature_pressure_parameter_report(parameters, config),
        "mass_fraction_keys": sorted(mass_fractions),
        "line_species": line_species,
        "gas_continuum_contributors": continuum_species,
        "opacity_paths": opacity_paths,
    }
    return wavelengths_cm, flux_lambda, metadata


def collect_prt_opacity_file_paths(obj: Any, max_depth: int = 4) -> list[str]:
    """Best-effort discovery of loaded pRT opacity file paths."""

    paths: set[str] = set()
    seen: set[int] = set()

    def visit(value: Any, depth: int) -> None:
        if depth > max_depth:
            return
        value_id = id(value)
        if value_id in seen:
            return
        seen.add(value_id)

        if isinstance(value, str):
            if ".petitRADTRANS" in value or value.endswith(".h5"):
                paths.add(value)
            return
        if isinstance(value, Mapping):
            for item in value.values():
                visit(item, depth + 1)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item, depth + 1)
            return
        if hasattr(value, "__dict__"):
            visit(vars(value), depth + 1)

    visit(obj, 0)
    return sorted(paths)


def convolve_to_resolution(
    wavelengths_cm: Any,
    flux: Any,
    resolving_power: float,
) -> Any:
    """Convolve a high-resolution spectrum with a Gaussian LSF.

    The convolution is performed on the native pRT grid assuming approximately
    constant spacing in ln(lambda), which is appropriate for pRT lbl opacities
    sampled at near-constant resolving power.
    """

    np = require_numpy()
    try:
        from scipy.ndimage import gaussian_filter1d
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise RuntimeError("SciPy is required for instrumental convolution.") from exc

    wavelengths_cm = np.asarray(wavelengths_cm, dtype=float)
    flux = np.asarray(flux, dtype=float)
    resolving_power = float(resolving_power)

    if resolving_power <= 0:
        raise ValueError("resolving_power must be positive.")
    if wavelengths_cm.ndim != 1 or flux.ndim != 1 or wavelengths_cm.size != flux.size:
        raise ValueError("wavelengths_cm and flux must be 1D arrays with the same length.")
    if wavelengths_cm.size < 3:
        raise ValueError("Need at least three model pixels to convolve.")

    log_lambda = np.log(wavelengths_cm)
    dlog = np.nanmedian(np.diff(log_lambda))
    if not np.isfinite(dlog) or dlog <= 0:
        raise ValueError("Model wavelength grid must be strictly increasing.")

    fwhm_log_lambda = 1.0 / resolving_power
    sigma_pixels = fwhm_log_lambda / (2.354820045 * dlog)
    if sigma_pixels <= 0:
        return flux.copy()

    return gaussian_filter1d(flux, sigma=sigma_pixels, mode="nearest")


def build_convolved_model(
    wavelengths_cm: Any,
    flux: Any,
    resolving_power: float,
) -> ConvolvedModel:
    """Convolve a rest-frame spectrum once for reuse across Kp/Vsys points."""

    np = require_numpy()
    wavelengths_cm = np.asarray(wavelengths_cm, dtype=float)
    convolved_flux = convolve_to_resolution(wavelengths_cm, flux, resolving_power)
    return ConvolvedModel(
        wavelengths_cm=wavelengths_cm,
        flux=np.asarray(convolved_flux, dtype=float),
        resolving_power=float(resolving_power),
    )


def rebin_spectrum(wavelengths_cm: Any, flux: Any, target_wavelengths_cm: Any) -> Any:
    """Linearly interpolate a model spectrum onto any target wavelength shape."""

    np = require_numpy()
    wave = np.asarray(wavelengths_cm, dtype=float)
    spec = np.asarray(flux, dtype=float)
    target = np.asarray(target_wavelengths_cm, dtype=float)

    if wave.ndim != 1 or spec.ndim != 1:
        raise ValueError("Model wavelength and flux arrays must be 1D.")
    if wave.size != spec.size:
        raise ValueError("Model wavelength and flux arrays must have the same length.")

    return np.interp(target, wave, spec, left=np.nan, right=np.nan)


def convolve_and_rebin_model(
    model_wavelengths_cm: Any,
    model_flux: Any,
    observed_wavelengths_cm: Any,
    resolving_power: float,
) -> Any:
    """Convolve a rest-frame model and interpolate it to observed wavelengths."""

    convolved = convolve_to_resolution(model_wavelengths_cm, model_flux, resolving_power)
    return rebin_spectrum(model_wavelengths_cm, convolved, observed_wavelengths_cm)


def planet_radial_velocity_kms(
    phases: Any,
    Kp: float,
    Vsys: float,
    barycentric_velocities: Optional[Any] = None,
    include_barycentric: bool = False,
    barycentric_sign: float = -1.0,
) -> Any:
    """Compute exposure velocities in km/s for model shifting.

    The orbital convention follows the existing cross-correlation code:
    ``Kp * sin(2*pi*phase)``.  If barycentric velocities are provided, the
    default sign is ``-1`` to match the current ``shift2BERV`` convention.
    """

    np = require_numpy()
    phases = np.asarray(phases, dtype=float)
    velocities = float(Kp) * np.sin(2.0 * np.pi * phases) + float(Vsys)

    if include_barycentric:
        if barycentric_velocities is None:
            raise ValueError("include_barycentric=True but no barycentric velocities were supplied.")
        velocities = velocities + float(barycentric_sign) * np.asarray(
            barycentric_velocities, dtype=float
        )

    return velocities


def shifted_model_cube(
    rest_wavelengths_cm: Any,
    rest_flux: Any,
    observed_wavelengths_cm: Any,
    phases: Any,
    Kp: float,
    Vsys: float,
    resolving_power: float,
    barycentric_velocities: Optional[Any] = None,
    velocity_config: Optional[Mapping[str, Any]] = None,
    convolved_rest_flux: Optional[Any] = None,
) -> Any:
    """Convolve, Doppler shift, and rebin a rest-frame model for all exposures."""

    np = require_numpy()
    rest_wave = np.asarray(rest_wavelengths_cm, dtype=float)
    obs_wave = np.asarray(observed_wavelengths_cm, dtype=float)
    phases = np.asarray(phases, dtype=float)

    if obs_wave.ndim != 2:
        raise ValueError(
            "observed_wavelengths_cm must have shape (n_orders, n_pixels) before "
            "building the shifted model cube."
        )
    if phases.ndim != 1:
        raise ValueError("phases must be a 1D array.")

    velocity_config = velocity_config or {}
    velocities = planet_radial_velocity_kms(
        phases=phases,
        Kp=Kp,
        Vsys=Vsys,
        barycentric_velocities=barycentric_velocities,
        include_barycentric=bool(velocity_config.get("include_barycentric", False)),
        barycentric_sign=float(velocity_config.get("barycentric_sign", -1.0)),
    )

    if convolved_rest_flux is None:
        convolved = convolve_to_resolution(rest_wave, rest_flux, resolving_power)
    else:
        convolved = np.asarray(convolved_rest_flux, dtype=float)
        if convolved.shape != rest_wave.shape:
            raise ValueError(
                "convolved_rest_flux must have the same shape as rest_wavelengths_cm."
            )

    n_orders, n_pixels = obs_wave.shape
    n_exp = phases.size
    cube = np.full((n_orders, n_exp, n_pixels), np.nan, dtype=float)

    dopplers = 1.0 + velocities / C_KM_S
    if np.any(dopplers <= 0):
        bad = velocities[dopplers <= 0]
        raise ValueError(f"Non-physical Doppler factor for velocities {bad} km/s.")

    for order_index in range(n_orders):
        rest_grid_for_observed_pixels = obs_wave[order_index][None, :] / dopplers[:, None]
        cube[order_index] = np.interp(
            rest_grid_for_observed_pixels,
            rest_wave,
            convolved,
            left=np.nan,
            right=np.nan,
        )

    return cube


def _odd_filter_width(width_pixels: int) -> int:
    width = int(width_pixels)
    if width < 3:
        raise ValueError("highpass_width_pixels must be at least 3.")
    if width % 2 == 0:
        width += 1
    return width


def _fill_nonfinite_for_filter(cube: Any) -> Any:
    np = require_numpy()

    cube = np.asarray(cube, dtype=float)
    fill = np.nanmedian(cube, axis=2, keepdims=True)
    return np.where(np.isfinite(cube), cube, fill)


def _median_continuum_scipy_reference(cube: Any, width_pixels: int) -> Any:
    """Reference median continuum: the original scipy.ndimage implementation."""

    np = require_numpy()
    try:
        from scipy.ndimage import median_filter
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise RuntimeError("SciPy is required for median high-pass preparation.") from exc

    width = _odd_filter_width(width_pixels)
    filled = _fill_nonfinite_for_filter(cube)
    return median_filter(filled, size=(1, 1, width), mode="nearest")


def _median_continuum_bottleneck(cube: Any, width_pixels: int) -> Any:
    """Exact centered moving median using bottleneck, with nearest-edge padding."""

    np = require_numpy()
    try:
        import bottleneck as bn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "bottleneck is required for preparation.median_filter_backend='bottleneck'. "
            "Install bottleneck or use median_filter_backend='scipy_reference'."
        ) from exc

    width = _odd_filter_width(width_pixels)
    radius = width // 2
    filled = _fill_nonfinite_for_filter(cube)
    flat = filled.reshape(-1, filled.shape[-1])
    padded = np.pad(flat, ((0, 0), (radius, radius)), mode="edge")
    moved = bn.move_median(padded, window=width, min_count=width, axis=1)
    continuum = moved[:, width - 1 : width - 1 + flat.shape[1]]
    return continuum.reshape(filled.shape)


def _median_continuum(cube: Any, width_pixels: int, backend: str = "auto") -> Any:
    backend = str(backend).lower()

    if backend in {"auto", "bottleneck"}:
        try:
            return _median_continuum_bottleneck(cube, width_pixels)
        except RuntimeError:
            if backend == "bottleneck":
                raise

    if backend in {"auto", "scipy", "scipy_reference", "ndimage"}:
        return _median_continuum_scipy_reference(cube, width_pixels)

    raise ValueError(
        "preparation.median_filter_backend must be auto, bottleneck, or scipy_reference."
    )


def _median_highpass(cube: Any, width_pixels: int, backend: str = "auto") -> Any:
    np = require_numpy()

    cube = np.asarray(cube, dtype=float)
    continuum = _median_continuum(cube, width_pixels, backend=backend)
    continuum = np.where(np.isfinite(continuum) & (continuum != 0), continuum, np.nan)
    return cube / continuum - 1.0


def _gaussian_highpass(cube: Any, width_pixels: int) -> Any:
    np = require_numpy()
    try:
        from scipy.ndimage import gaussian_filter1d
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise RuntimeError("SciPy is required for gaussian high-pass preparation.") from exc

    width = _odd_filter_width(width_pixels)
    sigma = width / 2.354820045
    cube = np.asarray(cube, dtype=float)
    filled = _fill_nonfinite_for_filter(cube)
    continuum = gaussian_filter1d(filled, sigma=sigma, axis=2, mode="nearest")
    continuum = np.where(np.isfinite(continuum) & (continuum != 0), continuum, np.nan)
    return cube / continuum - 1.0


def _uniform_highpass(cube: Any, width_pixels: int) -> Any:
    np = require_numpy()
    try:
        from scipy.ndimage import uniform_filter1d
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise RuntimeError("SciPy is required for uniform high-pass preparation.") from exc

    width = _odd_filter_width(width_pixels)
    cube = np.asarray(cube, dtype=float)
    filled = _fill_nonfinite_for_filter(cube)
    continuum = uniform_filter1d(filled, size=width, axis=2, mode="nearest")
    continuum = np.where(np.isfinite(continuum) & (continuum != 0), continuum, np.nan)
    return cube / continuum - 1.0


def _relative_flux_to_delta_mag(relative: Any) -> Any:
    np = require_numpy()
    flux_ratio = np.clip(1.0 + relative, 1.0e-300, None)
    return -2.5 * np.log10(flux_ratio)


def prepare_model_like_data(
    model_cube: Any,
    config: Mapping[str, Any],
    data_mask: Optional[Any] = None,
) -> Any:
    """Approximate the data preparation on the model cube.

    This is intentionally modest: it high-pass filters along the pixel axis,
    optionally converts to delta magnitudes, removes per-exposure means, and
    optionally standardizes each order/exposure.  It is not a full SYSREM
    transfer-function correction.
    """

    np = require_numpy()
    model_cube = np.asarray(model_cube, dtype=float)
    prep_cfg = config.get("preparation", {})
    method = str(prep_cfg.get("method", "median_highpass_delta_mag"))
    width = int(prep_cfg.get("highpass_width_pixels", 601))
    median_backend = str(prep_cfg.get("median_filter_backend", "auto"))

    if method in {"none", "raw"}:
        prepared = model_cube.copy()
    elif method in {
        "median_highpass",
        "median_highpass_relative_flux",
        "median_highpass_exact",
        "median_highpass_relative_flux_exact",
    }:
        prepared = _median_highpass(model_cube, width, backend=median_backend)
    elif method in {"median_highpass_delta_mag", "median_highpass_delta_mag_exact"}:
        relative = _median_highpass(model_cube, width, backend=median_backend)
        prepared = _relative_flux_to_delta_mag(relative)
    elif method == "median_highpass_delta_mag_scipy_reference":
        relative = _median_highpass(model_cube, width, backend="scipy_reference")
        prepared = _relative_flux_to_delta_mag(relative)
    elif method == "median_highpass_delta_mag_bottleneck":
        relative = _median_highpass(model_cube, width, backend="bottleneck")
        prepared = _relative_flux_to_delta_mag(relative)
    elif method == "gaussian_highpass_delta_mag_fast":
        relative = _gaussian_highpass(model_cube, width)
        prepared = _relative_flux_to_delta_mag(relative)
    elif method == "uniform_highpass_delta_mag_fast":
        relative = _uniform_highpass(model_cube, width)
        prepared = _relative_flux_to_delta_mag(relative)
    else:
        raise ValueError(
            "Unknown preparation.method. Use none, median_highpass_delta_mag_exact, "
            "median_highpass_delta_mag_scipy_reference, "
            "gaussian_highpass_delta_mag_fast, or uniform_highpass_delta_mag_fast."
        )

    mask = np.isfinite(prepared)
    if data_mask is not None:
        mask &= np.asarray(data_mask, dtype=bool)

    if bool(prep_cfg.get("remove_exposure_mean", True)):
        work = np.where(mask, prepared, np.nan)
        mean = np.nanmean(work, axis=2, keepdims=True)
        prepared = prepared - mean

    if bool(prep_cfg.get("standardize_per_exposure", True)):
        work = np.where(mask, prepared, np.nan)
        std = np.nanstd(work, axis=2, keepdims=True)
        prepared = np.where(std > 0, prepared / std, np.nan)

    prepared = np.where(mask, prepared, np.nan)
    return prepared


def benchmark_prepare_model_like_data(
    model_cube: Any,
    config: Mapping[str, Any],
    data_mask: Optional[Any] = None,
    methods: Optional[Sequence[str]] = None,
) -> list[dict[str, Any]]:
    """Time preparation methods and compare them to the SciPy median reference."""

    np = require_numpy()
    methods = list(methods or [
        "median_highpass_delta_mag_scipy_reference",
        str(config.get("preparation", {}).get("method", "median_highpass_delta_mag_exact")),
    ])

    reference_config = dict(config)
    reference_config["preparation"] = dict(config.get("preparation", {}))
    reference_config["preparation"]["method"] = "median_highpass_delta_mag_scipy_reference"

    start = time.perf_counter()
    reference = prepare_model_like_data(model_cube, reference_config, data_mask=data_mask)
    reference_elapsed = time.perf_counter() - start

    results: list[dict[str, Any]] = [
        {
            "method": "median_highpass_delta_mag_scipy_reference",
            "seconds": float(reference_elapsed),
            "max_abs_delta_vs_reference": 0.0,
            "rms_delta_vs_reference": 0.0,
        }
    ]

    for method in methods:
        if method == "median_highpass_delta_mag_scipy_reference":
            continue
        trial_config = dict(config)
        trial_config["preparation"] = dict(config.get("preparation", {}))
        trial_config["preparation"]["method"] = method

        start = time.perf_counter()
        prepared = prepare_model_like_data(model_cube, trial_config, data_mask=data_mask)
        elapsed = time.perf_counter() - start

        delta = prepared - reference
        finite = np.isfinite(delta)
        if finite.any():
            max_abs = float(np.nanmax(np.abs(delta[finite])))
            rms = float(np.sqrt(np.nanmean(delta[finite] * delta[finite])))
        else:
            max_abs = float("nan")
            rms = float("nan")

        results.append(
            {
                "method": str(method),
                "seconds": float(elapsed),
                "max_abs_delta_vs_reference": max_abs,
                "rms_delta_vs_reference": rms,
            }
        )

    return results


def wavelength_bounds_for_model(
    observed_wavelengths_cm: Any,
    max_abs_velocity_kms: float,
    margin_fraction: float = 0.01,
) -> list[float]:
    """Return pRT wavelength boundaries in micron with velocity/margin padding."""

    np = require_numpy()
    obs = np.asarray(observed_wavelengths_cm, dtype=float)
    finite = np.isfinite(obs)
    if not finite.any():
        raise ValueError("No finite observed wavelengths available for pRT boundaries.")

    velocity_pad = 1.0 + abs(float(max_abs_velocity_kms)) / C_KM_S
    lo_cm = np.nanmin(obs) / velocity_pad
    hi_cm = np.nanmax(obs) * velocity_pad
    span = hi_cm - lo_cm
    lo_cm -= float(margin_fraction) * span
    hi_cm += float(margin_fraction) * span

    return [lo_cm / MICRON_TO_CM, hi_cm / MICRON_TO_CM]


def parameters_from_config(config: Mapping[str, Any]) -> dict[str, float]:
    """Return fixed initial parameters for the smoke test/grid model."""

    params = dict(config.get("initial_parameters", {}))
    velocity_cfg = config.get("velocity", {})
    velocity_mode = str(velocity_cfg.get("mode", "shared_vsys")) if isinstance(velocity_cfg, Mapping) else "shared_vsys"
    required = ["Kp", "log_model_scale"]
    if velocity_mode == "per_night_offsets":
        mapping = velocity_cfg.get("per_night_offsets", {}) if isinstance(velocity_cfg, Mapping) else {}
        if not isinstance(mapping, Mapping) or not mapping:
            raise ValueError(
                "velocity.mode=per_night_offsets requires velocity.per_night_offsets "
                "in the retrieval config."
            )
        required.extend(str(parameter_name) for parameter_name in mapping.values())
    else:
        required.append("Vsys")
    profile_type = temperature_profile_type(config)
    if profile_type in {"fixed_two_point_inversion", "two_point_inversion", "free_two_point_inversion"}:
        required.extend(["T_deep", "delta_T_inv"])
    if profile_type == "free_two_point_inversion":
        required.extend(["logP_deep", "logP_upper"])
    elif profile_type == "free_two_point_inversion_direct":
        required.extend(["T_lower", "T_upper", "logP_lower", "logP_upper"])
    elif profile_type == "free_two_point_inversion_delta":
        required.extend(["T_lower", "delta_T_inv", "logP_upper", "delta_logP"])
    elif profile_type == "fixed_pressure_nodes":
        tp_cfg = config.get("tp_profile", {})
        temperature_parameters = tp_cfg.get("temperature_parameters", None)
        if temperature_parameters is None:
            raise ValueError("tp_profile.temperature_parameters is required for fixed_pressure_nodes.")
        required.extend(str(name) for name in temperature_parameters)
    required.extend(entry["abundance_parameter"] for entry in active_species_entries(config))
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError(f"Missing initial_parameters entries: {missing}.")
    return {key: float(value) for key, value in params.items()}


def log_run_summary(
    logger: logging.Logger,
    config: Mapping[str, Any],
    parameters: Mapping[str, float],
    wavelengths_cm: Optional[Any] = None,
    mask: Optional[Any] = None,
) -> None:
    """Log core configuration and data dimensions."""

    np = require_numpy()
    logger.info("Target: %s", config.get("target", {}).get("name", "unknown"))
    logger.info("Parameters: %s", dict(parameters))
    logger.info("pRT package version: %s", get_prt_version())
    logger.info("Configured line species: %s", _species_config(config).get("line_species", ["Fe"]))
    logger.info("Line opacity mode: %s", config.get("model", {}).get("line_opacity_mode", "lbl"))
    logger.info("Resolving power: %s", config.get("instrument", {}).get("resolving_power"))

    if wavelengths_cm is not None:
        wave_um = wavelengths_cm_to_micron(wavelengths_cm)
        logger.info(
            "Observed wavelength range: %.6f-%.6f micron",
            float(np.nanmin(wave_um)),
            float(np.nanmax(wave_um)),
        )

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        logger.info("Valid data pixels: %d / %d", int(np.sum(mask)), int(mask.size))
        logger.info("Masked pixel fraction: %.4f", 1.0 - float(np.sum(mask)) / float(mask.size))
