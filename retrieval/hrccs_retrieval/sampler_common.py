"""Shared sampler machinery for HRCCS pRT retrieval entry points.

The functions here are intentionally small and top-level so they can be used
by both dynesty and emcee multiprocessing pools.  They do not define a sampler;
they only manage parameter bounds, fixed Kp/Vsys insertion, per-process pRT
atmosphere caching, and the HRCCS likelihood call.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from retrieval.prt_emission_model import (
    derived_temperature_pressure_parameters,
    initialize_prt_atmosphere,
    validate_model_parameters,
)

from .ccf_likelihood import evaluate_objective
from .model_builder import build_prt_xcorr_template, parameters_with_updates


_SAMPLER_STATE: dict[str, Any] = {}
BETA_PARAMETER_NAMES = {"log_beta", "ln_beta"}
ALPHA_PARAMETER_NAMES = {"alpha", "log_alpha", "ln_alpha"}
PER_NIGHT_OFFSETS_MODE = "per_night_offsets"


def init_sampler_worker(state: dict[str, Any]) -> None:
    """Initialize read-only sampler state in the main process or pool workers.

    When ``cache_prt_atmosphere`` is true, each process initializes one pRT
    ``Radtrans`` object and reuses it for later likelihood calls.  The object is
    never pickled; it lives only in the process-local module state.
    """

    global _SAMPLER_STATE
    _SAMPLER_STATE = dict(state)
    _SAMPLER_STATE["atmosphere"] = None

    start = time.perf_counter()
    pid = os.getpid()
    identity = mp.current_process()._identity
    worker_index = int(identity[0]) if identity else 0
    cache_prt = bool(_SAMPLER_STATE.get("cache_prt_atmosphere", False))

    if cache_prt:
        _SAMPLER_STATE["atmosphere"] = initialize_prt_atmosphere(
            _SAMPLER_STATE["retrieval_config"],
            logger=None,
        )

    elapsed = float(time.perf_counter() - start)
    record = {
        "pid": int(pid),
        "worker_index": worker_index,
        "process_name": mp.current_process().name,
        "cache_prt_atmosphere": cache_prt,
        "atmosphere_initialized": _SAMPLER_STATE["atmosphere"] is not None,
        "seconds": elapsed,
    }
    _SAMPLER_STATE["worker_init_record"] = record

    log_path = _SAMPLER_STATE.get("worker_init_log_path")
    if log_path is not None:
        with Path(log_path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    if cache_prt:
        print(
            "HRCCS sampler worker initialized "
            f"pid={pid} index={worker_index} atmosphere_cached=True seconds={elapsed:.2f}",
            flush=True,
        )


def prior_transform_from_state(unit_cube: Any) -> Any:
    """Dynesty prior transform using module-level state for multiprocessing."""

    import numpy as np

    bounds = _SAMPLER_STATE["bounds"]
    theta = np.empty(len(bounds), dtype=float)
    for idx, (lo, hi) in enumerate(bounds):
        theta[idx] = lo + (hi - lo) * unit_cube[idx]
    return theta


def log_prior_from_state(theta: Any) -> float:
    """Uniform log prior for emcee using the configured YAML bounds."""

    import numpy as np

    theta = np.asarray(theta, dtype=float)
    bounds = _SAMPLER_STATE["bounds"]
    if theta.shape[0] != len(bounds):
        return -np.inf
    if not np.all(np.isfinite(theta)):
        return -np.inf
    for value, (lo, hi) in zip(theta, bounds):
        if value < lo or value > hi:
            return -np.inf
    state = _SAMPLER_STATE
    parameters = parameters_from_theta(theta, state)
    try:
        validate_model_parameters(parameters, state["retrieval_config"])
    except Exception:
        return -np.inf
    return 0.0


def parameters_from_theta(theta: Any, state: Mapping[str, Any]) -> dict[str, float]:
    """Build the physical parameter dictionary from sampled theta values."""

    updates = {name: float(value) for name, value in zip(state["names"], theta)}
    parameters = parameters_with_updates(state["initial"], updates)
    parameters.update({key: float(value) for key, value in state.get("fixed_parameters", {}).items()})
    if not state["sample_kp_vsys"] and not uses_per_night_velocity_offsets(state["retrieval_config"]):
        parameters["Kp"] = float(state["fixed_kp"])
        parameters["Vsys"] = float(state["fixed_vsys"])
    return parameters


def parameters_from_sample(
    sample: Any,
    names: Sequence[str],
    initial: Mapping[str, float],
    fixed_parameters: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Build a parameter dictionary from one sampled vector for summaries."""

    updates = {name: float(value) for name, value in zip(names, sample)}
    parameters = parameters_with_updates(initial, updates)
    parameters.update({key: float(value) for key, value in dict(fixed_parameters or {}).items()})
    return parameters


def summarize_derived_parameters(
    samples: Any,
    names: Sequence[str],
    initial: Mapping[str, float],
    fixed_parameters: Mapping[str, float],
    retrieval_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return percentile summaries for derived non-sampled T-P parameters."""

    import numpy as np

    samples = np.asarray(samples, dtype=float)
    if samples.size == 0:
        return {}
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)

    collected: dict[str, list[float]] = {}
    for row in samples:
        try:
            parameters = parameters_from_sample(row, names, initial, fixed_parameters)
            derived = derived_temperature_pressure_parameters(parameters, retrieval_config)
        except Exception:
            continue
        for key, value in derived.items():
            value = float(value)
            if np.isfinite(value):
                collected.setdefault(str(key), []).append(value)

    summaries: dict[str, Any] = {}
    for key, values in collected.items():
        if not values:
            continue
        p16, median, p84 = np.nanpercentile(np.asarray(values, dtype=float), [16.0, 50.0, 84.0])
        summaries[key] = {
            "median": float(median),
            "p16": float(p16),
            "p84": float(p84),
            "n_valid": int(len(values)),
        }
    return summaries


def fixed_parameters_from_config(retrieval_config: Mapping[str, Any]) -> dict[str, float]:
    """Return fixed scalar parameters configured outside the sampled vector."""

    fixed = retrieval_config.get("fixed_parameters", {})
    if fixed is None or fixed == "":
        return {}
    if not isinstance(fixed, Mapping):
        raise ValueError("fixed_parameters must be a YAML mapping when supplied.")
    return {str(key): float(value) for key, value in fixed.items()}


def _default_scale_transform(parameter_name: str) -> str:
    """Return the conventional physical transform for a scale parameter."""

    if parameter_name.startswith("log_"):
        return "pow10"
    if parameter_name.startswith("ln_"):
        return "exp"
    return "identity"


def _canonical_scale_transform(transform: Any, parameter_name: str) -> str:
    """Normalize scale-transform aliases."""

    if transform is None:
        return _default_scale_transform(parameter_name)
    value = str(transform).strip().lower()
    aliases = {
        "10**x": "pow10",
        "base10": "pow10",
        "log10": "pow10",
        "pow10": "pow10",
        "exp": "exp",
        "exponential": "exp",
        "identity": "identity",
        "linear": "identity",
        "none": "identity",
    }
    if value not in aliases:
        raise ValueError(
            f"Unknown scale transform {transform!r}; use pow10, exp, or identity."
        )
    return aliases[value]


def _physical_scale_value(raw_value: float, transform: str) -> float:
    """Transform a sampled/fixed scale parameter into physical units."""

    import math

    raw_value = float(raw_value)
    if transform == "pow10":
        value = 10.0 ** raw_value
    elif transform == "exp":
        value = math.exp(raw_value)
    elif transform == "identity":
        value = raw_value
    else:
        raise ValueError(f"Unknown canonical scale transform {transform!r}.")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Physical scale must be finite and positive; got {value}.")
    return float(value)


def alpha_configuration(
    names: Sequence[str],
    retrieval_config: Mapping[str, Any],
    objective: str,
) -> dict[str, Any]:
    """Resolve optional sampled/fixed alpha model-amplitude scaling."""

    sampled_candidates = sorted(ALPHA_PARAMETER_NAMES.intersection(set(names)))
    fixed_parameters = fixed_parameters_from_config(retrieval_config)
    fixed_candidates = sorted(ALPHA_PARAMETER_NAMES.intersection(set(fixed_parameters)))
    configured = retrieval_config.get("alpha_mode", None)

    if configured is None:
        if len(sampled_candidates) > 1:
            raise ValueError(f"Only one alpha parameter may be sampled; got {sampled_candidates}.")
        if len(fixed_candidates) > 1:
            raise ValueError(f"Only one alpha parameter may be fixed; got {fixed_candidates}.")
        if sampled_candidates and fixed_candidates:
            raise ValueError(
                "Alpha cannot be both sampled and fixed; "
                f"sampled={sampled_candidates}, fixed={fixed_candidates}."
            )
        if sampled_candidates:
            parameter = sampled_candidates[0]
            mode = "sampled"
            raw_value = None
        elif fixed_candidates:
            parameter = fixed_candidates[0]
            mode = "fixed"
            raw_value = float(fixed_parameters[parameter])
        else:
            return {
                "mode": "disabled",
                "parameter": None,
                "transform": None,
                "value": None,
            }
        transform = _canonical_scale_transform(None, parameter)
    else:
        if not isinstance(configured, Mapping):
            raise ValueError("alpha_mode must be a YAML mapping when supplied.")
        mode = str(configured.get("mode", "disabled")).lower()
        if mode == "disabled":
            if sampled_candidates or fixed_candidates:
                raise ValueError(
                    "alpha_mode.mode=disabled conflicts with alpha parameters in "
                    "sampler.sampled_parameters or fixed_parameters."
                )
            return {
                "mode": "disabled",
                "parameter": None,
                "transform": None,
                "value": None,
            }
        if mode not in {"sampled", "fixed"}:
            raise ValueError("alpha_mode.mode must be disabled, sampled, or fixed.")

        parameter = configured.get("parameter", None)
        if parameter is None:
            candidates = sampled_candidates if mode == "sampled" else fixed_candidates
            if len(candidates) != 1:
                raise ValueError(
                    f"alpha_mode.mode={mode} requires alpha_mode.parameter when "
                    f"it cannot be inferred uniquely; candidates={candidates}."
                )
            parameter = candidates[0]
        parameter = str(parameter)
        transform = _canonical_scale_transform(configured.get("transform", None), parameter)

        if mode == "sampled":
            if parameter not in names:
                raise ValueError(
                    f"Sampled alpha parameter {parameter!r} is missing from "
                    "sampler.sampled_parameters."
                )
            if parameter in fixed_parameters:
                raise ValueError(f"Alpha parameter {parameter!r} cannot be both sampled and fixed.")
            raw_value = None
        else:
            if parameter in fixed_parameters:
                raw_value = float(fixed_parameters[parameter])
            elif "value" in configured:
                raw_value = float(configured["value"])
            else:
                raise ValueError(
                    f"Fixed alpha parameter {parameter!r} needs fixed_parameters.{parameter} "
                    "or alpha_mode.value."
                )

    if str(objective) != "matched_filter_loglike":
        raise NotImplementedError(
            "alpha is only implemented for matched_filter_loglike; "
            f"got objective={objective!r}."
        )
    if mode == "fixed":
        _physical_scale_value(float(raw_value), transform)
    return {
        "mode": mode,
        "parameter": parameter,
        "transform": transform,
        "value": raw_value,
    }


def beta_configuration(
    names: Sequence[str],
    retrieval_config: Mapping[str, Any],
    objective: str,
) -> dict[str, Any]:
    """Resolve disabled/fixed/sampled beta mode from YAML and sampled names."""

    sampled = sorted(BETA_PARAMETER_NAMES.intersection(set(names)))
    fixed = fixed_parameters_from_config(retrieval_config)
    fixed_beta = sorted(BETA_PARAMETER_NAMES.intersection(set(fixed)))
    if len(sampled) > 1:
        raise ValueError(f"Only one beta parameter may be sampled; got {sampled}.")
    if len(fixed_beta) > 1:
        raise ValueError(f"Only one beta parameter may be fixed; got {fixed_beta}.")
    if sampled and fixed_beta:
        raise ValueError(
            f"Beta cannot be both sampled and fixed; sampled={sampled}, fixed={fixed_beta}."
        )
    if sampled:
        mode = {
            "mode": "sampled",
            "parameter": sampled[0],
            "transform": _default_scale_transform(sampled[0]),
            "value": None,
        }
    elif fixed_beta:
        mode = {
            "mode": "fixed",
            "parameter": fixed_beta[0],
            "transform": _default_scale_transform(fixed_beta[0]),
            "value": float(fixed[fixed_beta[0]]),
        }
    else:
        mode = {"mode": "disabled", "parameter": None, "transform": None, "value": None}

    if mode["mode"] == "disabled":
        return mode
    if str(objective) != "matched_filter_loglike":
        raise NotImplementedError(
            "beta/log_beta is only implemented for matched_filter_loglike. "
            f"Requested beta mode {mode} with objective={objective!r}."
        )
    return mode


def beta_from_state(parameters: Mapping[str, float], state: Mapping[str, Any]) -> float | None:
    """Return beta according to explicit sampler-state beta mode."""

    beta_cfg = state.get("beta_config", {"mode": "disabled"})
    mode = beta_cfg.get("mode", "disabled")
    if mode == "disabled":
        return None

    name = str(beta_cfg["parameter"])
    raw_value = float(beta_cfg["value"]) if mode == "fixed" else float(parameters[name])
    return _physical_scale_value(raw_value, str(beta_cfg["transform"]))


def alpha_from_state(parameters: Mapping[str, float], state: Mapping[str, Any]) -> float | None:
    """Return physical alpha according to explicit sampler-state alpha mode."""

    alpha_cfg = state.get("alpha_config", {"mode": "disabled"})
    mode = alpha_cfg.get("mode", "disabled")
    if mode == "disabled":
        return None
    name = str(alpha_cfg["parameter"])
    raw_value = float(alpha_cfg["value"]) if mode == "fixed" else float(parameters[name])
    return _physical_scale_value(raw_value, str(alpha_cfg["transform"]))


def beta_mode_label(beta_config: Mapping[str, Any]) -> str:
    """Return a concise beta-mode log label."""

    mode = beta_config.get("mode", "disabled")
    if mode == "disabled":
        return "disabled"
    if mode == "sampled":
        return f"sampled {beta_config.get('parameter')}"
    if mode == "fixed":
        return f"fixed {beta_config.get('parameter')}={float(beta_config.get('value')):.6g}"
    return str(mode)


def alpha_mode_label(alpha_config: Mapping[str, Any]) -> str:
    """Return a concise alpha-mode log label."""

    mode = alpha_config.get("mode", "disabled")
    if mode == "disabled":
        return "disabled"
    transform = alpha_config.get("transform")
    if mode == "sampled":
        return f"sampled {alpha_config.get('parameter')} ({transform})"
    if mode == "fixed":
        physical = _physical_scale_value(
            float(alpha_config.get("value")),
            str(transform),
        )
        return (
            f"fixed {alpha_config.get('parameter')}={float(alpha_config.get('value')):.6g} "
            f"({transform}, alpha={physical:.6g})"
        )
    return str(mode)


def physical_scale_from_parameters(
    parameters: Mapping[str, float],
    scale_config: Mapping[str, Any],
) -> float | None:
    """Return physical alpha/beta from a parameter dictionary and mode config."""

    mode = scale_config.get("mode", "disabled")
    if mode == "disabled":
        return None
    name = str(scale_config["parameter"])
    raw_value = float(scale_config["value"]) if mode == "fixed" else float(parameters[name])
    return _physical_scale_value(raw_value, str(scale_config["transform"]))


def physical_scale_posterior_summary(
    samples: Any,
    names: Sequence[str],
    scale_config: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return posterior percentiles for a physical alpha/beta scale."""

    import numpy as np

    mode = scale_config.get("mode", "disabled")
    if mode == "disabled":
        return None
    parameter = str(scale_config["parameter"])
    transform = str(scale_config["transform"])
    if mode == "fixed":
        physical = _physical_scale_value(float(scale_config["value"]), transform)
        return {
            "parameter": parameter,
            "transform": transform,
            "median": physical,
            "p16": physical,
            "p84": physical,
            "n_valid": 1,
        }

    if parameter not in names:
        raise ValueError(f"Scale parameter {parameter!r} is missing from sampled names.")
    array = np.asarray(samples, dtype=float)
    if array.size == 0:
        return {
            "parameter": parameter,
            "transform": transform,
            "median": None,
            "p16": None,
            "p84": None,
            "n_valid": 0,
        }
    if array.ndim == 1:
        array = array.reshape(1, -1)
    raw = array[:, list(names).index(parameter)]
    finite = np.isfinite(raw)
    raw = raw[finite]
    if transform == "pow10":
        physical = np.power(10.0, raw)
    elif transform == "exp":
        physical = np.exp(raw)
    elif transform == "identity":
        physical = raw
    else:
        raise ValueError(f"Unknown canonical scale transform {transform!r}.")
    physical = physical[np.isfinite(physical) & (physical > 0)]
    if physical.size == 0:
        return {
            "parameter": parameter,
            "transform": transform,
            "median": None,
            "p16": None,
            "p84": None,
            "n_valid": 0,
        }
    p16, median, p84 = np.nanpercentile(physical, [16.0, 50.0, 84.0])
    return {
        "parameter": parameter,
        "transform": transform,
        "median": float(median),
        "p16": float(p16),
        "p84": float(p84),
        "n_valid": int(physical.size),
    }


def validate_beta_configuration(names: Sequence[str], objective: str, retrieval_config: Mapping[str, Any] | None = None) -> None:
    """Backward-compatible beta validation wrapper."""

    beta_configuration(names, retrieval_config or {}, objective)


def velocity_mode(retrieval_config: Mapping[str, Any] | None) -> str:
    """Return the configured HRCCS residual-velocity mode."""

    if retrieval_config is None:
        return "shared_vsys"
    velocity_cfg = retrieval_config.get("velocity", {})
    if velocity_cfg is None:
        return "shared_vsys"
    if not isinstance(velocity_cfg, Mapping):
        raise ValueError("velocity must be a YAML mapping when supplied.")
    return str(velocity_cfg.get("mode", "shared_vsys"))


def uses_per_night_velocity_offsets(retrieval_config: Mapping[str, Any] | None) -> bool:
    """Return true when the retrieval samples independent night offsets."""

    return velocity_mode(retrieval_config) == PER_NIGHT_OFFSETS_MODE


def per_night_velocity_parameter_mapping(retrieval_config: Mapping[str, Any]) -> dict[str, str]:
    """Return the YAML night -> sampled-parameter velocity mapping."""

    if not uses_per_night_velocity_offsets(retrieval_config):
        return {}
    velocity_cfg = retrieval_config.get("velocity", {})
    mapping = velocity_cfg.get("per_night_offsets", {})
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError(
            "velocity.mode=per_night_offsets requires a non-empty "
            "velocity.per_night_offsets mapping."
        )
    parsed = {str(night): str(parameter_name) for night, parameter_name in mapping.items()}
    bad = [night for night, parameter_name in parsed.items() if not parameter_name]
    if bad:
        raise ValueError(f"Empty per-night velocity parameter name for nights: {bad}.")
    return parsed


def validate_velocity_configuration_for_blocks(
    retrieval_config: Mapping[str, Any],
    blocks: Sequence[Any],
    sampled_names: Sequence[str] | None = None,
) -> None:
    """Fail early for missing per-night velocity mappings or sampled priors."""

    if not uses_per_night_velocity_offsets(retrieval_config):
        return
    mapping = per_night_velocity_parameter_mapping(retrieval_config)
    selected_nights = sorted({str(block.night) for block in blocks})
    missing_nights = [night for night in selected_nights if night not in mapping]
    if missing_nights:
        raise ValueError(
            "velocity.mode=per_night_offsets is missing selected nights in "
            f"velocity.per_night_offsets: {missing_nights}. "
            f"Configured nights are {sorted(mapping)}."
        )
    if sampled_names is not None:
        sampled = set(str(name) for name in sampled_names)
        missing_parameters = sorted({mapping[night] for night in selected_nights if mapping[night] not in sampled})
        if missing_parameters:
            raise ValueError(
                "per-night velocity parameters must be sampled in this mode; "
                f"missing from sampler.sampled_parameters: {missing_parameters}."
            )
    if "Vsys" in set(str(name) for name in sampled_names or []):
        raise ValueError(
            "Do not sample a shared Vsys when velocity.mode=per_night_offsets. "
            "Use only Kp plus the configured per-night deltaV_* parameters."
        )


def velocity_offsets_from_parameters(
    parameters: Mapping[str, float],
    retrieval_config: Mapping[str, Any],
    blocks: Sequence[Any] | None = None,
) -> dict[str, float] | None:
    """Return night -> velocity offset values for per-night mode, else ``None``."""

    if not uses_per_night_velocity_offsets(retrieval_config):
        return None
    mapping = per_night_velocity_parameter_mapping(retrieval_config)
    required_nights = sorted({str(block.night) for block in blocks}) if blocks is not None else sorted(mapping)
    offsets: dict[str, float] = {}
    for night in required_nights:
        if night not in mapping:
            raise ValueError(
                f"No velocity.per_night_offsets entry for selected night {night!r}."
            )
        parameter_name = mapping[night]
        if parameter_name not in parameters:
            raise ValueError(
                f"Per-night velocity parameter {parameter_name!r} for night {night!r} "
                "is missing from the current parameter dictionary."
            )
        offsets[night] = float(parameters[parameter_name])
    return offsets


def log_likelihood_from_state(theta: Any) -> float:
    """Evaluate one HRCCS likelihood using module-level sampler state."""

    import numpy as np

    state = _SAMPLER_STATE
    parameters = parameters_from_theta(theta, state)

    try:
        validate_model_parameters(parameters, state["retrieval_config"])
        template = build_prt_xcorr_template(
            state["retrieval_config"],
            state["exopipe_config"],
            parameters,
            atmosphere=state.get("atmosphere"),
            logger=None,
        )
        result = evaluate_objective(
            blocks=state["blocks"],
            F_model=template["F_model"],
            Kp=parameters["Kp"],
            Vsys=float(parameters.get("Vsys", 0.0)),
            objective=state["objective"],
            alpha=alpha_from_state(parameters, state),
            beta=beta_from_state(parameters, state),
            velocity_offsets_by_night=velocity_offsets_from_parameters(
                parameters,
                state["retrieval_config"],
                state["blocks"],
            ),
        )
        return float(result["objective_value"])
    except Exception as exc:
        logger = state.get("logger")
        if logger is not None:
            logger.warning("Sampler model evaluation failed: %s", exc)
        return -np.inf


def log_probability_from_state(theta: Any) -> float:
    """Return ``log_prior(theta) + log_likelihood(theta)`` for emcee."""

    import numpy as np

    log_prior = log_prior_from_state(theta)
    if not np.isfinite(log_prior):
        return -np.inf
    return float(log_prior + log_likelihood_from_state(theta))


def multiprocessing_context() -> Any:
    """Return the multiprocessing context used for parallel sampler calls.

    On Linux/HPC nodes, ``fork`` is the practical default because the loaded
    data arrays can be inherited copy-on-write by worker processes.  If
    ``fork`` is unavailable, fall back to Python's platform default context.
    """

    try:
        return mp.get_context("fork")
    except ValueError:  # pragma: no cover - platform dependent
        return mp.get_context()


def dynesty_call_count(results: Any, fallback: int) -> int:
    """Extract the number of likelihood calls from dynesty results."""

    try:
        import numpy as np

        return int(np.sum(np.asarray(results.ncall)))
    except Exception:
        return int(fallback)


def read_worker_init_records(path: Path) -> list[dict[str, Any]]:
    """Read worker initialization JSONL records if present."""

    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def draw_benchmark_thetas(n_calls: int, bounds: Sequence[tuple[float, float]], seed: int = 12345) -> Any:
    """Draw deterministic representative theta points from the priors."""

    import numpy as np

    rng = np.random.default_rng(seed)
    unit = rng.uniform(size=(int(n_calls), len(bounds)))
    theta = np.empty_like(unit, dtype=float)
    for idx, (lo, hi) in enumerate(bounds):
        theta[:, idx] = lo + (hi - lo) * unit[:, idx]
    return theta


def parameter_names(sample_kp_vsys: bool, retrieval_config: Mapping[str, Any] | None = None) -> list[str]:
    """Return sampled parameter names for HRCCS samplers.

    When ``sampler.sampled_parameters`` is absent, the historical Fe-only
    default is preserved.  When present, the YAML list controls the sampled
    atmospheric/nuisance parameters, while the CLI still decides whether Kp and
    Vsys are sampled or inserted as fixed values.
    """

    configured = None
    if retrieval_config is not None:
        sampler_cfg = retrieval_config.get("sampler", {})
        configured = sampler_cfg.get("sampled_parameters", sampler_cfg.get("hrccs_parameters", None))

    if uses_per_night_velocity_offsets(retrieval_config):
        if configured is None:
            raise ValueError(
                "velocity.mode=per_night_offsets requires explicit "
                "sampler.sampled_parameters including Kp and the deltaV_* parameters."
            )
        names = [str(name) for name in configured]
        if "Vsys" in names:
            raise ValueError(
                "sampler.sampled_parameters must not include Vsys when "
                "velocity.mode=per_night_offsets."
            )
        mapping = per_night_velocity_parameter_mapping(retrieval_config)
        missing = sorted({"Kp", *mapping.values()} - set(names))
        if missing:
            raise ValueError(
                "per-night velocity mode requires these sampled parameters: "
                f"{missing}."
            )
        return names

    if configured is None:
        names = ["T_deep", "delta_T_inv", "log10_Fe"]
        if sample_kp_vsys:
            names = ["Kp", "Vsys"] + names
        return names

    names = [str(name) for name in configured]
    names = [name for name in names if name not in {"Kp", "Vsys"}]
    if sample_kp_vsys:
        names = ["Kp", "Vsys"] + names
    return names


def prior_bounds(retrieval_config: Mapping[str, Any], names: Sequence[str]) -> list[tuple[float, float]]:
    """Return uniform prior bounds from the retrieval YAML."""

    priors = retrieval_config.get("priors", {})
    bounds = []
    for name in names:
        if name not in priors:
            raise ValueError(f"Missing prior bounds for sampled parameter {name!r}.")
        lo, hi = priors[name]
        lo = float(lo)
        hi = float(hi)
        if not hi > lo:
            raise ValueError(f"Prior bounds for {name!r} must satisfy hi > lo; got [{lo}, {hi}].")
        bounds.append((lo, hi))
    return bounds


def initial_center_from_priors(
    initial: Mapping[str, float],
    names: Sequence[str],
    bounds: Sequence[tuple[float, float]],
) -> list[float]:
    """Return initial parameter centers clipped into the prior volume."""

    centers: list[float] = []
    for name, (lo, hi) in zip(names, bounds):
        value = float(initial.get(name, 0.5 * (lo + hi)))
        centers.append(float(min(max(value, lo), hi)))
    return centers


def initialize_walkers(
    initial: Mapping[str, float],
    names: Sequence[str],
    bounds: Sequence[tuple[float, float]],
    n_walkers: int,
    initial_spread: float,
    seed: int | None = None,
    retrieval_config: Mapping[str, Any] | None = None,
    fixed_parameters: Mapping[str, float] | None = None,
) -> Any:
    """Initialize emcee walkers inside the configured prior volume.

    Walkers are drawn around YAML ``initial_parameters`` where available.  The
    scatter is ``initial_spread`` times each prior width, with uniform fallback
    draws for proposals that leave the prior volume.
    """

    import numpy as np

    rng = np.random.default_rng(seed)
    bounds_array = np.asarray(bounds, dtype=float)
    lows = bounds_array[:, 0]
    highs = bounds_array[:, 1]
    widths = highs - lows
    centers = np.asarray(initial_center_from_priors(initial, names, bounds), dtype=float)
    spread = float(initial_spread)
    if spread < 0:
        raise ValueError("--initial-spread must be non-negative.")
    sigma = np.maximum(widths * spread, widths * 1.0e-12)

    def is_valid(proposal: Any) -> bool:
        if not np.all((proposal >= lows) & (proposal <= highs) & np.isfinite(proposal)):
            return False
        if retrieval_config is None:
            return True
        trial = parameters_with_updates(initial, {name: float(value) for name, value in zip(names, proposal)})
        trial.update({key: float(value) for key, value in dict(fixed_parameters or {}).items()})
        try:
            validate_model_parameters(trial, retrieval_config)
        except Exception:
            return False
        return True

    walkers = np.empty((int(n_walkers), len(names)), dtype=float)
    for walker_idx in range(int(n_walkers)):
        proposal = centers + rng.normal(0.0, sigma)
        attempts = 0
        while not is_valid(proposal) and attempts < 1000:
            proposal = centers + rng.normal(0.0, sigma)
            attempts += 1
        attempts = 0
        while not is_valid(proposal) and attempts < 10000:
            proposal = rng.uniform(lows, highs)
            attempts += 1
        if not is_valid(proposal):
            raise RuntimeError(
                "Could not initialize all emcee walkers inside the prior volume "
                "and derived T-P constraints. Check initial_parameters, priors, "
                "min_delta_T, and min_delta_logP."
            )
        walkers[walker_idx] = proposal

    return walkers
