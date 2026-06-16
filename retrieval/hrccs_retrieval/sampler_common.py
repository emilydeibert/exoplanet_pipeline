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

from retrieval.prt_emission_model import initialize_prt_atmosphere

from .ccf_likelihood import evaluate_objective
from .model_builder import build_prt_xcorr_template, parameters_with_updates


_SAMPLER_STATE: dict[str, Any] = {}


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
    return 0.0


def log_likelihood_from_state(theta: Any) -> float:
    """Evaluate one HRCCS likelihood using module-level sampler state."""

    import numpy as np

    state = _SAMPLER_STATE
    updates = {name: float(value) for name, value in zip(state["names"], theta)}
    parameters = parameters_with_updates(state["initial"], updates)
    if not state["sample_kp_vsys"]:
        parameters["Kp"] = float(state["fixed_kp"])
        parameters["Vsys"] = float(state["fixed_vsys"])

    try:
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
            Vsys=parameters["Vsys"],
            objective=state["objective"],
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


def parameter_names(sample_kp_vsys: bool) -> list[str]:
    """Return sampled parameter names for the Fe-only HRCCS retrieval."""

    names = ["T_deep", "delta_T_inv", "log10_Fe"]
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

    walkers = np.empty((int(n_walkers), len(names)), dtype=float)
    for walker_idx in range(int(n_walkers)):
        proposal = centers + rng.normal(0.0, sigma)
        valid = np.all((proposal >= lows) & (proposal <= highs) & np.isfinite(proposal))
        attempts = 0
        while not valid and attempts < 1000:
            proposal = centers + rng.normal(0.0, sigma)
            valid = np.all((proposal >= lows) & (proposal <= highs) & np.isfinite(proposal))
            attempts += 1
        if not valid:
            proposal = rng.uniform(lows, highs)
        walkers[walker_idx] = proposal

    return walkers
