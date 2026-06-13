"""Data loading and likelihood evaluation for the pRT emission smoke retrieval."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from .prt_emission_model import (
    MICRON_TO_CM,
    generate_prt_emission_model,
    initialize_prt_atmosphere,
    parameters_from_config,
    prepare_model_like_data,
    require_numpy,
    shifted_model_cube,
    standardize_wavelengths_to_cm,
    wavelength_bounds_for_model,
)


@dataclass
class RetrievalData:
    """Validated high-resolution retrieval arrays.

    Attributes
    ----------
    wavelengths_cm
        Shape ``(n_orders, n_pixels)``.
    flux
        Prepared spectra or residuals, shape ``(n_orders, n_exposures, n_pixels)``.
    uncertainties
        Same shape and units as ``flux``.
    phases
        Orbital phase for each exposure, shape ``(n_exposures,)``.
    good_mask
        Boolean mask with True for usable pixels, same shape as ``flux``.
    barycentric_velocities
        Optional km/s velocities, shape ``(n_exposures,)``.
    """

    wavelengths_cm: Any
    flux: Any
    uncertainties: Any
    phases: Any
    good_mask: Any
    times: Optional[Any] = None
    barycentric_velocities: Optional[Any] = None
    metadata: Optional[dict[str, Any]] = None

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.flux.shape)


def _load_np_array(path: Union[str, Path], key: Optional[str] = None) -> Any:
    np = require_numpy()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input array path does not exist: {path}")

    if path.suffix == ".npz":
        if key is None:
            raise ValueError(f"An npz key is required for {path}.")
        with np.load(path, allow_pickle=False) as data:
            if key not in data:
                raise KeyError(f"Key {key!r} not found in {path}. Available: {list(data.keys())}")
            return data[key]

    if key is not None:
        raise ValueError(f"A key was supplied for non-npz file {path}.")
    return np.load(path, allow_pickle=False)


def _path_and_key(paths: Mapping[str, Any], keys: Mapping[str, Any], name: str) -> Tuple[Optional[str], Optional[str]]:
    value = paths.get(name)
    key = keys.get(name)
    if value is None:
        return None, None

    if isinstance(value, str) and "::" in value:
        path, embedded_key = value.split("::", 1)
        if key is not None and key != embedded_key:
            raise ValueError(
                f"Conflicting keys for {name}: {embedded_key!r} in path and {key!r} in config."
            )
        return path, embedded_key

    return str(value), None if key in {"", None} else str(key)


def _load_named_array(
    paths: Mapping[str, Any],
    keys: Mapping[str, Any],
    name: str,
    required: bool,
) -> Optional[Any]:
    path, key = _path_and_key(paths, keys, name)
    if path is None:
        if required:
            raise ValueError(f"Missing data.paths.{name} in retrieval config.")
        return None
    return _load_np_array(path, key)


def _collapse_wavelengths(wavelengths: Any, flux_shape: Tuple[int, int, int], config: Mapping[str, Any]) -> Any:
    np = require_numpy()
    wavelengths = np.asarray(wavelengths, dtype=float)

    if wavelengths.ndim == 2:
        if wavelengths.shape != (flux_shape[0], flux_shape[2]):
            raise ValueError(
                "2D wavelengths must have shape (n_orders, n_pixels). "
                f"Got {wavelengths.shape}, expected {(flux_shape[0], flux_shape[2])}."
            )
        return wavelengths

    if wavelengths.ndim == 3:
        if wavelengths.shape != flux_shape:
            raise ValueError(
                "3D wavelengths must match flux shape exactly. "
                f"Got {wavelengths.shape}, expected {flux_shape}."
            )

        data_cfg = config.get("data", {})
        mode = str(data_cfg.get("time_dependent_wavelengths", "require_static"))
        reference = wavelengths[:, 0, :]
        diff = np.nanmax(np.abs(wavelengths - reference[:, None, :]))
        scale = np.nanmax(np.abs(reference))
        rel = diff / scale if scale > 0 else diff
        tol = float(data_cfg.get("wavelength_static_rtol", 1.0e-8))

        if mode == "require_static" and rel > tol:
            raise ValueError(
                "Wavelengths are time-dependent beyond data.wavelength_static_rtol. "
                "Provide a 2D common wavelength grid or set "
                "data.time_dependent_wavelengths=use_first_exposure after confirming "
                "that is scientifically intended."
            )
        if mode not in {"require_static", "use_first_exposure"}:
            raise ValueError(
                "data.time_dependent_wavelengths must be require_static or use_first_exposure."
            )
        return reference

    raise ValueError(
        "Wavelengths must have shape (n_orders, n_pixels), or a 3D array matching "
        "flux if data.time_dependent_wavelengths is configured."
    )


def _select_orders_pixels(data: RetrievalData, config: Mapping[str, Any]) -> RetrievalData:
    np = require_numpy()
    select_cfg = config.get("selection", {})

    orders = select_cfg.get("orders", None)
    if orders is not None:
        orders = np.asarray(orders, dtype=int)
        if orders.ndim != 1 or orders.size == 0:
            raise ValueError("selection.orders must be a non-empty 1D list.")
    else:
        orders = slice(None)

    pixel_start = select_cfg.get("pixel_start", None)
    pixel_stop = select_cfg.get("pixel_stop", None)
    pixel_slice = slice(
        None if pixel_start is None else int(pixel_start),
        None if pixel_stop is None else int(pixel_stop),
    )

    wave = data.wavelengths_cm[orders, pixel_slice]
    flux = data.flux[orders, :, pixel_slice]
    unc = data.uncertainties[orders, :, pixel_slice]
    mask = data.good_mask[orders, :, pixel_slice]

    wave_min = select_cfg.get("wavelength_min_micron", None)
    wave_max = select_cfg.get("wavelength_max_micron", None)
    if wave_min is not None or wave_max is not None:
        lo = -float("inf") if wave_min is None else float(wave_min) * MICRON_TO_CM
        hi = float("inf") if wave_max is None else float(wave_max) * MICRON_TO_CM
        wave_mask_2d = (wave >= lo) & (wave <= hi)
        mask = mask & wave_mask_2d[:, None, :]

        keep_pixels = np.any(wave_mask_2d, axis=0)
        if not keep_pixels.any():
            raise ValueError("Wavelength selection removed all pixels.")
        wave = wave[:, keep_pixels]
        flux = flux[:, :, keep_pixels]
        unc = unc[:, :, keep_pixels]
        mask = mask[:, :, keep_pixels]

    return replace(data, wavelengths_cm=wave, flux=flux, uncertainties=unc, good_mask=mask)


def _broadcast_mask(mask: Optional[Any], flux_shape: Tuple[int, int, int], config: Mapping[str, Any]) -> Any:
    np = require_numpy()
    if mask is None:
        return np.ones(flux_shape, dtype=bool)

    mask_array = np.asarray(mask)
    try:
        mask_array = np.broadcast_to(mask_array, flux_shape)
    except ValueError as exc:
        raise ValueError(
            f"Mask shape {mask_array.shape} is not broadcastable to flux shape {flux_shape}."
        ) from exc

    mask_cfg = config.get("data", {}).get("mask", {})
    true_means = str(mask_cfg.get("true_means", "bad")).lower()
    if true_means == "bad":
        return ~mask_array.astype(bool)
    if true_means == "good":
        return mask_array.astype(bool)
    raise ValueError("data.mask.true_means must be either 'bad' or 'good'.")


def _validate_data_arrays(data: RetrievalData) -> None:
    np = require_numpy()

    if data.flux.ndim != 3:
        raise ValueError(f"flux/residuals must be 3D; got shape {data.flux.shape}.")
    if data.uncertainties.shape != data.flux.shape:
        raise ValueError(
            "uncertainties must have the same shape as flux/residuals. "
            f"Got {data.uncertainties.shape} vs {data.flux.shape}."
        )
    if data.wavelengths_cm.shape != (data.flux.shape[0], data.flux.shape[2]):
        raise ValueError(
            "wavelengths must have shape (n_orders, n_pixels) after standardization. "
            f"Got {data.wavelengths_cm.shape}, expected {(data.flux.shape[0], data.flux.shape[2])}."
        )
    if data.phases.shape != (data.flux.shape[1],):
        raise ValueError(
            f"phases must have shape (n_exposures,), got {data.phases.shape} "
            f"for n_exposures={data.flux.shape[1]}."
        )
    if data.good_mask.shape != data.flux.shape:
        raise ValueError("good_mask must have the same shape as flux.")
    if data.barycentric_velocities is not None and data.barycentric_velocities.shape != data.phases.shape:
        raise ValueError("barycentric velocities must have the same shape as phases.")

    if not np.isfinite(data.phases).all():
        raise ValueError("phases contain NaN or infinite values.")

    for order_index in range(data.flux.shape[0]):
        if not np.any(data.good_mask[order_index]):
            raise ValueError(f"Order {order_index} is fully masked after validation/selection.")


def load_retrieval_data(config: Mapping[str, Any], logger: Optional[logging.Logger] = None) -> RetrievalData:
    """Load and validate retrieval arrays from ``config['data']``."""

    np = require_numpy()
    data_cfg = config.get("data", {})
    paths = data_cfg.get("paths", {})
    keys = data_cfg.get("keys", {})

    wavelengths = _load_named_array(paths, keys, "wavelengths", required=True)
    flux = np.asarray(_load_named_array(paths, keys, "flux", required=True), dtype=float)
    uncertainties = np.asarray(_load_named_array(paths, keys, "uncertainties", required=True), dtype=float)
    phases = np.asarray(_load_named_array(paths, keys, "phases", required=True), dtype=float)
    times = _load_named_array(paths, keys, "times", required=False)
    bary = _load_named_array(paths, keys, "barycentric_velocities", required=False)
    mask = _load_named_array(paths, keys, "masks", required=False)

    if bool(data_cfg.get("uncertainties_are_variance", False)):
        uncertainties = np.sqrt(uncertainties)

    wavelength_unit = str(data_cfg.get("wavelength_unit", "micron"))
    wavelengths_cm_full = standardize_wavelengths_to_cm(wavelengths, wavelength_unit)
    wavelengths_cm = _collapse_wavelengths(wavelengths_cm_full, flux.shape, config)

    good_mask = _broadcast_mask(mask, flux.shape, config)
    finite_mask = (
        np.isfinite(flux)
        & np.isfinite(uncertainties)
        & (uncertainties > 0)
        & np.isfinite(wavelengths_cm)[:, None, :]
    )
    good_mask = good_mask & finite_mask

    data = RetrievalData(
        wavelengths_cm=wavelengths_cm,
        flux=flux,
        uncertainties=uncertainties,
        phases=phases,
        good_mask=good_mask,
        times=None if times is None else np.asarray(times, dtype=float),
        barycentric_velocities=None if bary is None else np.asarray(bary, dtype=float),
        metadata={"wavelength_unit": wavelength_unit},
    )
    data = _select_orders_pixels(data, config)
    _validate_data_arrays(data)

    if logger is not None:
        n_orders, n_exp, n_pix = data.shape
        logger.info("Loaded retrieval data: %d orders, %d exposures, %d pixels", n_orders, n_exp, n_pix)
        logger.info("Masked pixel fraction: %.4f", 1.0 - float(np.sum(data.good_mask)) / float(data.good_mask.size))

    return data


def compute_log_likelihood(
    data: RetrievalData,
    prepared_model_cube: Any,
    log_model_scale: float = 0.0,
    fit_amplitude_analytically: bool = False,
) -> Tuple[float, float]:
    """Evaluate a Gaussian log likelihood for a prepared model cube.

    Returns ``(log_likelihood, amplitude)``.  If ``fit_amplitude_analytically``
    is True, the amplitude is the weighted least-squares scale factor.
    Otherwise it is ``10**log_model_scale``.
    """

    np = require_numpy()
    model = np.asarray(prepared_model_cube, dtype=float)
    if model.shape != data.flux.shape:
        raise ValueError(f"Model shape {model.shape} does not match data shape {data.flux.shape}.")

    valid = (
        data.good_mask
        & np.isfinite(data.flux)
        & np.isfinite(data.uncertainties)
        & (data.uncertainties > 0)
        & np.isfinite(model)
    )
    if not np.any(valid):
        raise ValueError("No valid pixels remain for likelihood evaluation.")

    d = data.flux[valid]
    m = model[valid]
    sigma = data.uncertainties[valid]
    inv_var = 1.0 / (sigma * sigma)

    if fit_amplitude_analytically:
        den = np.sum(inv_var * m * m)
        if den <= 0 or not np.isfinite(den):
            return -np.inf, np.nan
        amplitude = float(np.sum(inv_var * d * m) / den)
    else:
        amplitude = 10.0 ** float(log_model_scale)

    residual = d - amplitude * m
    loglike = -0.5 * np.sum(residual * residual * inv_var + np.log(2.0 * np.pi * sigma * sigma))
    return float(loglike), float(amplitude)


def run_kp_vsys_grid(
    data: RetrievalData,
    rest_wavelengths_cm: Any,
    rest_flux: Any,
    config: Mapping[str, Any],
    parameters: Optional[Mapping[str, float]] = None,
    logger: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    """Run the Fe-only Kp-Vsys likelihood grid for one fixed pRT spectrum."""

    np = require_numpy()
    parameters = parameters_from_config(config) if parameters is None else dict(parameters)
    grid_cfg = config.get("grid", {})
    instrument_cfg = config.get("instrument", {})

    kp_values = np.arange(
        float(grid_cfg.get("kp_min", 150.0)),
        float(grid_cfg.get("kp_max", 240.0)) + 0.5 * float(grid_cfg.get("kp_step", 2.0)),
        float(grid_cfg.get("kp_step", 2.0)),
    )
    vsys_values = np.arange(
        float(grid_cfg.get("vsys_min", -30.0)),
        float(grid_cfg.get("vsys_max", 30.0)) + 0.5 * float(grid_cfg.get("vsys_step", 1.0)),
        float(grid_cfg.get("vsys_step", 1.0)),
    )

    loglike = np.full((kp_values.size, vsys_values.size), np.nan, dtype=float)
    amplitude = np.full_like(loglike, np.nan)
    fit_amp = bool(grid_cfg.get("fit_amplitude_analytically", True))
    resolving_power = float(instrument_cfg["resolving_power"])
    velocity_cfg = config.get("velocity", {})

    if logger is not None:
        logger.info("Running Kp grid: %s", kp_values)
        logger.info("Running Vsys grid: %s", vsys_values)

    for k_index, kp in enumerate(kp_values):
        if logger is not None:
            logger.info("Grid row %d/%d: Kp=%.3f km/s", k_index + 1, kp_values.size, kp)

        for v_index, vsys in enumerate(vsys_values):
            cube = shifted_model_cube(
                rest_wavelengths_cm=rest_wavelengths_cm,
                rest_flux=rest_flux,
                observed_wavelengths_cm=data.wavelengths_cm,
                phases=data.phases,
                Kp=float(kp),
                Vsys=float(vsys),
                resolving_power=resolving_power,
                barycentric_velocities=data.barycentric_velocities,
                velocity_config=velocity_cfg,
            )
            prepared = prepare_model_like_data(cube, config, data_mask=data.good_mask)
            ll, amp = compute_log_likelihood(
                data,
                prepared,
                log_model_scale=float(parameters.get("log_model_scale", 0.0)),
                fit_amplitude_analytically=fit_amp,
            )
            loglike[k_index, v_index] = ll
            amplitude[k_index, v_index] = amp

    best_flat = int(np.nanargmax(loglike))
    best_k, best_v = np.unravel_index(best_flat, loglike.shape)
    best = {
        "Kp": float(kp_values[best_k]),
        "Vsys": float(vsys_values[best_v]),
        "log_likelihood": float(loglike[best_k, best_v]),
        "amplitude": float(amplitude[best_k, best_v]),
        "kp_index": int(best_k),
        "vsys_index": int(best_v),
    }

    if logger is not None:
        logger.info("Best grid point: %s", best)

    return {
        "Kp_grid": kp_values,
        "Vsys_grid": vsys_values,
        "log_likelihood": loglike,
        "amplitude": amplitude,
        "best": best,
    }


def build_prepared_model_for_parameters(
    data: RetrievalData,
    rest_wavelengths_cm: Any,
    rest_flux: Any,
    config: Mapping[str, Any],
    Kp: float,
    Vsys: float,
) -> Any:
    """Convenience helper used by smoke tests and fake-signal injection."""

    cube = shifted_model_cube(
        rest_wavelengths_cm=rest_wavelengths_cm,
        rest_flux=rest_flux,
        observed_wavelengths_cm=data.wavelengths_cm,
        phases=data.phases,
        Kp=Kp,
        Vsys=Vsys,
        resolving_power=float(config.get("instrument", {})["resolving_power"]),
        barycentric_velocities=data.barycentric_velocities,
        velocity_config=config.get("velocity", {}),
    )
    return prepare_model_like_data(cube, config, data_mask=data.good_mask)


def inject_fake_signal(
    data: RetrievalData,
    prepared_model_cube: Any,
    scale: float,
    noise_only: bool = False,
    seed: int = 12345,
) -> RetrievalData:
    """Return a data copy with an injected model signal.

    If ``noise_only`` is True, the base data are replaced with Gaussian noise
    using the supplied uncertainties before adding the injection.
    """

    np = require_numpy()
    model = np.asarray(prepared_model_cube, dtype=float)
    if model.shape != data.flux.shape:
        raise ValueError("Injected model shape must match data flux shape.")

    if noise_only:
        rng = np.random.default_rng(int(seed))
        base = rng.normal(loc=0.0, scale=data.uncertainties)
    else:
        base = data.flux.copy()

    injected = base + float(scale) * model
    injected = np.where(data.good_mask, injected, np.nan)
    return replace(data, flux=injected, metadata={**(data.metadata or {}), "injected_scale": float(scale)})


def save_grid_results(results: Mapping[str, Any], output_npz: Union[str, Path], output_json: Union[str, Path]) -> None:
    """Save grid arrays and a small JSON summary."""

    np = require_numpy()
    output_npz = Path(output_npz)
    output_json = Path(output_json)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_npz,
        Kp_grid=results["Kp_grid"],
        Vsys_grid=results["Vsys_grid"],
        log_likelihood=results["log_likelihood"],
        amplitude=results["amplitude"],
    )
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump({"best": results["best"]}, handle, indent=2, sort_keys=True)


def max_abs_grid_velocity(config: Mapping[str, Any]) -> float:
    """Estimate the maximum grid velocity for pRT wavelength padding."""

    grid_cfg = config.get("grid", {})
    params = parameters_from_config(config)
    values = [
        abs(float(grid_cfg.get("kp_min", params["Kp"]))),
        abs(float(grid_cfg.get("kp_max", params["Kp"]))),
        abs(float(grid_cfg.get("vsys_min", params["Vsys"]))),
        abs(float(grid_cfg.get("vsys_max", params["Vsys"]))),
    ]
    return sum(values)


def model_wavelength_bounds_for_data(data: RetrievalData, config: Mapping[str, Any]) -> list[float]:
    """Return pRT wavelength boundaries in micron for this data/grid selection."""

    return wavelength_bounds_for_model(
        observed_wavelengths_cm=data.wavelengths_cm,
        max_abs_velocity_kms=max_abs_grid_velocity(config),
        margin_fraction=float(config.get("model", {}).get("wavelength_margin_fraction", 0.01)),
    )


def _sampler_parameter_names(config: Mapping[str, Any]) -> list[str]:
    sampler_cfg = config.get("sampler", {})
    return list(
        sampler_cfg.get(
            "parameters",
            ["Kp", "Vsys", "T_deep", "delta_T_inv", "log10_Fe", "log_model_scale"],
        )
    )


def _prior_bounds(config: Mapping[str, Any], parameter_names: Sequence[str]) -> list[Tuple[float, float]]:
    priors = config.get("priors", {})
    bounds = []
    for name in parameter_names:
        if name not in priors:
            raise ValueError(f"Missing prior bounds for sampler parameter {name!r}.")
        lo, hi = priors[name]
        lo = float(lo)
        hi = float(hi)
        if hi <= lo:
            raise ValueError(f"Prior for {name!r} must have high > low; got {lo}, {hi}.")
        bounds.append((lo, hi))
    return bounds


def run_sampler(
    data: RetrievalData,
    config: Mapping[str, Any],
    wavelength_boundaries_micron: Sequence[float],
    output_dir: Union[str, Path],
    logger: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    """Run a minimal Fe-only dynesty retrieval.

    This is deliberately gated by ``sampler.confirm_grid_validated`` so the
    first production workflow remains: smoke test, Kp-Vsys grid, then sampler.
    """

    np = require_numpy()
    sampler_cfg = config.get("sampler", {})
    if not bool(sampler_cfg.get("confirm_grid_validated", False)):
        raise RuntimeError(
            "Sampler is gated until the Fe Kp-Vsys grid has been inspected. "
            "Set sampler.confirm_grid_validated=true in the YAML, or pass the "
            "CLI flag in run_fe_sampler.py after confirming the grid recovers "
            "the expected/injected Kp and Vsys."
        )

    try:
        import dynesty
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise RuntimeError("dynesty is required for the Fe-only sampler. Install dynesty.") from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parameter_names = _sampler_parameter_names(config)
    bounds = _prior_bounds(config, parameter_names)
    initial = parameters_from_config(config)
    resolving_power = float(config["instrument"]["resolving_power"])
    velocity_cfg = config.get("velocity", {})
    atmosphere = initialize_prt_atmosphere(
        config,
        wavelength_boundaries_micron=wavelength_boundaries_micron,
        logger=logger,
    )

    def prior_transform(unit_cube: Any) -> Any:
        theta = np.empty(len(parameter_names), dtype=float)
        for idx, (lo, hi) in enumerate(bounds):
            theta[idx] = lo + (hi - lo) * unit_cube[idx]
        return theta

    def log_likelihood(theta: Any) -> float:
        parameters = dict(initial)
        for name, value in zip(parameter_names, theta):
            parameters[name] = float(value)

        try:
            rest_wave, rest_flux, _ = generate_prt_emission_model(
                config=config,
                parameters=parameters,
                wavelength_boundaries_micron=wavelength_boundaries_micron,
                atmosphere=atmosphere,
                logger=None,
            )
            cube = shifted_model_cube(
                rest_wavelengths_cm=rest_wave,
                rest_flux=rest_flux,
                observed_wavelengths_cm=data.wavelengths_cm,
                phases=data.phases,
                Kp=parameters["Kp"],
                Vsys=parameters["Vsys"],
                resolving_power=resolving_power,
                barycentric_velocities=data.barycentric_velocities,
                velocity_config=velocity_cfg,
            )
            prepared = prepare_model_like_data(cube, config, data_mask=data.good_mask)
            loglike, _ = compute_log_likelihood(
                data,
                prepared,
                log_model_scale=parameters["log_model_scale"],
                fit_amplitude_analytically=False,
            )
            return loglike
        except Exception:
            return -np.inf

    sampler = dynesty.NestedSampler(
        log_likelihood,
        prior_transform,
        ndim=len(parameter_names),
        nlive=int(sampler_cfg.get("nlive", 100)),
        bound=str(sampler_cfg.get("bound", "multi")),
        sample=str(sampler_cfg.get("sample", "rwalk")),
    )
    sampler.run_nested(
        dlogz=float(sampler_cfg.get("dlogz", 0.5)),
        maxcall=None if sampler_cfg.get("maxcall", None) is None else int(sampler_cfg["maxcall"]),
    )
    results = sampler.results

    samples = np.asarray(results.samples)
    logl = np.asarray(results.logl)
    if hasattr(results, "logwt") and hasattr(results, "logz"):
        weights = np.exp(np.asarray(results.logwt) - float(results.logz[-1]))
        weights = weights / np.sum(weights)
    else:
        weights = np.ones(samples.shape[0]) / samples.shape[0]

    best_index = int(np.nanargmax(logl))
    best_parameters = dict(initial)
    for name, value in zip(parameter_names, samples[best_index]):
        best_parameters[name] = float(value)

    np.savez_compressed(
        output_dir / "fe_only_dynesty_samples.npz",
        samples=samples,
        log_likelihood=logl,
        weights=weights,
        parameter_names=np.asarray(parameter_names, dtype="U64"),
    )

    summary = {
        "parameter_names": parameter_names,
        "best_fit_parameters": best_parameters,
        "best_log_likelihood": float(logl[best_index]),
        "nlive": int(sampler_cfg.get("nlive", 100)),
        "dlogz": float(sampler_cfg.get("dlogz", 0.5)),
    }
    with (output_dir / "fe_only_dynesty_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    try:
        rest_wave, rest_flux, _ = generate_prt_emission_model(
            config=config,
            parameters=best_parameters,
            wavelength_boundaries_micron=wavelength_boundaries_micron,
            atmosphere=atmosphere,
            logger=None,
        )
        best_cube = shifted_model_cube(
            rest_wavelengths_cm=rest_wave,
            rest_flux=rest_flux,
            observed_wavelengths_cm=data.wavelengths_cm,
            phases=data.phases,
            Kp=best_parameters["Kp"],
            Vsys=best_parameters["Vsys"],
            resolving_power=resolving_power,
            barycentric_velocities=data.barycentric_velocities,
            velocity_config=velocity_cfg,
        )
        best_prepared = prepare_model_like_data(best_cube, config, data_mask=data.good_mask)
        np.savez_compressed(
            output_dir / "fe_only_best_fit_model.npz",
            rest_wavelengths_cm=rest_wave,
            rest_flux=rest_flux,
            prepared_model=best_prepared,
            observed_wavelengths_cm=data.wavelengths_cm,
        )
        from retrieval.plotting import save_best_fit_model_plot

        save_best_fit_model_plot(
            data,
            best_prepared,
            amplitude=10.0 ** float(best_parameters["log_model_scale"]),
            filename=output_dir / "fe_only_best_fit_model_order0_exp0.png",
        )
    except Exception as exc:
        if logger is not None:
            logger.warning("Could not save best-fit model diagnostic: %s", exc)

    try:
        import corner

        import matplotlib.pyplot as plt

        fig = corner.corner(samples, weights=weights, labels=parameter_names, show_titles=True)
        fig.savefig(output_dir / "fe_only_corner.png", dpi=250, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        if logger is not None:
            logger.warning("corner is not installed; skipping corner plot.")

    if logger is not None:
        logger.info("Sampler complete. Best parameters: %s", best_parameters)
        logger.info("Saved sampler outputs to %s", output_dir)

    return summary
