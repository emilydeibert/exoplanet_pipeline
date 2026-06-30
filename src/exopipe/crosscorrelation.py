from astropy import constants as cs 
from astropy import units as u
from scipy import interpolate
import numpy as np 
import batman
import sys

from exopipe import tools

def _to_float(x, unit=None):
    """Return float from either plain number or astropy Quantity."""
    if hasattr(x, "to_value"):
        return float(x.to_value(unit)) if unit is not None else float(x.value)
    return float(x)


def batman_transit_weights_from_phase(phase, params):
    """
    Use BATMAN to compute a transit light curve from orbital phase.

    phase should be in [0, 1), with phase=0 at mid-transit.
    Returns:
        transit_weight = 1 - flux, normalized to max=1
        transit_flux
        phase_centered
    """

    phase_centered = (phase + 0.5) % 1.0 - 0.5

    bp = batman.TransitParams()
    bp.t0 = 0.0
    bp.per = 1.0

    # These names may need tiny edits depending on your parameters.py.
    bp.inc = _to_float(params.inclination, u.deg)
    bp.rp = float(params.RpRstar)
    bp.a = float(params.aRRatio)

    bp.ecc = 0.
    bp.w = 90.

    bp.limb_dark = "quadratic"
    bp.u = params.u_list

    m = batman.TransitModel(bp, phase_centered)
    transit_flux = m.light_curve(bp)

    transit_weight = 1.0 - transit_flux

    # Avoid tiny numerical nonzero weights out of transit.
    transit_weight[transit_weight < 1e-10] = 0.0

    # Normalize so full-transit-ish points have weight ~1.
    if np.nanmax(transit_weight) > 0:
        transit_weight = transit_weight / np.nanmax(transit_weight)

    return transit_weight, transit_flux, phase_centered

def weighted_ccf_per_rv(data, ivar, model):
	"""
	data: (n_exp, n_pix_sel) residual magnitudes
	ivar: (n_exp, n_pix_sel) inverse variance in magnitudes
	model: (n_pix_sel,) delta-mag template, mean ~ 0

	Returns:
	  ccf: (n_exp,) weighted normalized dot product per exposure
	"""
	m = np.isfinite(data) & np.isfinite(ivar) & np.isfinite(model)[None, :] & (ivar > 0)
	if np.sum(m) == 0:
		return np.full(data.shape[0], np.nan)

	# zero-fill masked entries
	w = np.where(m, ivar, 0.0)
	d = np.where(m, data, 0.0)
	t = np.where(np.isfinite(model), model, 0.0)[None, :]

	# (optional but recommended) remove weighted mean per exposure
	wsum = np.sum(w, axis=1, keepdims=True)
	dmean = np.where(wsum > 0, np.sum(w * d, axis=1, keepdims=True) / wsum, 0.0)
	d = d - dmean

	# model mean (weighted global) — usually already ~0; keep stable:
	twsum = np.sum(w, axis=0, keepdims=True)
	# (we won't per-exposure recenter model; not needed)

	num = np.sum(w * d * t, axis=1)

	den_d = np.sum(w * d * d, axis=1)
	den_t = np.sum(w * t * t, axis=1)

	den = np.sqrt(den_d * den_t)
	return np.where(den > 0, num / den, np.nan)


def modelCorrelation_weighted(order_data, order_wave, order_sigmag, RV, wavMinMax, F_model):
	"""
	order_data: (n_exp, n_pix) SYSREM residual magnitudes for one order
	order_wave: (n_pix,) wavelength for that order (common grid)
	order_sigmag: (n_exp, n_pix) mag errors (same space as order_data)
	RV: (n_RV,) in km/s (floats)
	wavMinMax: [wavmin, wavmax] in same units as order_wave
	F_model: interpolation function for template vs wavelength (rest frame)

	Returns:
	  cmap: (n_exp, n_RV)
	"""
	wav = order_wave
	wav_idx = (wav >= wavMinMax[0]) & (wav <= wavMinMax[1])

	data = order_data[:, wav_idx]
	sig = order_sigmag[:, wav_idx]
	ivar = np.where(np.isfinite(sig) & (sig > 0), 1.0 / (sig * sig), 0.0)

	cmap = np.full((data.shape[0], len(RV)), np.nan, dtype=float)

	for vdx, v in enumerate(RV):
		# Evaluate rest-frame template at lambda_rest corresponding to observed lambda for velocity v
		# v is in km/s float
		lam_rest = order_wave[wav_idx] / (1.0 + (v * u.km/u.s / cs.c).decompose().value)
		model = F_model(lam_rest)

		# model standardization helps a lot (optional but recommended)
		mm = np.array(model, dtype=float)
		m = np.isfinite(mm)
		if m.sum() < 10:
			continue
		mm = mm - np.nanmean(mm[m])
		s = np.nanstd(mm[m])
		if s > 0:
			mm = mm / s

		cmap[:, vdx] = weighted_ccf_per_rv(data, ivar, mm)

	return cmap

def template_to_dmag(template_flux, mode="depth"):
	"""
	Convert a model to delta-mag template.

	template_flux:
	  - if mode="depth": assumed to be line depth (0..something), continuum ~0
	  - if mode="transmission": assumed ~1 in continuum with dips (<1)
	"""
	tf = np.array(template_flux, dtype=float)

	if mode == "depth":
		# Delta-mag proportional to depth (small signal approx)
		dmag = 1.0857362047581296 * tf
	elif mode == "transmission":
		# template is transmission (around 1); convert to mag residual
		# guard: clip to avoid log of <=0
		tfc = np.clip(tf, 1e-10, None)
		dmag = -2.5 * np.log10(tfc)
		dmag -= np.nanmean(dmag)
	else:
		raise ValueError("mode must be 'depth' or 'transmission'")

	dmag -= np.nanmean(dmag)
	return dmag


def Vp(kp, phases):
	# phases: (n_exp_used,)
	return kp * np.sin(2.0 * np.pi * phases)

def finalCorr_stack(Kp_grid, RV_grid, inCorr, phases, minv=None, maxv=None, weights=None):
	"""
	Stack per-exposure CCFs into Kp-RV space.

	Kp_grid: (n_kp,) km/s
	RV_grid: (n_rv,) km/s, output RV axis
	inCorr:  (n_exp, n_rv) per-exposure CCF map
	phases:  (n_exp,) orbital phases for those exposures
	weights: None or (n_exp,)

	If weights is None:
		Uses the historical behaviour: unweighted nanmean over exposures.

	If weights is provided:
		Uses a weighted nanmean over exposures. Exposures with weight <= 0
		or non-finite weight do not contribute.

	Returns:
		fmap: (n_kp, n_rv)
	"""

	RV_grid = np.asarray(RV_grid, dtype=float)
	fmap = np.full((len(Kp_grid), len(RV_grid)), np.nan, dtype=float)

	if weights is not None:
		weights = np.asarray(weights, dtype=float)
		weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)

	for kdx, kp in enumerate(Kp_grid):

		vp = Vp(kp, phases)  # (n_exp,)

		# shifted x-axis per exposure
		# Convention: RV_shifted = RV - vp
		# This means we move into planet rest frame when we interpolate.
		interp_stack = np.full_like(inCorr, np.nan, dtype=float)

		for edx in range(inCorr.shape[0]):

			# If transit weights are supplied, skip out-of-transit / zero-weight exposures.
			if weights is not None and weights[edx] <= 0:
				continue

			x = RV_grid - vp[edx]
			y = inCorr[edx, :]

			m = np.isfinite(x) & np.isfinite(y)
			if m.sum() < 5:
				continue

			xs = x[m]
			ys = y[m]
			s = np.argsort(xs)
			xs = xs[s]
			ys = ys[s]

			f = interpolate.interp1d(
				xs,
				ys,
				bounds_error=False,
				fill_value=np.nan,
			)

			interp_stack[edx, :] = f(RV_grid)

		if weights is None:
			# Historical emission behaviour.
			fmap[kdx, :] = np.nanmean(interp_stack, axis=0)

		else:
			# Transit-weighted version.
			w = weights[:, None]
			finite = np.isfinite(interp_stack)

			numerator = np.nansum(interp_stack * w, axis=0)
			denominator = np.nansum(w * finite, axis=0)

			fmap[kdx, :] = numerator / denominator

	return fmap












# def injectModel(wav, data, orbital, reduce_res, planet_motion, strength = 1.):

# 	planet_signal = np.zeros(np.shape(data))

# 	for odx, o in enumerate(data):

# 		planet_flux_interp = reduce_res(wav[odx][0])

# 	#planet_flux_interp = reduce_res(wav)

# 		for vdx, v in enumerate(planet_motion):
# 			Dopplerwav = wav[odx][0] * (1 + (v/cs.c).decompose().value)
# 			F = interpolate.interp1d(Dopplerwav, planet_flux_interp, kind='linear', fill_value=np.nan, bounds_error = False)
# 			interpModel = F(wav[odx][0])
# 			planet_signal[odx, vdx,:] = interpModel

# 	injected = data * (1 - strength * planet_signal)
# 	scale = (1 - strength * planet_signal)

# 	return injected, scale

def injectModel(wav, data, orbital, reduce_res, planet_motion, strength = 1.):

	planet_signal = np.zeros(np.shape(data))

	for odx in range(data.shape[0]):
		wave_ord = wav[odx][0]

		for vdx, v in enumerate(planet_motion):
			lam_shift = wave_ord / (1 + (v/cs.c).decompose().value)
			interpModel = reduce_res(lam_shift)
			planet_signal[odx, vdx, :] = interpModel

	injected = data * (1 - strength * planet_signal)
	scale = (1 - strength * planet_signal)

	return injected, scale











