from astropy import constants as cs 
from astropy import units as u
from scipy import interpolate
import numpy as np 
import sys

from exopipe import tools

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

def finalCorr_stack(Kp_grid, RV_grid, inCorr, phases, minv=None, maxv=None):
	"""
	Kp_grid: (n_kp,) km/s
	RV_grid: (n_rv,) km/s  (this will be the output RV axis)
	inCorr:  (n_exp, n_rv) per-exposure CCF map
	phases:  (n_exp,) orbital phases for those exposures

	Returns:
	  fmap: (n_kp, n_rv)
	"""
	RV_grid = np.asarray(RV_grid, dtype=float)
	fmap = np.full((len(Kp_grid), len(RV_grid)), np.nan, dtype=float)

	for kdx, kp in enumerate(Kp_grid):
		vp = Vp(kp, phases)  # (n_exp,)
		# shifted x-axis per exposure
		# Your convention: RV_shifted = RV - vp (matches your earlier code)
		# This means we "move into planet rest frame" when we interpolate.
		interp_stack = np.full_like(inCorr, np.nan, dtype=float)

		for edx in range(inCorr.shape[0]):
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

			f = interpolate.interp1d(xs, ys, bounds_error=False, fill_value=np.nan)
			interp_stack[edx, :] = f(RV_grid)

		fmap[kdx, :] = np.nanmean(interp_stack, axis=0)
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











