from astropy.convolution import convolve, Gaussian1DKernel
from scipy.ndimage import median_filter
from matplotlib import pyplot as plt
from astropy import units as u
from scipy import interpolate
from astropy.io import fits
from pathlib import Path
from scipy import signal
import importlib.util
import numpy as np 
import argparse
import glob
import sys
import os

from exopipe import reduction
from exopipe import crosscorrelation as cc
from exopipe import tools
from exopipe import sysrem

def mask_badpix(data, variance, badPix):
	"""
	Applies the bad pixel mask, +/- 2 pixel on each side,
	to the data.

	Parameters
	----------
	data: `np.ndarray`
		3D data cube of flux
	badPix: `np.ndarray`
		3D data cube of bad pixels

	Returns
	----------
	corrected: `np.ndarray`
		3D data cube with bad pixels corrected
	"""

	corrected = np.copy(data)
	corrected_var = np.copy(variance)

	for idx, f in enumerate(badPix):
		for odx, o in enumerate(f):
			for pdx, p in enumerate(o):
				if p == 0:
					pass
				elif p == 1:
					corrected[idx, odx, pdx-2:pdx+2] = np.nan
					corrected_var[idx, odx, pdx-2:pdx+2] = np.nan

	return corrected, corrected_var

def mad_clip_mask_timeaxis(x, nsig=5.0):
    """
    Returns mask True where values are outliers along time axis
    for each wavelength pixel independently.

    x: (n_exp, n_pix)
    """
    med = np.nanmedian(x, axis=0)                      # (n_pix,)
    resid = x - med[None, :]
    mad = np.nanmedian(np.abs(resid), axis=0)          # (n_pix,)

    # Avoid divide-by-zero: if mad==0, never flag based on it
    thresh = nsig * 1.4826 * mad                       # robust sigma estimate
    mask = resid > thresh[None, :]

    mask[:, mad == 0] = False
    return mask

def normalize_order_lightcurve(flux, variance=None, q=75):
    """
    Normalize each (order, exposure) by a robust scalar (percentile across pixels).

    flux: (n_ord, n_exp, n_pix)
    variance: same shape or None
    q: percentile used as scale (75 is a nice default)
    Returns:
      flux_n, var_n, scale  (scale has shape (n_ord, n_exp, 1))
    """
    flux = flux.astype(np.float64, copy=False)

    scale = np.nanpercentile(flux, q, axis=2, keepdims=True)  # (n_ord, n_exp, 1)
    # protect against zeros/infs
    bad = ~np.isfinite(scale) | (scale <= 0)
    scale[bad] = np.nan

    flux_n = flux / scale

    if variance is None:
        return flux_n, None, scale

    variance = variance.astype(np.float64, copy=False)
    var_n = variance / (scale ** 2)

    return flux_n, var_n, scale

def flatten_orders(flux_norm, width=301):
    """
    Remove low-frequency continuum per (exposure, order).
    flux_norm: (n_exp, n_ord, n_pix)
    Returns: flattened_flux, continuum
    """
    x = flux_norm.copy()

    # Fill NaNs with per-(exp,order) median across pixels
    fill = np.nanmedian(x, axis=2, keepdims=True)  # (n_exp, n_ord, 1)
    x = np.where(np.isfinite(x), x, fill)

    # Median filter only along pixel axis
    cont = median_filter(x, size=(1, 1, width), mode="nearest")
    cont = np.where((np.isfinite(cont) & (cont != 0)), cont, np.nan)

    flat = flux_norm / cont
    return flat, cont

def mask_order_edges(arr, n_edge=100, value=np.nan):
    """
    Mask n_edge pixels on both ends of the pixel axis.
    arr: (n_exp, n_ord, n_pix)
    """
    out = arr.copy()
    if n_edge <= 0:
        return out

    out[..., :n_edge] = value
    out[..., -n_edge:] = value
    return out


def correlateModel(srf, wave, RV, Kp, reduce_res, orbital):
	"""Correlates a single model with the relevant data
	from a single order."""

	cmap = cc.modelCorrelation(srf, wave, RV, [min(wave), max(wave)], reduce_res)
	fmap = cc.finalCorr(Kp, RV, cmap, orbital)

	return cmap, fmap


def shift2BERV(sysr, wave, error, berv, vsys):

	sysr_berv = np.full_like(sysr, np.nan)
	sysr_berv_error = np.full_like(error, np.nan)
	for odx, o in enumerate(sysr):
		r, re = tools.shift2rest(o, wave[odx][0], -1. * berv * u.km / u.s + vsys, error[odx])
		sysr_berv[odx] = r 
		sysr_berv_error[odx] = re

	return sysr_berv, sysr_berv_error

def main():
	return

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument("project_path", type=str)

	parser.add_argument(
		"--iter",
		type=int,
		default=None,
		help="Correlate a single SYSREM iteration"
		)

	parser.add_argument(
	    "--inject-sign",
	    choices=["positive", "negative"],
	    default="positive"
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

	RV = config.RV 
	Kp = config.Kp

	for model in config.models:

		print(model)

		mdl_inject = np.load(config.path2models+model+'_model_withEnvelope.npy', allow_pickle=True)
		model_flux_inject = mdl_inject[:,1]
		model_wvl_inject = mdl_inject[:,0] / 10.

		reduce_res_convolve = convolve(model_flux_inject, Gaussian1DKernel(stddev=config.ghost_res/2.35))

		reduce_res_inject = interpolate.interp1d(model_wvl_inject, reduce_res_convolve,
			kind = 'linear', bounds_error = False, fill_value = np.nan)

		mdl = np.load(config.path2models+model+'_model.npy', allow_pickle=True)
		model_flux = mdl[:,1]
		model_wvl = mdl[:,0] / 10.

		model_conv = convolve(model_flux, Gaussian1DKernel(stddev=config.ghost_res/2.35))

		model_conv = cc.template_to_dmag(model_conv)
		
		F_model= interpolate.interp1d(model_wvl, model_conv, kind='linear', bounds_error=False, fill_value=np.nan)		

		for night in config.nights:

			for camera in config.camera:

				with np.load(config.path2reduced+night+'_'+camera+'.npz') as data_array:
					flux = data_array['flux']
					badpix = data_array['badpix']
					variance = data_array['variance']
					phase = data_array['phase']
					wave = data_array['wave']
					berv = data_array['berv']
					airmass = data_array['airmass']
					bjd = data_array['bjd']
					bjd_scale = data_array['bjd_scale']

				inject_kp = params.K_p

				if args.inject_sign == "negative":
					inject_kp *= -1

				planet_motion = (tools.orbitalMotion(inject_kp, phase) + params.Vsys - (berv * u.km/u.s))

				flux, scale = cc.injectModel(wave, flux, phase, reduce_res_inject, planet_motion)
				variance = variance * scale**2

				flux_masked, variance_masked = mask_badpix(flux, variance, badpix)

				scale = np.nanpercentile(flux_masked, 75, axis=2, keepdims=True)
				scale[~np.isfinite(scale) | (scale == 0)] = np.nan
				flux_tmp = flux_masked / scale

				for odx in range(flux_masked.shape[1]):
					x = flux_tmp[:, odx, :]
					m = mad_clip_mask_timeaxis(x, nsig=5.0)
					flux_masked[:, odx, :][m] = np.nan
					variance_masked[:, odx, :][m] = np.nan

				norm_flux, norm_var, scale = normalize_order_lightcurve(flux_masked, variance_masked, q=75)

				flat_flux, cont = flatten_orders(norm_flux, width=601)
				flat_var = norm_var / cont**2

				# chop edges 
				flux2 = mask_order_edges(flat_flux, n_edge=100, value=np.nan)
				var2  = mask_order_edges(flat_var, n_edge=100, value=np.nan)

				err = np.sqrt(var2)
				mag, magerr = sysrem.flux_to_mag(flux2, err)

				orders_to_correlate = tools.orders2keep(wave, 0.000001, mdl)

				if args.iter is None:
					n_sysrem = config.sysremIters
					iter_list = np.arange(1, config.sysremIters+1)

				else:
					n_sysrem = args.iter
					iter_list = [args.iter]

				resid_mag_cube = sysrem.sysrem_total(mag, magerr, airmass, n_app=n_sysrem)

				for k in iter_list:
					sysr_k = resid_mag_cube[k-1]

					sysr_rest, error_rest = shift2BERV(sysr_k, wave, magerr, berv, params.Vsys * np.ones_like(berv))

					cmap_results = np.zeros((len(orders_to_correlate), sysr_rest.shape[1], len(RV)))
					fmap_results = np.zeros((len(orders_to_correlate), len(Kp), len(RV)))

					for idx, order in enumerate(orders_to_correlate):

						cmap = cc.modelCorrelation_weighted(
						order_data=sysr_rest[order, :, :],             # (n_exp, n_pix)
						order_wave=wave[order, 0, :],                  # (n_pix,)
						order_sigmag=error_rest[order, :, :],          # (n_exp, n_pix)
						RV=RV,
						wavMinMax=[np.nanmin(wave[order,0,:]), np.nanmax(wave[order,0,:])],
						F_model=F_model
						)

						fmap = cc.finalCorr_stack(Kp, RV, cmap, phase)
									
						cmap_results[idx] = cmap 
						fmap_results[idx] = fmap

					np.savez_compressed(
						f"{config.path2reduced}/injected/{night}_{camera}_{model}_{k}_iters_injected_{args.inject_sign}.npz",
						cmap = cmap_results,
						fmap = fmap_results,
						orders = orders_to_correlate,
						k = k,
						model = model,
						inject_kp = inject_kp
					)

	main()

