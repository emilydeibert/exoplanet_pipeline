"""
Author: Emily Deibert
Last Modified: November 20, 2024
Description: Reduces the data from the raw, DRS products.

"""
from scipy.ndimage import median_filter
from matplotlib import pyplot as plt
from astropy import units as u
from scipy import interpolate
from astropy.io import fits
from scipy import signal
import importlib.util
import numpy as np 
import glob
import sys
import os

from exopipe import reduction

from pathlib import Path

project_path = Path(sys.argv[1])

config_file = project_path / "config.py"
spec = importlib.util.spec_from_file_location("config", str(config_file))
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

params_file = project_path / "parameters.py"
spec = importlib.util.spec_from_file_location("parameters", str(params_file))
params = importlib.util.module_from_spec(spec)
spec.loader.exec_module(params)

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

    thresh = nsig * 1.4826 * mad     
    mask = resid > thresh[None, :]

    mask[:, mad == 0] = False
    return mask


def normalize_order_lightcurve(flux, variance=None, q=75):
    """
    Normalize each (order, exposure) by a scalar (percentile across pixels).

    flux: (n_ord, n_exp, n_pix)
    variance: same shape or None
    q: percentile used as scale (75 default)
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


def main():
	return


if __name__ == '__main__':

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


			np.savez_compressed(
				f"{config.path2reduced}/{night}_{camera}_analysis_ready.npz",
				flux=flux_masked.astype(np.float32),
				norm_flux = flux2,
				variance=variance_masked.astype(np.float32),
				norm_var = var2,
				wave=wave.astype(np.float64),
				phase = phase,
				berv = berv,
				airmass = airmass,
				bjd = bjd,
				bjd_scale = bjd_scale
				)

	main()



























