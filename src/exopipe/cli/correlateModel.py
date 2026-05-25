from astropy.convolution import convolve, Gaussian1DKernel
from astropy import units as u
from scipy import interpolate
import importlib.util
import numpy as np 
import sys

from exopipe import crosscorrelation as cc
from exopipe import tools

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


def correlateModel(srf, wave, RV, Kp, reduce_res, orbital):
	"""Correlates a single model with the relevant data
	from a single order."""

	cmap = cc.modelCorrelation(srf, wave, RV, [min(wave), max(wave)], reduce_res)
	fmap = cc.finalCorr(Kp, RV, cmap, orbital)

	return cmap, fmap


def shift2BERV(sysr, wave, error, berv, vsys):

	sysr_berv = np.zeros(np.shape(sysr))
	sysr_berv_error = np.zeros(np.shape(error))
	for odx, o in enumerate(sysr):
		r, re = tools.shift2rest(o, wave[odx][0], -1. * berv * u.km / u.s + vsys, error[odx])
		sysr_berv[odx] = r 
		sysr_berv_error[odx] = re

	return sysr_berv, sysr_berv_error


def main():
	return

import matplotlib.pyplot as plt

if __name__ == '__main__':

	RV = config.RV 
	Kp = config.Kp 
	models = config.models
	nights = config.nights

	ITERS = 4

	for model in models:

		print(model)

		mdl = np.load(config.path2models+model+'_model.npy', allow_pickle=True)
		model_flux = mdl[:,1]
		model_wvl = mdl[:,0] / 10.

		model_conv = convolve(model_flux, Gaussian1DKernel(stddev=config.ghost_res/2.35))

		model_conv = cc.template_to_dmag(model_conv)
		
		F_model= interpolate.interp1d(model_wvl, model_conv, kind='linear', bounds_error=False, fill_value=np.nan)

		for ndx, night in enumerate(nights):
			print(night)

			for camera in config.camera:
				print(camera)

				if camera == 'red':
					cameraDict = config.redCameraDict
				elif camera == 'blue':
					cameraDict = config.blueCameraDict

				data_array = np.load(f"{config.path2reduced}/sysrem/{night}_{camera}_sysrem_{ITERS}.npz")
				sysr = data_array['sysrem']
				variance = data_array['magerr']
				wave = data_array['wave']
				berv = data_array['berv']
				Vsys = params.Vsys * np.ones_like(berv)
				phase = data_array['phase']

				sysr_rest, error_rest = shift2BERV(sysr, wave, variance, berv, Vsys)

				orders_to_correlate = tools.orders2keep(wave, 0.000001, mdl)

				cmap_results = np.zeros((len(orders_to_correlate), sysr.shape[1], len(RV)))
				fmap_results = np.zeros((len(orders_to_correlate), len(Kp), len(RV)))

				idx = 0

				for odx, o in enumerate(sysr_rest):

					if odx in orders_to_correlate:

						# per order correlation
						cmap = cc.modelCorrelation_weighted(
						order_data=sysr_rest[odx, :, :],             # (n_exp, n_pix)
						order_wave=wave[odx, 0, :],                  # (n_pix,)
						order_sigmag=error_rest[odx, :, :],          # (n_exp, n_pix)
						RV=RV,
						wavMinMax=[np.nanmin(wave[odx,0,:]), np.nanmax(wave[odx,0,:])],
						F_model=F_model
						)

						fmap = cc.finalCorr_stack(Kp, RV, cmap, phase)
								
						cmap_results[idx] = cmap 
						fmap_results[idx] = fmap

						idx += 1

				np.savez_compressed(
					f"{config.path2reduced}/results/{night}_{camera}_{model}_{ITERS}_iters.npz",
					cmap = cmap_results,
					fmap = fmap_results,
					orders = orders_to_correlate
					)

	main()

