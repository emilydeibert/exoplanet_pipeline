from astropy.convolution import convolve, Gaussian1DKernel
from astropy import units as u
from scipy import interpolate
import importlib.util
import numpy as np 
import argparse
import sys

from exopipe import crosscorrelation as cc
from exopipe import tools

from pathlib import Path

#project_path = Path(sys.argv[1])

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

	parser = argparse.ArgumentParser()
	parser.add_argument("project_path", type=str)
	parser.add_argument("--sysrem-mode", default="full")  # full | single | list
	parser.add_argument("--k", type=int, default=15)       # used for full/single
	parser.add_argument("--ks", type=str, default=None)    # e.g. "1,3,5,10"

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

	if args.sysrem_mode == "full":
		k_list = list(range(args.k))

	elif args.sysrem_mode == "single":
		k_list = [args.k]

	elif args.sysrem_mode == "list":
		k_list = [int(x) for x in args.ks.split(",")]

	else:
		raise ValueError("Unknown SYSREM mode")

	RV = config.RV 
	Kp = config.Kp 
	models = config.models
	nights = config.nights

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

				# if camera == 'red':
				# 	cameraDict = config.redCameraDict
				# elif camera == 'blue':
				# 	cameraDict = config.blueCameraDict

				sysr_array = np.load(f"{config.path2reduced}/{night}_{camera}_sysrem.npz")
				data_array = np.load(f"{config.path2reduced}/{night}_{camera}_analysis_ready.npz")

				sysr = sysr_array['sysrem']
				magerr = sysr_array['magerr']
				wave = data_array['wave']
				berv = data_array['berv']
				Vsys = params.Vsys * np.ones_like(berv)
				phase = data_array['phase']

				orders_to_correlate = tools.orders2keep(wave, 0.000001, mdl)

				for k in k_list:
					sysr_k = sysr[k-1]

					sysr_rest, error_rest = shift2BERV(sysr_k, wave, magerr, berv, Vsys)

					cmap_results = np.zeros((len(orders_to_correlate), sysr_rest.shape[1], len(RV)))
					fmap_results = np.zeros((len(orders_to_correlate), len(Kp), len(RV)))

					#idx = 0

					#for odx, o in enumerate(sysr_rest):
					for idx, order in enumerate(orders_to_correlate):

						#if odx in orders_to_correlate:

						# per order correlation
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

						#idx += 1

					np.savez_compressed(
						f"{config.path2reduced}/results/{night}_{camera}_{model}_k{k}_iters.npz",
						cmap = cmap_results,
						fmap = fmap_results,
						orders = orders_to_correlate,
						k = k+1,
						model = model
						)

	main()

