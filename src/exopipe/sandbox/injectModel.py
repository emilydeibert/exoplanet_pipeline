"""
Author: Emily Deibert
Last Modified: 2021-07-26
Description: Script to cross-correlate atmospheric models with data.
"""

from astropy import units as u
from scipy import interpolate
import numpy as np 
import sys
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.io import fits
from astropy.wcs import WCS
from scipy import signal
import warnings
import glob

sys.path.append('../src/')
import crosscorrelation as cc
import parameters as params
import reduction
import sysrem
import config
import tools
import os

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

def runSYSREM(data, wave, error, airmass, iters):

	mags, merrs = sysrem.GetMagnitudes(data, wave, error)
	residuals = sysrem.TotalSYSREM(mags, merrs, airmass, iters)

	return residuals

def correct_badPix(data, badPix):
	"""
	Applies the bad pixel mask, +/- 1 pixel on each side,
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

	for idx, f in enumerate(badPix):
		for odx, o in enumerate(f):
			for pdx, p in enumerate(o):
				if p == 0:
					pass
				elif p == 1:
					corrected[idx, odx, pdx-2:pdx+2] = np.nan

	return corrected

def testSYSREM(data, error, wave, airmass, maxIters, camera, night):
	"""
	Runs a varying number of iterations of SYSREM on the data.
	"""

	mags, merrs = sysrem.GetMagnitudes(data, wave, error)
	residuals = sysrem.TotalSYSREM(mags, merrs, airmass, 1)
	residuals.dump(config.path2reduced+'/sysrem/'+night+'_'+camera+'_sysrem_1_injected.npy')

	for idx in range(2, maxIters+1):
		for odx, o in enumerate(mags):
			residuals[odx] = sysrem.SYSREM(residuals[odx], merrs[odx], airmass)
		residuals.dump(config.path2reduced+'/sysrem/'+night+'_'+camera+'_sysrem_'+str(idx)+'_injected.npy')
		print(idx)

	return

def main():
	return


if __name__ == '__main__':

	RV = config.RV 
	Kp = config.Kp #np.arange(-300, 300, 1)
	models = ['Fe']#['Fe']#config.models

	for model in models:

		print(model)
		mdl = np.load(config.path2models+model+'_model_withEnvelope.npy', allow_pickle=True)
		model_flux = mdl[:,1] * -1
		model_wvl = mdl[:,0] / 10.

		model_conv = convolve(model_flux, Gaussian1DKernel(stddev=config.ghost_res/2.35))


		reduce_res = interpolate.interp1d(model_wvl, model_conv, kind='linear', 
			fill_value=[np.nan], bounds_error = False)

		mdl2 = np.load(config.path2models+model+'_model.npy', allow_pickle=True)
		model_flux2 = mdl2[:,1]
		model_wvl2 = mdl2[:,0] / 10.

		model_conv2 = convolve(model_flux2, Gaussian1DKernel(stddev=config.ghost_res/2.35))


		reduce_res2 = interpolate.interp1d(model_wvl2, model_conv2, kind='linear', 
			fill_value=[np.nan], bounds_error = False)

		for night in config.nights:

			data = [np.load(config.path2reduced+night+'_'+camera+'_flux.npy', allow_pickle=True) for camera in config.camera]
			wave = [np.load(config.path2reduced+night+'_'+camera+'_wave.npy', allow_pickle=True) for camera in config.camera]
			error = [np.load(config.path2reduced+night+'_'+camera+'_error.npy', allow_pickle=True) for camera in config.camera]
			airmass = np.load(config.path2reduced+night+'_'+'red'+'_airmass.npy', allow_pickle=True)
			berv = np.load(config.path2reduced+night+'_'+'red'+'_berv.npy', allow_pickle=True)
			Vsys = params.Vsys * np.ones_like(berv)#np.load(config.path2reduced+'red'+'_RV.npy', allow_pickle=True) * u.km / u.s
			orbital = np.load(config.path2reduced+night+'_'+'red'+'_orbital.npy', allow_pickle=True)
			badPix = [np.load(config.path2reduced+night+'_'+camera+'_badPix.npy', allow_pickle=True) for camera in config.camera]

			planet_motion = tools.orbitalMotion(params.K_p, orbital) + (Vsys) - (berv * u.km / u.s)
			#planet_motion_reverse = tools.orbitalMotion(-1. * params.K_p, orbital)

			for cameradx in [0, 1]:

				if config.camera[cameradx] == 'red':
					cameraDict = config.redCameraDict
					good_orders = np.arange(config.redOrders)
				elif config.camera[cameradx] == 'blue':
					cameraDict = config.blueCameraDict
					good_orders = np.arange(config.blueOrders)

				orders_to_correlate = np.arange(cameraDict['orders'])
				orders_to_correlate = tools.orders2keep(wave[cameradx], 0.000001, mdl)
				orders_to_correlate = list(set(orders_to_correlate).intersection(good_orders))

				injected = cc.injectModel(wave[cameradx][orders_to_correlate], data[cameradx][orders_to_correlate], orbital, 
					reduce_res, planet_motion)#planet_motion_reverse)

				corrected = correct_badPix(injected, badPix[cameradx][orders_to_correlate])
				corrected_error = correct_badPix(error[cameradx][orders_to_correlate], badPix[cameradx][orders_to_correlate])

				zapped = np.zeros(np.shape(corrected))
				zapped_error = np.zeros(np.shape(corrected_error))
				for odx, o in enumerate(corrected):
					z = reduction.zap(o, 5)
					zErr = reduction.errorZap(o, corrected_error[odx], 5)
					zapped[odx] = z
					zapped_error[odx] = zErr

				med_frame = np.nanmedian(zapped, axis=1)

				normalized = np.zeros(np.shape(zapped))
				normalized_err = np.zeros(np.shape(zapped))

				for odx, o in enumerate(zapped):
					for fdx, f in enumerate(o):
						new = f / med_frame[odx]
						new_err = zapped_error[odx,fdx,:] / med_frame[odx]

						normalized[odx,fdx,:] = new / np.nanmedian(new)
						normalized_err[odx,fdx,:] = new_err / np.nanmedian(new)

				testSYSREM(normalized, normalized_err, wave[cameradx][orders_to_correlate], airmass, 15, config.camera[cameradx], night)
				#sysrems = np.asarray(sysrems)
				#sysrems.dump('sysremz.npz')
				print('done sysrem')

				#if config.camera[cameradx] == 'red':
				#	idx = 6
				#elif config.camera[cameradx] == 'blue':
				#	idx = 8

				#sysr = runSYSREM(normalized, wave[cameradx][orders_to_correlate], normalized_err, airmass, idx)

				for idx in range(1, 16):
					print(idx)

					sysr = np.load(config.path2reduced+'sysrem/'+night+'_'+config.camera[cameradx]+'_sysrem_'+str(idx)+'_injected.npy', allow_pickle=True)


					cmap_results = {}
					fmap_results = {}

					sysr_rest, error_rest = shift2BERV(sysr, wave[cameradx][orders_to_correlate], normalized_err, 
					 	berv, Vsys)
				#sysr_rest = sysr
				#error_rest = normalized_err

					for odx, o in enumerate(sysr_rest):
						cmap, fmap = correlateModel(np.nan_to_num(o), wave[cameradx][orders_to_correlate][odx,0,:], 
						RV, Kp, reduce_res2, orbital)

						cmap_results[str(orders_to_correlate[odx])] = cmap 
						fmap_results[str(orders_to_correlate[odx])] = fmap

					np.save(config.path2reduced+'injected/'+night+'_'+config.camera[cameradx]+'_cmap_'+model+'_'+str(idx)+'_iters.npy', cmap_results)
					np.save(config.path2reduced+'injected/'+night+'_'+config.camera[cameradx]+'_fmap_'+model+'_'+str(idx)+'_iters.npy', fmap_results)

					print(config.camera[cameradx])
				print(night)

	main()