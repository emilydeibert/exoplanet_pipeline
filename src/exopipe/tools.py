"""
Author: Emily Deibert
Last Modified: 2023-10-23
Description: Various tools used in analysis.
"""

import numpy as np
from scipy import interpolate
from astropy import units as u 
from astropy import constants as cs

def vac2air(vacwave):
	"""
	Converts a wavelength in vacuum to a wavelength in air.
	Input must be in Angstroms (though would be nice to introduce
	astropy units flexibility).

	Parameters
	----------
	vacwave: array
		input wavelength axis in vacuum

	Returns
	----------
	airwave: array
		output wavelength axis in air
	"""

	s = 1e4 / vacwave # in AA
	n = 1. + 8.34254e-5 + 2.406147e-2 / (130. - s**2) + 1.5998e-4 / (38.9 - s**2)
	airwave = vacwave / n

	return airwave


def align2common(spec, wave, err, frame=0):
	"""
	Aligns all spectra to common frame.

	Parameters
	----------
	spec: array
		Array containing spectra; shape = (frames, pixels)
	wave: array
		Array containing wavelengths; shape = (frames, pixels)
	err: array
		Array containing errors; shape = (frame, pixels)
	frame: int
		Frame to align to (default = 1st frame)

	Returns
	----------
	aligned: array
		Aligned spectra
	common_wave: array
		Common wavelength axis
	aligned_err: array
		Aligned errors
	"""

	aligned = np.zeros(np.shape(spec))
	aligned_err = np.zeros(np.shape(err))

	for fdx, f in enumerate(spec):

		interp = interpolate.interp1d(wave[fdx, :], f, kind='linear', fill_value=(np.nan, np.nan), bounds_error=False)
		interp_err = interpolate.interp1d(wave[fdx, :], err[fdx, :], kind='linear', fill_value=(np.nan, np.nan), bounds_error=False)

		align_y = interp(wave[frame, :])
		align_y_err = interp_err(wave[frame, :])

		aligned[fdx, :] = align_y
		aligned_err[fdx, :] = align_y_err

	common_wave = wave[frame, :]

	return aligned, common_wave, aligned_err


def undoDoppler(lambda_obs, v):
	"""
	Function to undo Doppler shift.
	
	Parameters
	----------
	lambda_obs: array
		Observed wavelength array
	v: astropy.units.quantity.Quantity
		Velocity of Doppler shift

	Returns
	----------
	array
		Un-Doppler-shifted array
	"""

	return lambda_obs / (1 + v/cs.c)


def Doppler(lyambda, v):
	'''
	Function to Doppler shift.

	Parameters
	----------
	lyambda: array
		Array of wavelengths to Doppler shift
	v: astrop.units.quantity.Quantity
		Velocity to Doppler shift to

	Returns
	----------
	array
		Doppler-shifted array
	'''

	return lyambda * (1 + v/cs.c)


def orbitalMotion(K, phase):
	'''
	Function to compute orbital motion.

	Parameters
	----------
	K: astrop.units.quantity.Quantity
		Keplerian velocity semi-amplitude
	phase: array
		Array containing orbital phase values

	Returns
	----------
	array
		Array containing orbital motion at each orbital phase
	'''

	return K * np.sin(2. * np.pi * phase)


def shift2rest(flux, wave, velocity, error):
	'''
	Function to shift to a rest frame

	Parameters
	----------
	flux: array
		Array that will be shifted to a rest frame
	wave: array
		Wavelength array corresponding to flux
	velocity: array
		Velocity of the shift
	error: array
		Corresponding errors to flux

	Returns
	----------
	rest_frame: array
		Array containing values shifted to rest frame
	rest_error: array
		Array containing error values shifted to rest frame

	'''
	rest_frame = np.zeros(np.shape(flux))
	rest_error = np.zeros(np.shape(error))

	for fdx, f in enumerate(flux):
		
		shifted_lambda = Doppler(wave, velocity[fdx])
		interp = interpolate.interp1d(wave, f, kind='linear', fill_value=(np.nan, np.nan), bounds_error=False)
		interp_err = interpolate.interp1d(wave, error[fdx, :], kind='linear', fill_value = (np.nan, np.nan), bounds_error = False)

		resting = interp(shifted_lambda)
		resting_error = interp_err(shifted_lambda)

		rest_frame[fdx, :] = resting
		rest_error[fdx, :] = resting_error

	return rest_frame, rest_error

def orders2keep(wave, thresh, model):

	wvl = model[:,0] / 10.
	flux = model[:,1]

	wave = wave[:,0,:]

	orders = []

	for odx, o in enumerate(wave):
		lims = np.where((wvl > min(o))&(wvl < max(o)))[0]
		model_lims = flux[lims]
		if not (model_lims > thresh).any():
			pass
		else:
			orders.append(odx)

	return orders