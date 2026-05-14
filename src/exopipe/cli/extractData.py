"""
Author: Emily Deibert
Last Modified: July 11, 2024
Description: Extracts raw data and necessary metadata from GHOST files.
"""

from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units as u
from astropy.time import Time
from astropy.io import fits
from pathlib import Path
import importlib.util
import numpy as np
import glob
import sys
import os
from exopipe import extract

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

location = EarthLocation.of_site('Gemini South')

def get_data(files, camera):
	"""
	From a list of .fits files, computs the data cube for 
	spectroscopic data, wavelength, and metadata.

	Parameters
	----------
	files: list
		list of relevant .fits files
	camera: string
		`red` or `blue`

	Returns
	----------
	"""

	files = sorted(files)

	if camera == 'red':
		cameraDict = config.redCameraDict
	elif camera == 'blue':
		cameraDict = config.blueCameraDict

	data = np.zeros((cameraDict['orders'], len(files), cameraDict['pixels']))
	badPix = np.zeros(np.shape(data))
	wave = np.zeros(np.shape(data))
	variance = np.zeros(np.shape(data))

	snr = np.zeros(np.shape(data))

	airmass = np.zeros(len(files))
	BERV = np.zeros(len(files))
	BJD = np.zeros(len(files))

	for idx, f in enumerate(files):
		temp = fits.open(f)

		target = SkyCoord(
			ra=temp[0].header['RA'] * u.deg,
			dec=temp[0].header['DEC'] * u.deg,
			frame='fk5'
			)

		utc = temp[0].header['DATE-OBS']
		utstart = temp[0].header['UTSTART']
		exptime = temp[0].header['EXPTIME']

		start = Time(
			f"{utc}T{utstart}",
			format='isot',
			scale='utc',
			location=location
			)

		t = start + (exptime / 2.0) * u.s

		airmass[idx] = temp[0].header['AIRMASS']

		# BERV
		berv_idx = target.radial_velocity_correction(obstime=t)
		BERV[idx] = berv_idx.to(u.km/u.s).value

		# BJD_TDB
		ltt_bary = t.light_travel_time(target)
		bjd = t.tdb + ltt_bary
		BJD[idx] = bjd.jd

		data[:, idx, :] = temp[1].data
		wave[:, idx, :] = temp[4].data 
		badPix[:, idx, :] = temp[3].data
		variance[:, idx, :] = temp[2].data ## this is the variance

	sigma = np.sqrt(variance)

	snr_pix = np.where(
    	(sigma > 0) & np.isfinite(sigma),
    	data / sigma,
    	np.nan
		)

	# mask bad pixels
	snr_pix = np.where(badPix == 0, snr_pix, np.nan)

	# one number per exposure
	# only uses the orders within the useable range
	snr_exp_good = np.nanmedian(snr_pix[cameraDict['goodOrders']], axis=(0, 2))

	snr_exp = np.nanmedian(snr_pix, axis=(0, 2))

	snr = np.where(variance > 0, data / sigma, np.nan)

	return data, wave, variance, badPix, airmass, snr, snr_exp, BERV, BJD

def main():
	return

if __name__ == '__main__':

	for night in config.nights:

		for camera in config.camera:

			files = sorted(glob.glob(config.path2raw+night+'/*'+camera+'001_calibrated.fits'))

			data, wave, variance, badPix, airmass, snr, snr_exp, BERV, BJD = get_data(files, camera)

			BJD = Time(BJD, format='jd', scale='tdb', precision=9)

			orbital = extract.orbital_phase(params.midTransit, BJD, params.period)


			out = f"{config.path2reduced}{night}_{camera}.npz"

			np.savez_compressed(
    		out,
    		phase=orbital,
    		berv=BERV,
    		airmass=airmass,
    		bjd=BJD.tdb.jd.astype(np.float64),
    		bjd_scale="TDB",
    		wave=wave,
    		flux=data.astype(np.float32),
    		variance=variance.astype(np.float32),
    		badpix=badPix.astype(bool),
    		snr=snr.astype(np.float32),
    		snr_exp=snr_exp.astype(np.float32)
			)

	main()


