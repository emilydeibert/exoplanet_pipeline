"""
Author: Emily Deibert
Last Modified: 17-10-2023
Description: Tools to reduce extracted GHOST files.
"""

from astropy.stats import median_absolute_deviation
import numpy as np

def zap(data, MAD):
	""" Tool to zap cosmic rays or other bad pixels.
	Clean on a per-order basis.
	Checks the time axis at each wavelength for pixels that are outside the specified MAD.
	Parameters
	----------
	data: array
		array for a single order containing (frame x pixel) for the spectrum
	MAD: int
		number of median absolute deviations outside of which to consider something a cosmic ray or bad pixel
	Returns
	----------
	clean: array
		a copy of the data, with the cosmic rays/bad pixels replaced by the median value in that wavelength channel
	"""

	clean = np.copy(data)
	for i in range(np.shape(clean)[1]):
		clean[:,i][np.where(clean[:,i] > np.median(clean[:,i] + MAD * median_absolute_deviation(clean[:,i])))[0]] = np.nan #np.median(clean[:,i])
	
	return clean

def errorZap(data, error, MAD):
	""" Tool to apply the zap of cosmic rays or other bad pixels to the errors.
	Clean on a per-order basis.
	Checks the time axis at each wavelength for pixels that are outside the specified MAD.
	Parameters
	----------
	data: array
		array for a single order containing (frame x pixel) for the spectrum
	MAD: int
		number of median absolute deviations outside of which to consider something a cosmic ray or bad pixel
	error: array
		array for a single order containing (frame x pixel) for the errors on the spectrum
	Returns
	----------
	clean: array
		a copy of the data, with the cosmic rays/bad pixels replaced by the median value in that wavelength channel
	"""

	clean = np.copy(data)
	clerror = np.copy(error)
	
	for i in range(np.shape(clean)[1]):
		clerror[:,i][np.where(clean[:,i] > np.median(clean[:,i] + MAD * median_absolute_deviation(clean[:,i])))[0]] = np.nan #np.median(clerror[:,i])
	
	return clerror


# def zap(data, error, MAD):
# 	""" Tool to zap cosmic rays or other bad pixels.
# 	Clean on a per-order basis.
# 	Checks the time axis at each wavelength for pixels that are outside the specified MAD.
# 	Parameters
# 	----------
# 	data: array
# 		array for a single order containing (frame x pixel) for the spectrum
# 	MAD: int
# 		number of median absolute deviations outside of which to consider something a cosmic ray or bad pixel
# 	Returns
# 	----------
# 	clean: array
# 		a copy of the data, with the cosmic rays/bad pixels replaced by the median value in that wavelength channel
# 	"""
# 	zapped = np.zeros(np.shape(data))
# 	zapped_variance = np.zeros(np.shape(error))

# 	for odx, o in enumerate(data):
# 		clean = np.copy(data)
# 		clerror = np.copy(error)
# 		for i in range(np.shape(clean)[1]):
# 			clean[:,i][np.where(clean[:,i] > np.median(clean[:,i] + MAD * median_absolute_deviation(clean[:,i])))[0]] = np.nan #np.median(clean[:,i])
# 			clerror[:,i][np.where(clean[:,i] > np.median(clean[:,i] + MAD * median_absolute_deviation(clean[:,i])))[0]] = np.nan #np.median(clerror[:,i])
# 		zapped[odx] = clean
# 		zapped_variance[odx] = clerror
		
# 	return zapped, zapped_variance