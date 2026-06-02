"""
Author: Emily Deibert
Last Modified: 
Description: Runs SYSREM on the data
"""

import importlib.util
import numpy as np 
import sys

from exopipe import sysrem

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


def main():

	for night in config.nights:
		print(night)

		for camera in config.camera:
			print(camera)

			data_array = np.load(config.path2reduced+night+'_'+camera+'_analysis_ready.npz')
			flux = data_array['norm_flux']
			variance = data_array['norm_var']
			wave = data_array['wave']
			airmass = data_array['airmass']
			phase = data_array['phase']
			berv = data_array['berv']
			bjd = data_array['bjd']
			bjd_scale = data_array['bjd_scale']

			err = np.sqrt(variance)
			mag, magerr = sysrem.flux_to_mag(flux, err)   # same shape as flux

			resid_mag = sysrem.sysrem_total(
				mag, magerr, airmass,
				n_app = config.sysremIters,
				tol = 1e-4,
				max_iter = 50)

			np.savez_compressed(
				f"{config.path2reduced}/{night}_{camera}_sysrem.npz",
				sysrem = resid_mag.astype(np.float32),
				magerr = magerr.astype(np.float32))#,
				#wave = wave,
				#phase = phase,
				#berv = berv,
				#bjd = bjd,
				#bjd_scale = bjd_scale)

	return

if __name__ == '__main__':
	main()
