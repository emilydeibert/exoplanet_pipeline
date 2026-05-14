"""
Author: Emily Deibert
Last Modified: July 11, 2024
Description: Runs SYSREM on the data
"""

ITERS = 4

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

def sysrem_stack_components(mag_cube, magerr_cube, airmass, n_app,
                            orders=None, **kwargs):
    """
    Returns:
      resid_final: (n_ord_sel, n_exp, n_pix)
      comps:       (n_app, n_ord_sel, n_exp, n_pix)  [removed each iteration]
    """
    mag_cube = np.asarray(mag_cube, dtype=np.float64)
    magerr_cube = np.asarray(magerr_cube, dtype=np.float64)

    if orders is not None:
        mag_cube = mag_cube[orders]
        magerr_cube = magerr_cube[orders]

    n_ord, n_exp, n_pix = mag_cube.shape
    comps = np.full((n_app, n_ord, n_exp, n_pix), np.nan, dtype=np.float32)

    resid = mag_cube.copy()

    for k in range(int(n_app)):
        new_resid = np.full_like(resid, np.nan, dtype=np.float64)

        for o in range(n_ord):
            # one SYSREM application for one order
            out, a_j, c_i = sysrem.sysrem_one(resid[o], magerr_cube[o], airmass,
                                       return_trend=True, **kwargs)

            # component removed this iteration (in mag space)
            comp = resid[o] - out

            comps[k, o] = comp.astype(np.float32)
            new_resid[o] = out

        resid = new_resid

    return resid, comps


def reconstruct_residual_from_components(mag0, comps, k):
    """
    mag0:  (n_ord, n_exp, n_pix) initial magnitudes
    comps: (n_app, n_ord, n_exp, n_pix)
    k: number of iterations to apply (1..n_app)
    """
    return mag0 - np.nansum(comps[:k], axis=0)

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
				n_app = ITERS,
				tol = 1e-4,
				max_iter = 50,
				sigma_floor=("percentile", 5),
				weight_clip_quantile=0.995)

			np.savez_compressed(
				f"{config.path2reduced}/sysrem/{night}_{camera}_sysrem_{ITERS}.npz",
				sysrem = resid_mag,
				magerr = magerr,
				wave = wave,
				phase = phase,
				berv = berv,
				bjd = bjd,
				bjd_scale = bjd_scale)

			resid_final, comps = sysrem_stack_components(mag, magerr, airmass, n_app=15,
                                            tol=1e-4, max_iter=50,
                                            sigma_floor=("percentile", 5),
                                            weight_clip_quantile=0.995)

			np.savez_compressed(f"{config.path2reduced}/sysrem/{night}_{camera}_sysrem_components.npz",
                    comps=comps)

	return

if __name__ == '__main__':
	main()
















