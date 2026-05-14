from astropy.stats import sigma_clip
import matplotlib.pyplot as plt 
import numpy as np
import sys

sys.path.append('../src/')
import config

def combine_and_std(nights, cameras, model, RV, Kp, ITERS = 4, RV_lims = [225,-225], Kp_lims = [0,300], orders = None, sigma_cut = 3):

	indiv_results = []

	for camera in cameras:

		for night in nights:

			fmap = np.load(f"{config.path2reduced}results/{night}_{camera}_{model}_{ITERS}_iters_cgpt.npz")["fmap"]

			summed = -1. * np.sum(fmap, axis=0)
			summed = summed[Kp_lims[0]:Kp_lims[-1], RV_lims[0]:RV_lims[-1]]

			indiv_results.append(summed)

	all_summed = np.nansum(indiv_results, axis=0)
	all_summed = all_summed - np.nanmedian(all_summed)

	sigma_clipped = sigma_clip(all_summed, sigma_upper = sigma_cut, sigma_lower = 100)
	sigma_clipped_std = np.nanstd(sigma_clipped)

	fin = all_summed / sigma_clipped_std

	maximum = np.where(fin == np.nanmax(fin))

	max_RV = RV[RV_lims[0]:RV_lims[-1]][maximum[1][0]]
	max_Kp = Kp[Kp_lims[0]:Kp_lims[-1]][maximum[0][0]]

	return fin, [np.nanmax(fin), max_RV, max_Kp]

def plot_final(RV, Kp, final):

	fig = plt.figure()

	ax_map = fig.add_subplot(223)
	ax_map.pcolormesh(RV, Kp, final)
	ax_map.set_xlabel('RV [km/s]')
	ax_map.set_ylabel('Kp [km/s]')

	maximum = np.where(final == np.nanmax(final))

	ax_RV = fig.add_subplot(221)
	ax_RV.plot(RV, final[maximum[0][0], :])
	ax_RV.set_xlim(min(RV), max(RV))
	ax_RV.set_xticks([])
	ax_RV.set_ylabel('SNR')
	ax_RV.set_title(f'Max. RV at {RV[maximum[1][0]]} km/s')

	ax_Kp = fig.add_subplot(224)
	ax_Kp.plot(final[:,maximum[1][0]], Kp)
	ax_Kp.set_ylim(min(Kp), max(Kp))
	ax_Kp.set_yticks([])
	ax_Kp.set_xlabel('SNR')
	ax_Kp.set_title(f'Max. Kp at {Kp[maximum[0][0]]} km/s')

	plt.subplots_adjust(hspace=0.05, wspace=0.05)
	plt.show()

	return






















