""" 
Author: Jason Gabriel
Modified: Friday, June 19th
Description: Interactive slider plot to visualize plots, relative to your stage in the pipeline
"""

from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from pathlib import Path
import argparse
import sys
import os


# -- To run as an argument: -- #

#config_path = Path(sys.argv[1])
#sys.path.insert(0, str(config_path))

# add %matplotlib widget -- if running on jupter notebook


# -- Hardcoded Path -- #
config_path = Path("/Applications/SPECTRE-GHOST/wasp189b-summer2026/") ## CHANGE PATH TO YOUR CONFIG FILE LOCATION
sys.path.insert(0, str(config_path))

import config

redOrders = np.arange(config.redOrders)
blueOrders = np.arange(config.blueOrders)
sysremIters = config.sysremIters

camera = input("Which camera would you like to plot? (red/blue): ").strip().lower()
if camera == "red":
    orders = redOrders
elif camera == "blue":
    orders = blueOrders

# -- Path Availability -- #
if os.path.exists(config.path2reduced+config.nights[0]+'_'+camera+'.npz'):
    raw_file = np.load(config.path2reduced+config.nights[0]+'_'+camera+'.npz')
else:
    sys.exit("FAILED: raw files not found! Ensure the file is named correctly and downloaded!")


if os.path.exists(config.path2reduced+config.nights[0]+'_'+camera+'_analysis_ready.npz'):
    analysis_file = np.load(config.path2reduced+config.nights[0]+'_'+camera+'_analysis_ready.npz')
    has_analysis = True
else:
    has_analysis = False
    print("WARNING: file not found: proceeding without analysis file!")

if  os.path.exists(config.path2reduced+config.nights[0]+'_'+camera+'_sysrem.npz'):
    sysrem_file = np.load(config.path2reduced+config.nights[0]+'_'+camera+'_sysrem.npz')
    has_sysrem = True
else:
    print("WARNING: file not found: proceeding without sysrem file!")
    has_sysrem = False


# -- Data Loading -- #
order = orders.min()
file_selection = raw_file if not has_analysis else analysis_file

wave_data = file_selection['wave'][:, :, ::2].astype(np.float32, copy=False)
flux_data = file_selection['flux'][:, :, ::2].astype(np.float32, copy=False)

wvl = wave_data[order]
flux = flux_data[order]
phase = file_selection['phase']
vmax = np.nanpercentile(flux, 90) 
plt.rcParams['font.family'] = 'EB Garamond'

fig, ax = plt.subplots(1, 2, figsize=(10, 8))
plt.subplots_adjust(bottom=0.13)

order_slider_ax = plt.axes([0.2, 0.04, 0.6, 0.03])

ts_spectra = ax[0].imshow(
    flux,
    aspect='auto',
    origin='lower',
    extent=[wvl.min(), wvl.max(),
    phase.min(), phase.max()],
    vmax = vmax)

ax[0].set_title(f"Time-Series Spectra, Order {order}", fontsize=14)
ax[0].set_ylabel("Orbital Phase")
ax[0].set_xlabel("Wavelength [nm]") 


if has_sysrem:
    iteration_slider_ax = plt.axes([0.58, 0.01, 0.25, 0.03])
    n_iterations = 1

    sysrem_data = sysrem_file['sysrem'][:, :, :, ::2].astype(np.float32, copy=False)
    sys = sysrem_data[n_iterations][order]
    sys_vmax = np.nanpercentile(sysrem_data, 85)

    sys_spectra = ax[1].imshow(
        sys,
        aspect='auto',
        origin='lower',
        extent=[wvl.min(), wvl.max(),
        phase.min(), phase.max()],
        vmax = sys_vmax)

    ax[1].set_title(f"SYSREM Iteration {n_iterations}, Order {order}", fontsize=14)
    ax[1].set_ylabel("Orbital Phase")
    ax[1].set_xlabel("Wavelength [nm]")

    # Slider Updating #
    def update_iteration(val):
        global n_iterations
        n_iterations = int(val)
        new_sys = sysrem_data[n_iterations][order]
        vmax = np.nanpercentile(new_sys, 85)

        sys_spectra.set_data(new_sys)
        sys_spectra.set_clim(vmax=vmax)

        ax[1].set_title(f"SYSREM {n_iterations}, Order {order}", fontsize=14)
        ax[1].set_ylabel("Orbital Phase")
        ax[1].set_xlabel("Wavelength [nm]")
        fig.canvas.draw_idle()

    iteration_slider = matplotlib.widgets.Slider(iteration_slider_ax, 'Iteration Selection', 0, sysremIters, valinit=1, valfmt='%0.0f', dragging=False, valstep=1)
    iteration_slider.on_changed(update_iteration)
    
# Slider Updating #
def update_order(val):
    global order
    order = int(val)
    
    new_flux = flux_data[order]
    new_wvl = wave_data[order]
    
    vmax = np.nanpercentile(new_flux, 90)
    
    if has_sysrem:
        new_sys = sysrem_data[n_iterations][order]
        sys_vmax = np.nanpercentile(new_sys, 85)

        sys_spectra.set_data(new_sys)
        sys_spectra.set_clim(vmax=sys_vmax)
        sys_spectra.set_extent([
        new_wvl.min(),
        new_wvl.max(),
        phase.min(),
        phase.max()])
        
        ax[1].set_title(f"SYSREM Iteration {n_iterations}, Order {order}", fontsize=14)
        ax[1].set_ylabel("Orbital Phase")
        ax[1].set_xlabel("Wavelength [nm]")
        
    ts_spectra.set_data(new_flux)
    ts_spectra.set_clim(vmax=vmax)
    ts_spectra.set_extent([
        new_wvl.min(),
        new_wvl.max(),
        phase.min(),
        phase.max()])

    ax[0].set_title(f"Time-Series Spectra, Order {order}", fontsize=14)
    ax[0].set_ylabel("Orbital Phase")
    ax[0].set_xlabel("Wavelength [nm]")   
    fig.canvas.draw_idle()


order_slider = matplotlib.widgets.Slider(order_slider_ax, 'Order Selection', orders.min(), orders.max(), valinit=orders.min(), valfmt='%0.0f', dragging=False, valstep=1)
order_slider.on_changed(update_order)

plt.show()