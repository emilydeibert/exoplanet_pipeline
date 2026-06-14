from scipy.stats import binned_statistic
from astropy import units as u 
from astropy import constants as cs 
import matplotlib.pyplot as plt
import numpy as np
import sys

from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS import physical_constants as cst
from petitRADTRANS import physics as phys

fastChemDict = {}
fastChemDict['Fe'] = 'Fe'
fastChemDict['Fe+'] = 'Fe1'
fastChemDict['Ti'] = 'Ti'
fastChemDict['Ti+'] = 'Ti1'
fastChemDict['TiO'] = 'O1Ti1'
fastChemDict['V'] = 'V'
fastChemDict['V+'] = 'V1'
fastChemDict['VO'] = 'O1V1'
fastChemDict['Ca'] = 'Ca'
fastChemDict['Ca+'] = 'Ca1'
fastChemDict['Cr'] = 'Cr'
fastChemDict['FeH'] = 'Fe1H1'
fastChemDict['K'] = 'K'
fastChemDict['OH'] = 'H1O1'
fastChemDict['Mg'] = 'Mg'
fastChemDict['Si'] = 'Si'
fastChemDict['Al'] = 'Al'
fastChemDict['CaH'] = 'Ca1H1'
fastChemDict['Na'] = 'Na'

speciesWeight = {}
speciesWeight['Fe'] = 55.845
speciesWeight['Ti'] = 47.867
speciesWeight['Fe+'] = 55.845
speciesWeight['TiO'] = 63.866
speciesWeight['V'] = 50.9415
speciesWeight['V+'] = 50.9415
speciesWeight['VO'] = 181.88
speciesWeight['Ca'] = 40.078
speciesWeight['Ca+'] = 40.078
speciesWeight['Cr'] = 51.9961
speciesWeight['FeH'] = 56.853
speciesWeight['K'] = 39.0983
speciesWeight['OH'] = 17.007
speciesWeight['Mg'] = 24.305
speciesWeight['Si'] = 28.0855
speciesWeight['Al'] = 26.981539
speciesWeight['CaH'] = 41.09
speciesWeight['Na'] = 22.989769

petitDict = {}
petitDict['Fe'] = 'Fe'
petitDict['Ti'] = 'Ti'
petitDict['Fe+'] = 'Fe+'
petitDict['TiO'] = 'TiO'#_48_Exomol_McKemmish'
petitDict['V'] = 'V'
petitDict['V+'] = 'V_+'
petitDict['VO'] = 'VO'#_ExoMol_McKemmish'
petitDict['Ca'] = 'Ca'
petitDict['Ca+'] = 'Ca+'
petitDict['Cr'] = 'Cr'
petitDict['FeH'] = 'FeH'#_main_iso'
petitDict['K'] = 'K'
petitDict['OH'] = 'OH'#_main_iso'
petitDict['Mg'] = 'Mg'
petitDict['Si'] = 'Si'
petitDict['Al'] = 'Al'
petitDict['CaH'] = 'CaH'
petitDict['Mg+'] = 'Mg+'
petitDict['Na'] = 'Na'

vac2airDict = {}
vac2airDict['Fe'] = True
vac2airDict['TiO'] = True
vac2airDict['Ti'] = True
vac2airDict['Ti+'] = False
vac2airDict['Fe+'] = False
vac2airDict['V'] = True
vac2airDict['V+'] = True
vac2airDict['VO'] = True
vac2airDict['Ca'] = True
vac2airDict['Ca+'] = True
vac2airDict['Cr'] = True
vac2airDict['FeH'] = True
vac2airDict['K'] = True
vac2airDict['OH'] = True
vac2airDict['Mg'] = True
vac2airDict['Si'] = True
vac2airDict['Al'] = True
vac2airDict['CaH'] = True 
vac2airDict['Na'] = True

### CHANGE HERE ###
T1 = 8100
P1 = 10**(-8)
T2 = 2200
P2 = 10**(-2)
M_pl = 1.41 * u.M_jupiter
R_pl = 1.940 * u.R_jupiter

Tstar = 9360 * u.Kelvin
Rstar = 1.67 * u.R_sun
### CHANGE HERE ###

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

def T_p(T1, P1, T2, P2, pressures = np.logspace(-6, 0, 100)):
	""" Based on the two-point T-p profile.
	"""
	temperatures = np.zeros(np.shape(pressures))
	Tslope = (np.log10(P2) - np.log10(P1)) / (T2 - T1)
	b = np.log10(P2) - Tslope * T2
	for idx, p in enumerate(pressures):
		if p < P1:
			temperatures[idx] = T1
		elif p > P2:
			temperatures[idx] = T2
		else:
			temperatures[idx] = (np.log10(p) - b) / Tslope
	return pressures, temperatures

def grav(M_pl, R_pl):
	g = (cs.G * M_pl) / (R_pl)**2
	val = np.log10(g.decompose().to(u.cm / u.s**2).value)
	gravity = 1e1**val
	return gravity

def remove_env(wave, spec, px, order):
	""" Tool to remove the lower envelope of a model.
	Created by Miranda Herman. 
	"""
	binned = binned_statistic(wave, spec, statistic='min', bins=px)
	bin_mids = binned[1][1:] - (binned[1][1] - binned[1][0])/2
	fit = np.polyfit(bin_mids, binned[0], order)
	env = np.polyval(fit, wave)
	return spec - env

def massFractions(species, temperature, mmw_val = 2.33, fastchem=True):

	if fastchem:

		#if species in ['Ti', 'Ti+']:
		#	chemistry = np.genfromtxt('chemistry_select_titiplus.dat', delimiter='\t', names=True)
		#elif species in ['Al', 'CaH', 'Mg+']:
		#	chemistry = np.genfromtxt('chemistry.dat', delimiter='\t', names=True)
		#else:
		chemistry = np.genfromtxt('chemistry.dat', delimiter='\t', names=True)

		VMR = chemistry[fastChemDict[species]]

		if mmw_val == 'FastChem':
			MMW = chemistry['m_gmol']
		else:
			MMW = mmw_val * np.ones_like(temperature)

		if species in ['Fe+', 'OH', 'V+', 'Si']:
			X = (speciesWeight[species] / MMW) * VMR
		else:
			X = (speciesWeight[species] / MMW) * VMR#[-1] * np.ones(len(VMR))

		mass_fractions = {}
		mass_fractions['H2'] = 0.74 * np.ones_like(temperature)
		mass_fractions['He'] = 0.24 * np.ones_like(temperature)
		mass_fractions[petitDict[species]] = X

	return mass_fractions, MMW, VMR

def generate_atmosphere(species, pressures, temperatures, mass_fractions, MMW, gravity):

	atmosphere = Radtrans(
		pressures = pressures,
		line_species = [petitDict[species]],
		rayleigh_species = ['H2', 'He'],
		gas_continuum_contributors = ['H2-H2', 'H2-He'],
		wavelength_boundaries = [0.3, 1.08],
		line_opacity_mode='lbl')

	frequencies, flux, _ = atmosphere.calculate_flux(
		temperatures=temperatures,
		mass_fractions=mass_fractions,
		mean_molar_masses = MMW,
		reference_gravity = gravity,
		frequencies_to_wavelengths=False)

	wvl = (cst.c/frequencies/1e-4) * u.micron
	flux = (flux/1e-6) * 10**(-6) * u.erg * u.cm**(-2) * u.s**(-1) * u.Hz**(-1)

	return wvl, flux

def star(wvl, temp = Tstar):

	wvl_in_cm = wvl.to(u.cm).value
	planck = phys.planck_function_hz(temp.value, phys.wavelength2frequency(wvl_in_cm))
	return planck

def generateModel(flux, wvl, planck, vac_to_air, R_star = Rstar, removeEnv = True):

	spec = (flux.value) / (planck * R_star.to(u.R_jupiter).value ** 2)
	if removeEnv:
		depth = remove_env(wvl.value, spec, 400, 4)
	else:
		depth = spec

	if vac_to_air:
		wvl = vac2air(wvl.to(u.AA).value)
	else:
		wvl = wvl.to(u.AA).value

	mld = np.column_stack((wvl.transpose(), depth.transpose()))

	return mld

def plotAll(pressures, temperature, VMR, mld, species):

	# plot VMR and T-p profile
	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	ax1.plot(VMR, pressures)
	ax1.set_xscale('log')
	ax1.set_yscale('log')
	ax1.invert_yaxis()
	ax1.set_xlabel('VMR')
	ax1.set_ylabel('Pressure')
	ax1.set_title(species)

	ax2 = fig.add_subplot(122)
	ax2.plot(temperature, pressures)
	ax2.set_yscale('log')
	ax2.invert_yaxis()
	ax2.set_xlabel('Temperature')
	ax2.set_ylabel('Pressure')
	ax2.set_title('T-p profile')
	plt.show()

	fig2 = plt.figure()
	ax = fig2.add_subplot(111)
	ax.plot(mld[:,0], 1 + mld[:,1])
	ax.set_xlabel('Wavelength [Angstrom]')
	ax.set_title(species)
	ax.set_xlim(min(mld[:,0]), max(mld[:,0]))
	ax.set_ylabel('1 + Fp/Fstar')
	plt.show()

	return

def main():

	species = sys.argv[1]

	pressures, temperatures = T_p(T1 = T1, P1 = P1, T2 = T2, P2 = P2)
	mass_fractions, MMW, VMR = massFractions(species, temperatures)
	gravity = grav(M_pl, R_pl)
	wvl, flux = generate_atmosphere(species, pressures, temperatures, mass_fractions, MMW, gravity)
	planck = star(wvl)

	mld = generateModel(flux, wvl, planck, vac2airDict[species])#, removeEnv=False)
	mld_without = generateModel(flux, wvl, planck, vac2airDict[species], removeEnv=False)

	mld.dump('./wasp178b/'+species+'_model.npy') #### CHANGE FOLDER
	mld_without.dump('./wasp178b/'+species+'_model_withContinuum.npy') #### CHANGE FOLDER
	print(species)

	if len(sys.argv) > 2:

		if sys.argv[2] == 'plot':
			plotAll(pressures, temperatures, VMR, mld, species)
	return


if __name__ == '__main__':
	main()














