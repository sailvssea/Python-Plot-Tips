# Computation of the ionosphere piercing point (IPP)

# Input :
#        latR = receiver position latitude  [rad]
#        lonR = receiver position longitude [rad]
#        azS  = satellite azimuth [rad]
#        elS  = satellite elevation [rad]
# Output :
#        latpp = ionosphere piercing point latitude  [rad]
#        lonpp = ionosphere piercing point longitude [rad]
#        fpp   = slant factor (mapping function)

import numpy as np


def IPP_Cal(latR, lonR, azS, elS):

	if elS <= 10./180*np.pi:   #  cutoff angle mask
		latpp = np.nan
		lonpp = np.nan
		fpp   = np.nan
		return latpp, lonpp, fpp
		
	# define the earth radius [km]
	R = 6378.1363
	# define the ionosphere thin shell height [km]
	hI = 350.

	k = (R/(R+hI))*np.cos(elS)
	phipp = (np.pi/2) - elS - np.arcsin(k)

	# set azimuth from -180 to 180
	#azS = np.mod((azS+np.pi), 2.*np.pi) - np.pi

	# latitude of the ionosphere piercing point
	latpp = np.arcsin(np.sin(latR)*np.cos(phipp) + np.cos(latR)*np.sin(phipp)*np.cos(azS))

	# longitude of the ionosphere piercing point
	#if ((latpp > 70*np.pi/180) & (np.tan(phipp)*np.cos(azS) > np.tan((np.pi/2) - latR))) |\
	#	((latpp < -70*np.pi/180) & (np.tan(phipp)*np.cos(azS + np.pi) > np.tan((np.pi/2) + latR))):

	#	lonpp = lonR + np.pi - np.arcsin(np.sin(phipp)*np.sin(azS/np.cos(latpp)))

	#else:
	lonpp = lonR + np.arcsin(np.sin(phipp)*np.sin(azS)/np.cos(latpp))

	# slant (obliquity) factor
	fpp = (1-(k**2))**(-1/2)

	return latpp, lonpp, fpp
