# read the CS FIle From Matlab Code
#
#
# Format:
# %%

import scipy.io as sio
import numpy as np
import pandas as pd
import IPP_Cal
from os.path import exists



# Define the Station Location
latR = 22.29584206/180*np.pi
lonR = 113.9077209/180*np.pi


if exists('eleBDS.mat') and exists('aziBDS.mat'):
	# %% for BDS
	# load the matlab file gives out the ele and azi
	elemat_BDS = sio.loadmat('eleBDS.mat')
	azimat_BDS = sio.loadmat('aziBDS.mat')

	# Note.
	# Dict structure from loadmat
	# _globals_, _header_, _version_, "varible"
	
	ele_BDS = elemat_BDS['sat_pos_Ele']
	azi_BDS = azimat_BDS['sat_pos_Azi']
	
	# load the hdf file gives out the CS information
	cs_BDS_HDF = pd.HDFStore('BDS_CS.h5')
	if cs_BDS_HDF.info().find('Empty') is -1:
		cs_BDS_HDF.close()
		# get the ele_BDS and azi_BDS size
		cs_BDS = pd.read_hdf('BDS_CS.h5','dt',mode='r+')
		if ele_BDS.shape != azi_BDS.shape:
			print('Warning the shape of the elevation and azimuth is not consistent!')
		else:
			# create two zero array with the same shape of the ele/azi array, one for L1 AND the other for L2
			cs_L1_BDS = np.zeros(ele_BDS.shape)
			cs_L2_BDS = np.zeros(ele_BDS.shape)
			# index every row/record and insert the cs size to the cs_L1_BDS and cs_L2_BDS varibles
			for row in cs_BDS.itertuples(name=None):
				Epoch = row[8]
				PRN   = row[7]
				CS_L1 = row[10]
				CS_L2 = row[11]
				# insert manipulation
				cs_L1_BDS[Epoch-1,PRN-1] = CS_L1
				cs_L2_BDS[Epoch-1,PRN-1] = CS_L2
	else:
		# get the ele_BDS and azi_BDS size
		if ele_BDS.shape != azi_BDS.shape:
			print('Warning the shape of the elevation and azimuth is not consistent!')
		else:
			# create two zero array with the same shape of the ele/azi array, one for L1 AND the other for L2
			cs_L1_BDS = np.zeros(ele_BDS.shape)
			cs_L2_BDS = np.zeros(ele_BDS.shape)
	
	# %% calculate the IPP coordinate for all the visible satellites
	latipp = np.zeros(ele_BDS.shape)
	lonipp = np.zeros(ele_BDS.shape)
	fpp_BDS    = np.zeros(ele_BDS.shape)
	
	for x,y,p,q,f in np.nditer([ele_BDS,azi_BDS,latipp,lonipp,fpp_BDS], op_flags = ['readwrite']):
		p[...],q[...],f[...] = IPP_Cal.IPP_Cal(latR,lonR,y/180*np.pi,x/180*np.pi)
	
	# rad2deg
	latipp_BDS = np.rad2deg(latipp)
	lonipp_BDS = np.rad2deg(lonipp)
	
	Data_Matrix_BDS = np.dstack((ele_BDS,azi_BDS,latipp_BDS,lonipp_BDS,fpp_BDS,cs_L1_BDS,cs_L2_BDS))
	print('BDS finished...')


if exists('eleGLO.mat') and exists('aziGLO.mat'):
	# %% for GLO
	
	# load the matlab file gives out the ele and azi
	elemat_GLO = sio.loadmat('eleGLO.mat')
	azimat_GLO = sio.loadmat('aziGLO.mat')

	# Note.
	# Dict structure from loadmat
	# _globals_, _header_, _version_, "varible"
	
	ele_GLO = elemat_GLO['sat_pos_Ele']
	azi_GLO = azimat_GLO['sat_pos_Azi']
	# load the hdf file gives out the CS information
	cs_GLO_HDF = pd.HDFStore('GLO_CS.h5')
	if cs_GLO_HDF.info().find('Empty') is -1:
		cs_GLO_HDF.close()
		# get the ele_GLO and azi_GLO size
		cs_GLO = pd.read_hdf('GLO_CS.h5','dt',mode='r+')
		if ele_GLO.shape != azi_GLO.shape:
			print('Warning the shape of the elevation and azimuth is not consistent!')
		else:
			# create two zero array with the same shape of the ele/azi array, one for L1 AND the other for L2
			cs_L1_GLO = np.zeros(ele_GLO.shape)
			cs_L2_GLO = np.zeros(ele_GLO.shape)
			# index every row/record and insert the cs size to the cs_L1_GLO and cs_L2_GLO varibles
			for row in cs_GLO.itertuples(name=None):
				Epoch = row[8]
				PRN   = row[7]
				CS_L1 = row[10]
				CS_L2 = row[11]
				# insert manipulation
				cs_L1_GLO[Epoch-1,PRN-1] = CS_L1
				cs_L2_GLO[Epoch-1,PRN-1] = CS_L2
	else:
		# get the ele_GLO and azi_GLO size
		if ele_GLO.shape != azi_GLO.shape:
			print('Warning the shape of the elevation and azimuth is not consistent!')
		else:
			# create two zero array with the same shape of the ele/azi array, one for L1 AND the other for L2
			cs_L1_GLO = np.zeros(ele_GLO.shape)
			cs_L2_GLO = np.zeros(ele_GLO.shape)
	
	# %% calculate the IPP coordinate for all the visible satellites
	latipp = np.zeros(ele_GLO.shape)
	lonipp = np.zeros(ele_GLO.shape)
	fpp_GLO    = np.zeros(ele_GLO.shape)
	
	for x,y,p,q,f in np.nditer([ele_GLO,azi_GLO,latipp,lonipp,fpp_GLO], op_flags = ['readwrite']):
		p[...],q[...],f[...] = IPP_Cal.IPP_Cal(latR,lonR,y/180*np.pi,x/180*np.pi)
	
	# rad2deg
	latipp_GLO = np.rad2deg(latipp)
	lonipp_GLO = np.rad2deg(lonipp)
	
	Data_Matrix_GLO = np.dstack((ele_GLO,azi_GLO,latipp_GLO,lonipp_GLO,fpp_GLO,cs_L1_GLO,cs_L2_GLO))
			
	print('GLO finished...')

if exists('eleGPS.mat') and exists('aziGPS.mat'):
	# %% for GPS
	
	# load the matlab file gives out the ele and azi
	elemat_GPS = sio.loadmat('eleGPS.mat')
	azimat_GPS = sio.loadmat('aziGPS.mat')

	# Note.
	# Dict structure from loadmat
	# _globals_, _header_, _version_, "varible"
	
	ele_GPS = elemat_GPS['sat_pos_Ele']
	azi_GPS = azimat_GPS['sat_pos_Azi']
	# load the hdf file gives out the CS information
	cs_GPS_HDF = pd.HDFStore('GPS_CS.h5')
	if cs_GPS_HDF.info().find('Empty') is -1:
		cs_GPS_HDF.close()
		# get the ele_GPS and azi_GPS size
		cs_GPS = pd.read_hdf('GPS_CS.h5','dt',mode='r+')
		if ele_GPS.shape != azi_GPS.shape:
			print('Warning the shape of the elevation and azimuth is not consistent!')
		else:
			# create two zero array with the same shape of the ele/azi array, one for L1 AND the other for L2
			cs_L1_GPS = np.zeros(ele_GPS.shape)
			cs_L2_GPS = np.zeros(ele_GPS.shape)
			# index every row/record and insert the cs size to the cs_L1_GPS and cs_L2_GPS varibles
			for row in cs_GPS.itertuples(name=None):
				Epoch = row[8]
				PRN   = row[7]
				CS_L1 = row[10]
				CS_L2 = row[11]
				# insert manipulation
				cs_L1_GPS[Epoch-1,PRN-1] = CS_L1
				cs_L2_GPS[Epoch-1,PRN-1] = CS_L2
	else:
		# get the ele_GPS and azi_GPS size
		if ele_GPS.shape != azi_GPS.shape:
			print('Warning the shape of the elevation and azimuth is not consistent!')
		else:
			# create two zero array with the same shape of the ele/azi array, one for L1 AND the other for L2
			cs_L1_GPS = np.zeros(ele_GPS.shape)
			cs_L2_GPS = np.zeros(ele_GPS.shape)

	
	# %% calculate the IPP coordinate for all the visible satellites
	latipp = np.zeros(ele_GPS.shape)
	lonipp = np.zeros(ele_GPS.shape)
	fpp_GPS    = np.zeros(ele_GPS.shape)
	
	for x,y,p,q,f in np.nditer([ele_GPS,azi_GPS,latipp,lonipp,fpp_GPS], op_flags = ['readwrite']):
		p[...],q[...],f[...] = IPP_Cal.IPP_Cal(latR,lonR,y/180*np.pi,x/180*np.pi)
	
	# rad2deg
	latipp_GPS = np.rad2deg(latipp)
	lonipp_GPS = np.rad2deg(lonipp)
	
	Data_Matrix_GPS = np.dstack((ele_GPS,azi_GPS,latipp_GPS,lonipp_GPS,fpp_GPS,cs_L1_GPS,cs_L2_GPS))		
	print('GPS finished...')

# %% calculate the IPP coordinate for all the visible satellites

# %% save the combination matrix of GPS,GLO,BDS
sio.savemat('el_az_ipp_cs.mat',{"Matrix_GPS":Data_Matrix_GPS,"Matrix_GLO":Data_Matrix_GLO})
