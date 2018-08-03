# plot the el_az_ipp_cs file on the map from the EleAziCS_combine.py
#
# el_az_ipp_cs 
#      depth (axis 2) : #1 ele #2 azi #3 latipp 
#                       #4 lonipp #5 fppipp #6 cs1 #7 cs2
# 
# Format:
# %%

import scipy.io as sio
from mpl_toolkits.basemap import Basemap # Note: the Basemap from mpl_toolkits is only supported by the Python 2.6 or less.
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib
import numpy as np
import pandas as pd

# load the el_az_ipp_cs
ELAZIPPCS = sio.loadmat('el_az_ipp_cs.mat')

# BDS
#el_az_ipp_cs_BDS = ELAZIPPCS['Matrix_BDS']
# GLO
el_az_ipp_cs_GLO = ELAZIPPCS['Matrix_GLO']
# GPS
el_az_ipp_cs_GPS = ELAZIPPCS['Matrix_GPS']


# configure the figure/plot setting/environment
# Create a fixed sized figure (width, height in inches)
fig = plt.figure(figsize=(9.0,8.0))  # create the figure with the fix size
fig.subplots_adjust(left= 0.10, right= 0.95,\
					bottom= 0.05, top= 0.95,\
					wspace= 0.2, hspace= 0.2)


# %% GPS L1
ax1 = fig.add_subplot(221)
# setup Lambert Conformal basemap.
m = Basemap(llcrnrlon=100.,llcrnrlat=10.,urcrnrlon=135.,urcrnrlat=32,\
			 projection='lcc',resolution='l',lon_0=118.,lat_0=22.5,lat_ts=20.)

# draw coastlines.
#m.drawcoastlines()
# draw a boundary around the map, fill the background.
# this background will end up being the ocean color, since
# the continents will be drawn on top.
m.drawmapboundary()
# fill continents, set lake color same as ocean color.
m.fillcontinents(zorder=0) # on the background layer position ( zorder = 0 means the background/first/bottom layer)
# draw parallels and meridians.
# label parallels on right and top
# meridians on bottom and left
# labels = [left,right,top,bottom]
parallels = np.arange(10.,31.,5.)
m.drawparallels(parallels,labels=[1,0,0,0])
meridians = np.arange(100.,131.,10.)
m.drawmeridians(meridians,labels=[0,0,1,0])
# plot Hong Kong Location
lon, lat =  114.175, 22.292
# convert to map projection coords.
# Note that lon,lat can be scalars, lists or numpy arrays
xpt,ypt = m(lon,lat)
# convert back to lat/lon
lonpt, latpt = m(xpt,ypt,inverse=True)
# Define the new marker for the path
star = mpath.Path.unit_regular_star(5)
circle = mpath.Path.unit_circle()
# concatenate the circle with an internal cutout of the star
verts = np.concatenate([circle.vertices, star.vertices[::-1, ...]])
codes = np.concatenate([circle.codes, star.codes])
cut_star = mpath.Path(verts, codes)
m.plot(xpt,ypt,marker=cut_star,markersize=12,mfc='r',mec='r') #plot a r dot there
# put some text next to the dot, offset a little bit
# (the offset is in map projection coordinates)
#plt.text(xpt+50000,ypt+50000,'Hong Kong')

# plot the satellite and its cs records
for i in np.arange(31):
	latIPP = el_az_ipp_cs_GPS[:,i,2] # latIPP
	lonIPP = el_az_ipp_cs_GPS[:,i,3] # lonIPP
	cs_L1  = np.absolute(el_az_ipp_cs_GPS[:,i,5]) # cs on L1
	cs_L2  = np.absolute(el_az_ipp_cs_GPS[:,i,6]) # cs on L2
	# mask data
	latIPP_cs = np.ma.masked_where(cs_L1 == 0, latIPP)
	latIPP_no_cs = np.ma.masked_where(cs_L1 != 0, latIPP)
	lonIPP_cs = np.ma.masked_where(cs_L1 == 0, lonIPP)
	lonIPP_no_cs = np.ma.masked_where(cs_L1 != 0, lonIPP)
	m.scatter(lonIPP_cs,latIPP_cs, s = 40, c = 'r', marker = '^', alpha = 1.0, zorder = 2,latlon = True)
	m.scatter(lonIPP_no_cs,latIPP_no_cs, s = 5, c = 'b', marker = '.', alpha = 1.0, zorder = 1,latlon = True)

plt.text(0.88, 0.09,'GPS L1', fontsize = 12, fontweight = 'bold', ha = 'center', va = 'center', transform = ax1.transAxes)
# %%
# Plot the TC trajectory
latlon_TC = pd.read_csv('TEMBIN.TC',usecols=['Time','Latitude','Longitude'])
lon_TC = np.array(latlon_TC['Longitude'])
lat_TC = np.array(latlon_TC['Latitude'])
m.plot(lon_TC,lat_TC,linestyle='-',linewidth= 2,color = 'r', zorder = 2, latlon=True)
# scatter plot point for every record
# color bounds map for time stamp
m.scatter(lon_TC[11],lat_TC[11],marker= 'o',s= 300,c= 'c',alpha = 0.8,zorder=3,latlon=True)


# %% GPS L2

ax2 = fig.add_subplot(222)
# setup Lambert Conformal basemap.
m = Basemap(llcrnrlon=100.,llcrnrlat=10.,urcrnrlon=135.,urcrnrlat=32,\
			 projection='lcc',resolution='l',lon_0=118.,lat_0=22.5,lat_ts=20.)

# draw coastlines.
#m.drawcoastlines()
# draw a boundary around the map, fill the background.
# this background will end up being the ocean color, since
# the continents will be drawn on top.
m.drawmapboundary()
# fill continents, set lake color same as ocean color.
m.fillcontinents(zorder=0) # on the background layer position ( zorder = 0 means the background/first/bottom layer)
# draw parallels and meridians.
# label parallels on right and top
# meridians on bottom and left
# labels = [left,right,top,bottom]
parallels = np.arange(10.,31.,5.)
m.drawparallels(parallels,labels=[1,0,0,0])
meridians = np.arange(100.,131.,10.)
m.drawmeridians(meridians,labels=[0,0,1,0])
# plot Hong Kong Location
lon, lat =  114.175, 22.292
# convert to map projection coords.
# Note that lon,lat can be scalars, lists or numpy arrays
xpt,ypt = m(lon,lat)
# convert back to lat/lon
lonpt, latpt = m(xpt,ypt,inverse=True)
# Define the new marker for the path
star = mpath.Path.unit_regular_star(5)
circle = mpath.Path.unit_circle()
# concatenate the circle with an internal cutout of the star
verts = np.concatenate([circle.vertices, star.vertices[::-1, ...]])
codes = np.concatenate([circle.codes, star.codes])
cut_star = mpath.Path(verts, codes)
m.plot(xpt,ypt,marker=cut_star,markersize=12,mfc='r',mec='r') #plot a r dot there
# put some text next to the dot, offset a little bit
# (the offset is in map projection coordinates)
#plt.text(xpt+50000,ypt+50000,'Hong Kong')

# plot the satellite and its cs records
for i in np.arange(31):
	latIPP = el_az_ipp_cs_GPS[:,i,2] # latIPP
	lonIPP = el_az_ipp_cs_GPS[:,i,3] # lonIPP
	cs_L1  = np.absolute(el_az_ipp_cs_GPS[:,i,5]) # cs on L1
	cs_L2  = np.absolute(el_az_ipp_cs_GPS[:,i,6]) # cs on L2
	# mask data
	latIPP_cs = np.ma.masked_where(cs_L2 == 0, latIPP)
	latIPP_no_cs = np.ma.masked_where(cs_L2 != 0, latIPP)
	lonIPP_cs = np.ma.masked_where(cs_L2 == 0, lonIPP)
	lonIPP_no_cs = np.ma.masked_where(cs_L2 != 0, lonIPP)
	m.scatter(lonIPP_cs,latIPP_cs, s = 40, c = 'r', marker = '^', alpha = 1.0, zorder = 2,latlon = True)
	m.scatter(lonIPP_no_cs,latIPP_no_cs, s = 5, c = 'b', marker = '.', alpha = 1.0, zorder = 1,latlon = True)

plt.text(0.88, 0.09,'GPS L2', fontsize = 12, fontweight = 'bold', ha = 'center', va = 'center', transform = ax2.transAxes)
# Plot the TC trajectory
latlon_TC = pd.read_csv('TEMBIN.TC',usecols=['Time','Latitude','Longitude'])
lon_TC = np.array(latlon_TC['Longitude'])
lat_TC = np.array(latlon_TC['Latitude'])
m.plot(lon_TC,lat_TC,linestyle='-',linewidth= 2,color = 'r', zorder = 2, latlon=True)
# scatter plot point for every record
# color bounds map for time stamp
m.scatter(lon_TC[11],lat_TC[11],marker= 'o',s= 300,c= 'c',alpha = 0.8,zorder=3,latlon=True)

# %% GLO L1

ax3 = fig.add_subplot(223)
# setup Lambert Conformal basemap.
m = Basemap(llcrnrlon=100.,llcrnrlat=10.,urcrnrlon=135.,urcrnrlat=32,\
			 projection='lcc',resolution='l',lon_0=118.,lat_0=22.5,lat_ts=20.)

# draw coastlines.
#m.drawcoastlines()
# draw a boundary around the map, fill the background.
# this background will end up being the ocean color, since
# the continents will be drawn on top.
m.drawmapboundary()
# fill continents, set lake color same as ocean color.
m.fillcontinents(zorder=0) # on the background layer position ( zorder = 0 means the background/first/bottom layer)
# draw parallels and meridians.
# label parallels on right and top
# meridians on bottom and left
# labels = [left,right,top,bottom]
parallels = np.arange(10.,31.,5.)
m.drawparallels(parallels,labels=[1,0,0,0])
meridians = np.arange(100.,131.,10.)
m.drawmeridians(meridians,labels=[0,0,1,0])
# plot Hong Kong Location
lon, lat =  114.175, 22.292
# convert to map projection coords.
# Note that lon,lat can be scalars, lists or numpy arrays
xpt,ypt = m(lon,lat)
# convert back to lat/lon
lonpt, latpt = m(xpt,ypt,inverse=True)
# Define the new marker for the path
star = mpath.Path.unit_regular_star(5)
circle = mpath.Path.unit_circle()
# concatenate the circle with an internal cutout of the star
verts = np.concatenate([circle.vertices, star.vertices[::-1, ...]])
codes = np.concatenate([circle.codes, star.codes])
cut_star = mpath.Path(verts, codes)
m.plot(xpt,ypt,marker=cut_star,markersize=12,mfc='r',mec='r') #plot a r dot there
# put some text next to the dot, offset a little bit
# (the offset is in map projection coordinates)
#plt.text(xpt+50000,ypt+50000,'Hong Kong')

# plot the satellite and its cs records
for i in np.arange(31):
	latIPP = el_az_ipp_cs_GLO[:,i,2] # latIPP
	lonIPP = el_az_ipp_cs_GLO[:,i,3] # lonIPP
	cs_L1  = np.absolute(el_az_ipp_cs_GLO[:,i,5]) # cs on L1
	cs_L2  = np.absolute(el_az_ipp_cs_GLO[:,i,6]) # cs on L2
	# mask data
	latIPP_cs = np.ma.masked_where(cs_L1 == 0, latIPP)
	latIPP_no_cs = np.ma.masked_where(cs_L1 != 0, latIPP)
	lonIPP_cs = np.ma.masked_where(cs_L1 == 0, lonIPP)
	lonIPP_no_cs = np.ma.masked_where(cs_L1 != 0, lonIPP)
	m.scatter(lonIPP_cs,latIPP_cs, s = 40, c = 'r', marker = '^', alpha = 1.0, zorder = 2,latlon = True)
	m.scatter(lonIPP_no_cs,latIPP_no_cs, s = 5, c = 'b', marker = '.', alpha = 1.0, zorder = 1,latlon = True)

plt.text(0.88, 0.09,'GLO L1', fontsize = 12, fontweight = 'bold', ha = 'center', va = 'center', transform = ax3.transAxes)
# Plot the TC trajectory
latlon_TC = pd.read_csv('TEMBIN.TC',usecols=['Time','Latitude','Longitude'])
lon_TC = np.array(latlon_TC['Longitude'])
lat_TC = np.array(latlon_TC['Latitude'])
m.plot(lon_TC,lat_TC,linestyle='-',linewidth= 2,color = 'r', zorder = 2, latlon=True)
# scatter plot point for every record
# color bounds map for time stamp
m.scatter(lon_TC[11],lat_TC[11],marker= 'o',s= 300,c= 'c',alpha = 0.8,zorder=3,latlon=True)
# %% GLO L2

ax4 = fig.add_subplot(224)
# setup Lambert Conformal basemap.
m = Basemap(llcrnrlon=100.,llcrnrlat=10.,urcrnrlon=135.,urcrnrlat=32,\
			 projection='lcc',resolution='l',lon_0=118.,lat_0=22.5,lat_ts=20.)

# draw coastlines.
#m.drawcoastlines()
# draw a boundary around the map, fill the background.
# this background will end up being the ocean color, since
# the continents will be drawn on top.
m.drawmapboundary()
# fill continents, set lake color same as ocean color.
m.fillcontinents(zorder=0) # on the background layer position ( zorder = 0 means the background/first/bottom layer)
# draw parallels and meridians.
# label parallels on right and top
# meridians on bottom and left
# labels = [left,right,top,bottom]
parallels = np.arange(10.,31.,5.)
m.drawparallels(parallels,labels=[1,0,0,0])
meridians = np.arange(100.,131.,10.)
m.drawmeridians(meridians,labels=[0,0,1,0])
# plot Hong Kong Location
lon, lat =  114.175, 22.292
# convert to map projection coords.
# Note that lon,lat can be scalars, lists or numpy arrays
xpt,ypt = m(lon,lat)
# convert back to lat/lon
lonpt, latpt = m(xpt,ypt,inverse=True)
# Define the new marker for the path
star = mpath.Path.unit_regular_star(5)
circle = mpath.Path.unit_circle()
# concatenate the circle with an internal cutout of the star
verts = np.concatenate([circle.vertices, star.vertices[::-1, ...]])
codes = np.concatenate([circle.codes, star.codes])
cut_star = mpath.Path(verts, codes)
m.plot(xpt,ypt,marker=cut_star,markersize=12,mfc='r',mec='r') #plot a r dot there
# put some text next to the dot, offset a little bit
# (the offset is in map projection coordinates)
#plt.text(xpt+50000,ypt+50000,'Hong Kong')

# plot the satellite and its cs records
for i in np.arange(31):
	latIPP = el_az_ipp_cs_GLO[:,i,2] # latIPP
	lonIPP = el_az_ipp_cs_GLO[:,i,3] # lonIPP
	cs_L1  = np.absolute(el_az_ipp_cs_GLO[:,i,5]) # cs on L1
	cs_L2  = np.absolute(el_az_ipp_cs_GLO[:,i,6]) # cs on L2
	# mask data
	latIPP_cs = np.ma.masked_where(cs_L2 == 0, latIPP)
	latIPP_no_cs = np.ma.masked_where(cs_L2 != 0, latIPP)
	lonIPP_cs = np.ma.masked_where(cs_L2 == 0, lonIPP)
	lonIPP_no_cs = np.ma.masked_where(cs_L2 != 0, lonIPP)
	m.scatter(lonIPP_cs,latIPP_cs, s = 40, c = 'r', marker = '^', alpha = 1.0, zorder = 2,latlon = True)
	m.scatter(lonIPP_no_cs,latIPP_no_cs, s = 5, c = 'b', marker = '.', alpha = 1.0, zorder = 1,latlon = True)

plt.text(0.88, 0.09,'GLO L2', fontsize = 12, fontweight = 'bold', ha = 'center', va = 'center', transform = ax4.transAxes)
# Plot the TC trajectory
latlon_TC = pd.read_csv('TEMBIN.TC',usecols=['Time','Latitude','Longitude'])
lon_TC = np.array(latlon_TC['Longitude'])
lat_TC = np.array(latlon_TC['Latitude'])
m.plot(lon_TC,lat_TC,linestyle='-',linewidth= 2,color = 'r', zorder = 2, latlon=True)
# scatter plot point for every record
# color bounds map for time stamp
m.scatter(lon_TC[11],lat_TC[11],marker= 'o',s= 300,c= 'c',alpha = 0.8,zorder=3,latlon=True)

fig.text(0.5, 0.02, 'August 20, 2012 (doy 233)', fontsize = 12, ha='center')
plt.show()