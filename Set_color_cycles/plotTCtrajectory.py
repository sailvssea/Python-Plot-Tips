# make a plot of TC trajectory
# lambert conformal conic map projection, drawing coastlines, state and 
# country boundaries, and parallels/meridians.
#
#the data is interpolated to the native projection grid

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import matplotlib.patches as mpathes
from matplotlib.legend_handler import HandlerPatch
import numpy as np
import pandas as pd
import cartopy.feature as cfeature

from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from datetime import datetime

def make_legend_arrow(legend, orig_handle,xdescent,ydescent,width, height, fontsize):
    p = mpathes.FancyArrow(0, 0.5*height, width, 0, length_includes_head= True, head_width=0.75*height)
    return p


"""
Plot figure
"""

fig = plt.figure(figsize=(6,6))

# create axis projection
ax = plt.axes(projection = ccrs.PlateCarree())

# draw coastlines, states and country boundaries, edges of map
Coastline = cfeature.NaturalEarthFeature('physical','land','10m',\
                                         facecolor = 'lightgray', edgecolor = 'lightgray')

ax.add_feature(Coastline)

# set the limitation of the map
ax.set_extent([105,125,12,32], ccrs.PlateCarree())


# add the grid lines
gl = ax.gridlines(crs = ccrs.PlateCarree(), draw_labels = True,\
                    linewidth = 1, color = 'black', alpha = 0.5, linestyle = '--')

#gl.xlabels_top = False
#gl.ylabels_left = False
gl.xlocator = mticker.FixedLocator([104.9,107,110,113,116,119,122,125.1])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 10, 'color': 'black'}
gl.ylabel_style = {'size': 10, 'color': 'black'}


# plot Hong Kong Location
lon, lat = 114.175, 22.292
# define the new marker with path
star = mpath.Path.unit_regular_star(5)
circle = mpath.Path.unit_circle()
verts = np.concatenate([circle.vertices, star.vertices[::-1, ...]])
codes = np.concatenate([circle.codes, star.codes])
cut_star = mpath.Path(verts, codes)
# plot on the map
plt.plot(lon, lat, marker = cut_star, markersize = 12, mfc = 'r', mec = 'r', zorder = 3, transform = ccrs.PlateCarree())

# Load the TC trajectory file
"""
different colors based on different tropical cyclones
"""
#ax.set_color_cycle([plt.cm.jet(i) for i in np.linspace(0,1,6)])
colors = [plt.cm.jet(i) for i in np.linspace(0,1,3)]
ax.set_prop_cycle('color',colors)
"""
HAGIBIS 1407 June
"""
latlon = pd.read_csv('2014HAGIBIS.TC', usecols=['Time','Latitude','Longitude'])
lon = np.array(latlon['Longitude'])
lat = np.array(latlon['Latitude'])
plt.plot(lon[:15], lat[:15], linestyle ='-', linewidth= 1.5, zorder = 2, transform = ccrs.PlateCarree())

Arrow0 = plt.arrow(lon[14], lat[14], -(lon[14]-lon[15]), -(lat[14]-lat[15]),
                  head_width=0.2, head_length=0.35, edgecolor=colors[0], linewidth =1.52,
                  facecolor=colors[0], zorder=3, transform=ccrs.PlateCarree())

# color boundds map for time stamp
#colors = np.arange(len(lon))
#norm = matplotlib.colors.BoundaryNorm(colors,256)
#plt.scatter(lon, lat, c = colors, cmap = 'jet', norm = norm, zorder = 3, transform = ccrs.PlateCarree())


"""
RAMMASUN 1409 July
"""
latlon = pd.read_csv('2014RAMMASUN.TC', usecols=['Time', 'Latitude', 'Longitude'])
lon = np.array(latlon['Longitude'])
lat = np.array(latlon['Latitude'])
plt.plot(lon[:-4], lat[:-4], linestyle='-', linewidth= 1.5,
         zorder=2, transform=ccrs.PlateCarree())
Arrow1 = plt.arrow(lon[-5], lat[-5], -(lon[-5]-lon[-4]), -(lat[-5]-lat[-4]),
                  head_width=0.2, head_length=0.35, edgecolor=colors[1], linewidth =1.52,
                  facecolor=colors[1], zorder=3, transform=ccrs.PlateCarree())
# color boundds map for time stamp
#colors = np.arange(len(lon))
#norm = matplotlib.colors.BoundaryNorm(colors, 256)
#plt.scatter(lon, lat, c=colors, cmap='jet', norm=norm,
#            zorder=3, transform=ccrs.PlateCarree())


"""
KALMAEGI 1415 September
"""
latlon = pd.read_csv('2014KALMAEGI.TC', usecols=['Time', 'Latitude', 'Longitude'])
lon = np.array(latlon['Longitude'])
lat = np.array(latlon['Latitude'])
plt.plot(lon[:-5], lat[:-5], linestyle='-', linewidth= 1.5,
         zorder=2, transform=ccrs.PlateCarree())
Arrow2 = plt.arrow(lon[-6], lat[-6], -(lon[-6]-lon[-5]), -(lat[-6]-lat[-5]),
                  head_width=0.2, head_length=0.35, edgecolor=colors[2], linewidth =1.52,
                  facecolor=colors[2], zorder=3, transform=ccrs.PlateCarree())
# color boundds map for time stamp
#colors = np.arange(len(lon))
#norm = matplotlib.colors.BoundaryNorm(colors, 256)
#plt.scatter(lon, lat, c=colors, cmap='jet', norm=norm,
#            zorder=3, transform=ccrs.PlateCarree())


# color boundds map for time stamp
#colors = np.arange(len(lon))
#norm = matplotlib.colors.BoundaryNorm(colors, 256)
#plt.scatter(lon, lat, c=colors, cmap='jet', norm=norm,
#            zorder=3, transform=ccrs.PlateCarree())
# legends
plt.legend([Arrow0,Arrow1,Arrow2],\
            ['HAGIBIS','RAMMASUN','KALMAEGI',],\
            handler_map ={mpathes.FancyArrow : HandlerPatch(patch_func=make_legend_arrow)},\
            shadow = None, facecolor = 'white', edgecolor = 'black', framealpha = 1)

plt.savefig('Trajectory.svg')
plt.show()
plt.close(fig)
plt.close('all')
