# source activate py35_qt4
# conda create -n py35_qt4 python=3.5 pyqt=4 anaconda
# conda install -c menpo mayavi=4.5.0
# conda install basemap

import pdb
import h5py
import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
#['GTK', 'GTKAgg', 'GTKCairo', 'MacOSX', 'Qt4Agg', 'Qt5Agg', 'TkAgg', 'WX', 'WXAgg', 'CocoaAgg', 'GTK3Cairo', 'GTK3Agg', 'WebAgg', 'nbAgg']
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mayavi import mlab

################################################################
# Parameters
path = '/Users/silver/Dropbox/phd/projects/87a/abs/data/B15.h5'

################################################################
# Load the data
dat = h5py.File(path, 'r')
keys = list(dat.keys())
print("Keys: %s" % keys)

########
# Read the geometry
rad = np.array(dat['radius'])
tht = np.array(dat['theta'])
phi = np.array(dat['phi'])

# Read data
vex = np.array(dat['vex'])
den = np.array(dat['den'])

# Read elements
ar36 = np.array(dat['ar36'])
c12  = np.array(dat['c12' ])
ca40 = np.array(dat['ca40'])
ca44 = np.array(dat['ca44'])
co56 = np.array(dat['co56'])
cr48 = np.array(dat['cr48'])
fe52 = np.array(dat['fe52'])
fe56 = np.array(dat['fe56'])
he4  = np.array(dat['he4' ])
mg24 = np.array(dat['mg24'])
nn   = np.array(dat['n'   ])
ne20 = np.array(dat['ne20'])
ni56 = np.array(dat['ni56'])
o16  = np.array(dat['o16' ])
pp   = np.array(dat['p'   ])
s32  = np.array(dat['s32' ])
sc44 = np.array(dat['sc44'])
si28 = np.array(dat['si28'])
ti44 = np.array(dat['ti44'])
x56  = np.array(dat['x56' ])

#for key in dat:
#    if key in ['phi', 'theta', 'radius']:
#        continue
#    print(np.sum(np.isnan(np.array(dat[key])[:,:,1199])), np.sum(np.isnan(np.array(dat[key])[:,:,:1199])))

plt.figure()
plt.loglog(rad[:-1], rad[:-1]**2*np.sum(den, axis=(0,1)))

#plt.figure()
#mm = Basemap(width=12000000,height=9000000,projection='hammer',
#            resolution='c',lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)
#mm.bluemarble()

plt.figure()
mm = Basemap(projection='hammer', lon_0 = 0, llcrnrlon=True, llcrnrlat=True)
plt_phi, plt_tht = np.rad2deg(phi), np.rad2deg(tht-np.pi/2)
plt_phi, plt_tht = np.meshgrid(plt_phi, plt_tht, indexing='ij')
plt_den = np.zeros((den.shape[0]+1, den.shape[1]+1))
plt_den[:-1,:-1] = np.sum(den[:,:,:-1], axis=2)
mm.pcolormesh(plt_phi, plt_tht, plt_den, latlon=True)

#plt.figure()
#sparse = 30
#tst_phi = phi[::sparse]
#tst_tht = tht[::sparse]
#mm = Basemap(projection='hammer', lon_0 = 0, llcrnrlon=True, llcrnrlat=True)
#plt_phi, plt_tht = np.meshgrid(np.rad2deg(tst_phi[:-1]), np.rad2deg(tst_tht[:-1]-np.pi/2), indexing='ij')
#mm.pcolormesh(plt_phi, plt_tht, np.sum(den[::sparse,::sparse,:-1], axis=2), latlon=True)

plt.show()

#  #from mayavi import mlab
#  #
#  #mlab.figure(bgcolor=(0, 0, 0), size=(400, 400))
#  #
#  #src = mlab.pipeline.scalar_field(den[:,:,:-1])
#  ## Our data is not equally spaced in all directions:
#  #src.spacing = [1, 1, 1.5]
#  #src.update_image_data = True
#  #
#  #
#  ## Extract some inner structures: the ventricles and the inter-hemisphere
#  ## fibers. We define a volume of interest (VOI) that restricts the
#  ## iso-surfaces to the inner of the brain. We do this with the ExtractGrid
#  ## filter.
#  #blur = mlab.pipeline.user_defined(src, filter='ImageGaussianSmooth')
#  #voi = mlab.pipeline.extract_grid(blur)
#  #voi.set(x_min=125, x_max=193, y_min=92, y_max=125, z_min=34, z_max=75)
#  #
#  #mlab.pipeline.iso_surface(voi, contours=[1610, 2480], colormap='Spectral')
#  #
#  ## Add two cut planes to show the raw MRI data. We use a threshold filter
#  ## to remove cut the planes outside the brain.
#  #thr = mlab.pipeline.threshold(src, low=1120)
#  #cut_plane = mlab.pipeline.scalar_cut_plane(thr,
#  #                                plane_orientation='y_axes',
#  #                                colormap='black-white',
#  #                                vmin=1400,
#  #                                vmax=2600)
#  #cut_plane.implicit_plane.origin = (136, 111.5, 82)
#  #cut_plane.implicit_plane.widget.enabled = False
#  #
#  #cut_plane2 = mlab.pipeline.scalar_cut_plane(thr,
#  #                                plane_orientation='z_axes',
#  #                                colormap='black-white',
#  #                                vmin=1400,
#  #                                vmax=2600)
#  #cut_plane2.implicit_plane.origin = (136, 111.5, 82)
#  #cut_plane2.implicit_plane.widget.enabled = False
#  #
#  ## Extract two views of the outside surface. We need to define VOIs in
#  ## order to leave out a cut in the head.
#  #voi2 = mlab.pipeline.extract_grid(src)
#  #voi2.set(y_min=112)
#  #outer = mlab.pipeline.iso_surface(voi2, contours=[1776, ],
#  #                                        color=(0.8, 0.7, 0.6))
#  #
#  #voi3 = mlab.pipeline.extract_grid(src)
#  #voi3.set(y_max=112, z_max=53)
#  #outer3 = mlab.pipeline.iso_surface(voi3, contours=[1776, ],
#  #                                         color=(0.8, 0.7, 0.6))
#  #
#  #
#  #mlab.view(-125, 54, 326, (145.5, 138, 66.5))
#  #mlab.roll(-175)
#  #
#  #mlab.show()

## ##from mayavi import mlab
## ##mlab.figure(1, bgcolor=(0.48, 0.48, 0.48), fgcolor=(0, 0, 0),
## ##               size=(400, 400))
## ##mlab.clf()
## ##
## #################################################################################
## ### Display points at city positions
## ##coords = np.array(coords)
## ### First we have to convert latitude/longitude information to 3D
## ### positioning.
## ##lat, long = coords.T * np.pi / 180
## ##x = np.cos(long) * np.cos(lat)
## ##y = np.cos(long) * np.sin(lat)
## ##z = np.sin(long)
## ##
## ##points = mlab.points3d(x, y, z,
## ##                     scale_mode='none',
## ##                     scale_factor=0.03,
## ##                     color=(0, 0, 1))
## ##
## #################################################################################
## ### Display connections between cities
## ##connections = np.array(connections)
## ### We add lines between the points that we have previously created by
## ### directly modifying the VTK dataset.
## ##points.mlab_source.dataset.lines = connections
## ##points.mlab_source.reset()
## ### To represent the lines, we use the surface module. Using a wireframe
## ### representation allows to control the line-width.
## ##mlab.pipeline.surface(points, color=(1, 1, 1),
## ##                              representation='wireframe',
## ##                              line_width=4,
## ##                              name='Connections')
## ##
## #################################################################################
## ### Display city names
## ##for city, index in cities.items():
## ##    label = mlab.text(x[index], y[index], city, z=z[index],
## ##                      width=0.016 * len(city), name=city)
## ##    label.property.shadow = True
## ##
## #################################################################################
## ### Display continents outline, using the VTK Builtin surface 'Earth'
## ##from mayavi.sources.builtin_surface import BuiltinSurface
## ##continents_src = BuiltinSurface(source='earth', name='Continents')
## ### The on_ratio of the Earth source controls the level of detail of the
## ### continents outline.
## ##continents_src.data_source.on_ratio = 2
## ##continents = mlab.pipeline.surface(continents_src, color=(0, 0, 0))
## ##
## #################################################################################
## ### Display a semi-transparent sphere, for the surface of the Earth
## ##
## ### We use a sphere Glyph, throught the points3d mlab function, rather than
## ### building the mesh ourselves, because it gives a better transparent
## ### rendering.
## ##sphere = mlab.points3d(0, 0, 0, scale_mode='none',
## ##                                scale_factor=2,
## ##                                color=(0.67, 0.77, 0.93),
## ##                                resolution=50,
## ##                                opacity=0.7,
## ##                                name='Earth')
## ##
## ### These parameters, as well as the color, where tweaked through the GUI,
## ### with the record mode to produce lines of code usable in a script.
## ##sphere.actor.property.specular = 0.45
## ##sphere.actor.property.specular_power = 5
## ### Backface culling is necessary for more a beautiful transparent
## ### rendering.
## ##sphere.actor.property.backface_culling = True
## ##
## #################################################################################
## ### Plot the equator and the tropiques
## ##theta = np.linspace(0, 2 * np.pi, 100)
## ##for angle in (- np.pi / 6, 0, np.pi / 6):
## ##    x = np.cos(theta) * np.cos(angle)
## ##    y = np.sin(theta) * np.cos(angle)
## ##    z = np.ones_like(theta) * np.sin(angle)
## ##
## ##    mlab.plot3d(x, y, z, color=(1, 1, 1),
## ##                        opacity=0.2, tube_radius=None)
## ##
## ##mlab.view(63.4, 73.8, 4, [-0.05, 0, 0])
## ##mlab.show()

# # #x, y, z = np.ogrid[-10:10:20j, -10:10:20j, -10:10:20j]
# # #s = np.sin(x*y*z)/(x*y*z)
# # #
# # #src = mlab.pipeline.scalar_field(s)
# # #mlab.pipeline.iso_surface(src, contours=[s.min()+0.1*s.ptp(), ], opacity=0.3)
# # #mlab.pipeline.iso_surface(src, contours=[s.max()-0.1*s.ptp(), ],)
# # #
# # #mlab.show()

####plt_den = np.log10(den[::4,::2,0:800:20])
####plt_tht = tht[:-1:2][np.newaxis, :, np.newaxis]-np.pi/2
####plt_phi = phi[:-1:4][:, np.newaxis, np.newaxis]
####plt_rad = np.log(rad[0:800:20][np.newaxis, np.newaxis, :])
####xx = plt_rad*np.cos(plt_tht) * np.cos(plt_phi)
####yy = plt_rad*np.cos(plt_tht) * np.sin(plt_phi)
####zz = plt_rad*np.sin(plt_tht)
####zz = np.repeat(zz, 45, axis=0)
####
####src = mlab.pipeline.scalar_scatter(xx, yy, zz, plt_den)
#####pts = mlab.pipeline.glyph(src, scale_mode='none', scale_factor=1)
####field = mlab.pipeline.delaunay3d(src)
####mlab.pipeline.iso_surface(field, contours=[plt_den.min()+0.1*plt_den.ptp(), ], opacity=0.3)
#####mlab.pipeline.iso_surface(field, contours=[plt_den.min()+0.3*plt_den.ptp(), ], opacity=0.3)
#####mlab.pipeline.iso_surface(field, contours=[plt_den.min()+0.5*plt_den.ptp(), ], opacity=0.3)
#####mlab.pipeline.iso_surface(field, contours=[plt_den.min()+0.7*plt_den.ptp(), ], opacity=0.3)
####mlab.pipeline.iso_surface(field, contours=[plt_den.min()+0.9*plt_den.ptp(), ], opacity=0.3)
####
####mlab.show()

#####x, y, z = np.random.random((3, 100))
#####data = x**2 + y**2 + z**2
#####src = mlab.pipeline.scalar_scatter(x, y, z, data)
#####pts = mlab.pipeline.glyph(src, scale_mode='none', scale_factor=.1)
#####field = mlab.pipeline.delaunay3d(src)
