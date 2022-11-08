# phd_87a_abs_plt
# Dennis Alp 2017-10-08
# This runs with numpy 1.11.1, gradient() has been updated.
# The seconds argument int 1.11.1 is the difference between the
# points, but has been updated to the coordinate positions.

import sys
import pdb
import numpy as np
import h5py
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.basemap import Basemap
from astropy.io import fits
from scipy.stats import norm
from scipy.integrate import simps
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap

#For LaTeX style font in plots
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

# Prepare the color cycler
cyc_col = cycler('color', plt.cm.tab10(np.linspace(0,1,10)))
cyc_lin = cycler('linestyle', ['-', '--', ':', '-.'])
plt.rc('axes', prop_cycle=(cyc_lin * cyc_col))

################################################################
# Parameters
name = sys.argv[1]
verp = '/Users/silver/box/phd/pro/sne/abs/verner95/fit_par.dat'
path = '/Users/silver/dat/wongwathanarat15/' + name + '.h5'
figp = '/Users/silver/box/phd/pro/sne/abs/art/figs/' + name + '_'
atop = '/Users/silver/box/phd/pro/sne/abs/ismabs.v1.2/atomic_data/AtomicData.fits'
ismp = '/Users/silver/box/phd/pro/sne/abs/ismabs.v1.2/sample/'
xspp = '/Users/silver/box/phd/pro/87a/lim/cha/16756/repro/spectra/'

# Constants, cgs
M_Sun = 1.989e33 # g
uu = 1.660539040e-24 # g
tim = { # s
            'B15': 13489751.4508,
            'L15': 12630166.8468,
            'N20': 12488928.8787,
            'W15': 12755626.5895,
            'IIb':  1565943.04402,
            }
now = 1e4*24*60*60 # s
now = 10433*24*60*60 # s, time elapsed from SN 1987A (1987-02-23) to Chandra observation 16756 (2015-09-17)
now = float(sys.argv[2])*24*60*60 
#now = 340*365*24*60*60

# 50 shades of KTH
cdict = {'red':   [(0.0,  1.0, 1.0),
                   (0.3,   36/255,  36/255),
                   (1.0,   25/255,  25/255)],

         'green': [(0.0,  1.0, 1.0),
                   (0.3,  160/255, 160/255),
                   (1.0,   84/255,  84/255)],

         'blue':  [(0.0,  1.0, 1.0),
                   (0.3,  216/255, 216/255),
                   (1.0,  166/255, 166/255)]}
kth_blues = LinearSegmentedColormap('kth_blues', cdict)
plt.register_cmap(name='kth_blues', data=cdict)

# Desired order
sor_ele = ['p', 'n', 'he4', 'c12', 'o16', 'ne20', 'mg24', 'si28', 's32', 'ar36', 'ca40', 'ca44', 'sc44', 'ti44', 'cr48', 'fe52', 'fe56', 'co56', 'ni56', 'x56']

################################################################
# Load the data
dat = h5py.File(path, 'r')
keys = list(dat.keys())
#print("Keys: %s" % keys)

########
# Read the geometry
rad = np.array(dat['radius'])
tht = np.array(dat['theta'])
phi = np.array(dat['phi'])

# Read data
vex = np.array(dat['vex'])
den = np.array(dat['den'])

ato_idx = {'ar36':  0,
           'c12' :  1,
           'ca40':  2,
           'ca44':  3,
           'co56':  4,
           'cr48':  5,
           'fe52':  6,
           'fe56':  7,
           'he4' :  8,
           'mg24':  9,
           'n'   : 10,
           'ne20': 11,
           'ni56': 12,
           'o16' : 13,
           'p'   : 14,
           's32' : 15,
           'sc44': 16,
           'si28': 17,
           'ti44': 18,
           'x56' : 19}

lab_pri = {'ar36': '$^{36}$Ar',
           'c12' : '$^{12}$C',
           'ca40': '$^{40}$Ca',
           'ca44': '$^{44}$Ca',
           'co56': '$^{56}$Co',
           'cr48': '$^{48}$Cr',
           'fe52': '$^{52}$Fe',
           'fe56': '$^{56}$Fe',
           'he4' : '$^{4}$He',
           'mg24': '$^{24}$Mg',
           'n'   : '$^{1}$n',
           'ne20': '$^{20}$Ne',
           'ni56': '$^{56}$Ni',
           'o16' : '$^{16}$O',
           'p'   : '$^{1}$H',
           's32' : '$^{32}$S',
           'sc44': '$^{44}$Sc',
           'si28': '$^{28}$Si',
           'ti44': '$^{44}$Ti',
           'x56' : '$^{56}$X'}

nn = len(ato_idx)
nat_ord = [ato_idx[ii] for ii in sor_ele]
ato_mas = uu*np.array([35.96754510600, 12.00000000000, 39.96259098000, 43.95548180000, 55.93983930000, 47.95403200000, 51.94811400000, 55.93493630000, 4.002603254150, 23.98504170000, 1.008664915880, 19.99244017540, 55.94213200000, 15.99491461956, 1.007276466879, 31.97207100000, 43.95940280000, 27.97692653250, 43.95969010000, 55.93493630000])



################################################################
# Help functions
def verner95(ee, zz, ne):
    # ee energy
    # zz atomic number
    # ne number electrons
    # nn principal quantum number
    # ll azimuthal quantum number
    # eth subshell ionization threshold energy
    # the last are fit parameters
#   nn, ll, eth, e0, s0, ya, pp, yw

    nn = 1
    ll = 0
    ii = (ver_fit_par[:, 0] == zz) & (ver_fit_par[:, 1] == ne)# & (ver_fit_par[:, 2] == nn) & (ver_fit_par[:, 3] == ll)
    eth= ver_fit_par[ii, 4][np.newaxis, :]
    e0 = ver_fit_par[ii, 5][np.newaxis, :]
    s0 = ver_fit_par[ii, 6][np.newaxis, :]
    ya = ver_fit_par[ii, 7][np.newaxis, :]
    pp = ver_fit_par[ii, 8][np.newaxis, :]
    yw = ver_fit_par[ii, 9][np.newaxis, :]
    ll = ver_fit_par[ii, 3][np.newaxis, :]
    
    yy = ee[:, np.newaxis]/e0
    qq = 5.5+ll-0.5*pp
    fy = ((yy-1)**2+yw**2)*yy**-qq*(1+np.sqrt(yy/ya))**-pp
    fy = np.where(ene[:, np.newaxis] > eth, fy, 0)
    return np.sum(s0*fy*1e-18, axis=1)
    
def plt_hlp(lon, lat, dat, name, lab, cmap):
    # Initialize
    fig = plt.figure()
    plt.gca().axis('off')
    mm = Basemap(projection='hammer', lon_0 = 0, llcrnrlon=True, llcrnrlat=True)
    mm.drawmapboundary(fill_color='w', linewidth=0)
    fig = plt.gcf()
    fig.set_size_inches(5, 3.75)
    
    # Prepare coordinates
    plt_lon, plt_lat = np.rad2deg(lon), np.rad2deg(lat-np.pi/2)
    plt_lon, plt_lat = np.meshgrid(plt_lon, plt_lat, indexing='ij')

    # Pad because Basemap will ignore the last entry (closed surface)
    plt_dat = np.zeros((dat.shape[0]+1, dat.shape[1]+1))
    plt_dat[:-1,:-1] = dat

    # Plot
    mesh = mm.pcolormesh(plt_lon, plt_lat, plt_dat, latlon=True, edgecolors='face', lw=0, cmap=cmap)
    cbar = mm.colorbar(mesh, location='bottom', pad="5%")
    cbar.set_label(lab)

    # Finalize
    fig.savefig(figp + name + '_skym.pdf', bbox_inches='tight', pad_inches=0., dpi=300)
    plt.close()
    
    ########
    # Histogram
    (mu, sigma) = norm.fit(dat.ravel())
    fig = plt.figure()
    wei = (plt_lat[:-1, :-1]+plt_lat[1:, 1:])/2.
    wei = np.abs(np.cos(np.deg2rad(wei))).ravel()
    nn, bins, patches = plt.hist(dat.ravel(), bins=100, normed=1, weights=wei)
    plt.plot(bins, mlab.normpdf(bins, mu, sigma), 'r', linewidth=2)
    plt.xlabel(lab + ', $\mu$ = ' + str(mu) + ', $\sigma$ = ' + str(sigma))
    fig.savefig(figp + name + '_hist.pdf', bbox_inches='tight', pad_inches=0., dpi=300)
    plt.close()

def plt_pos(lon, lat, dat, name, lab):
    # Initialize
    fig = plt.figure()
    plt.gca().axis('off')
    mm = Basemap(projection='hammer', lon_0 = 0, llcrnrlon=True, llcrnrlat=True)
    mm.drawmapboundary(fill_color='w', linewidth=0)
    fig = plt.gcf()
    fig.set_size_inches(5, 3.75)
    
    # Prepare coordinates
    plt_lon, plt_lat = np.rad2deg(lon), np.rad2deg(lat-np.pi/2)
    plt_lon, plt_lat = np.meshgrid(plt_lon, plt_lat, indexing='ij')

    # Pad because Basemap will ignore the last entry (closed surface)
    plt_dat = np.zeros((dat.shape[0]+1, dat.shape[1]+1))
    plt_dat[:-1,:-1] = dat

    # Plot
    mesh = mm.pcolormesh(plt_lon, plt_lat, plt_dat, latlon=True, edgecolors='face', lw=0, cmap=kth_blues)
    cbar = mm.colorbar(mesh, location='bottom', pad="5%")
    cbar.set_label(lab.replace('PDF of the o', 'O'))

    # Finalize
    fig.savefig(figp + name + '.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
    
    ########
    # Histogram
    (mu, sigma) = norm.fit(dat.ravel())
    fig = plt.figure(figsize=(5, 3.75))
    wei = (plt_lat[:-1, :-1]+plt_lat[1:, 1:])/2.
    wei = np.abs(np.cos(np.deg2rad(wei))).ravel()
    nn, bins, patches = plt.hist(dat.ravel(), bins=100, normed=1, ec=(36/255., 160/255., 216/255.), weights=wei, color=(36/255., 160/255., 216/255.))
    plt.plot(bins, mlab.normpdf(bins, mu, sigma), color=(25/255., 84/255., 166/255.), linewidth=3)
    plt.xlabel(lab + ', $\mu$ = ' + str(np.round(mu,1)) + ', $\sigma$ = ' + str(np.round(sigma,1)))
    fig.savefig(figp + name + '_hist.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
    plt.close()
    
def plt_sel(lon, lat, dat, name, lab, sel_lon, sel_lat, sel_col):
    # Initialize
    fig = plt.figure()
    plt.gca().axis('off')
    mm = Basemap(projection='hammer', lon_0 = 0, llcrnrlon=True, llcrnrlat=True)
    mm.drawmapboundary(fill_color='w', linewidth=0)
    fig = plt.gcf()
    fig.set_size_inches(5, 3.75)
    
    # Prepare coordinates
    plt_lon, plt_lat = np.rad2deg(lon), np.rad2deg(lat-np.pi/2)
    plt_lon, plt_lat = np.meshgrid(plt_lon, plt_lat, indexing='ij')

    # Pad because Basemap will ignore the last entry (closed surface)
    plt_dat = np.zeros((dat.shape[0]+1, dat.shape[1]+1))
    plt_dat[:-1,:-1] = dat

    # Plot
    mesh = mm.pcolormesh(plt_lon, plt_lat, plt_dat, latlon=True, edgecolors='face', lw=0)
    cbar = mm.colorbar(mesh, location='bottom', pad="5%")
    for idx in range(0, sel_lon.size):
        xx, yy = mm(np.rad2deg(mid_phi[sel_lon[idx], 0,0]), np.rad2deg(mid_tht[0, sel_lat[idx], 0]))
        mm.plot(xx, yy, color=sel_col[idx], marker='o', markersize=12)
    cbar.set_label(lab)

    # Finalize
    fig.savefig(figp + name + '_pts.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)

def get_sel():
    sel_phi, sel_tht = np.zeros(10), np.zeros(10)
    sel_leg = []
    nn_p = num_tot[:,:, ato_idx['p']]
    sel_phi[0], sel_tht[0] = np.unravel_index(nn_p.argmax(), nn_p.shape)
    sel_leg.append('Max H')
    sel_phi[1], sel_tht[1] = np.unravel_index(nn_p.argmin(), nn_p.shape)
    sel_leg.append('Min H')
    
    nn_c12 = num_tot[:,:, ato_idx['c12']]
    sel_phi[2], sel_tht[2] = np.unravel_index(nn_c12.argmax(), nn_c12.shape)
    sel_leg.append('Max C12')
    sel_phi[3], sel_tht[3] = np.unravel_index(nn_c12.argmin(), nn_c12.shape)
    sel_leg.append('Min C12')

    nn_onemg = num_tot[:,:, ato_idx['o16']]+num_tot[:,:, ato_idx['ne20']]+num_tot[:,:, ato_idx['mg24']]
    sel_phi[4], sel_tht[4] = np.unravel_index(nn_onemg.argmax(), nn_onemg.shape)
    sel_leg.append('Max ONeMg')
    sel_phi[5], sel_tht[5] = np.unravel_index(nn_onemg.argmin(), nn_onemg.shape)
    sel_leg.append('Min ONeMg')

    nn_si28 = num_tot[:,:, ato_idx['si28']]
    sel_phi[6], sel_tht[6] = np.unravel_index(nn_si28.argmax(), nn_si28.shape)
    sel_leg.append('Max Si28')
    sel_phi[7], sel_tht[7] = np.unravel_index(nn_si28.argmin(), nn_si28.shape)
    sel_leg.append('Min Si28')

    nn_x56 = num_tot[:,:, ato_idx['fe56']]+num_tot[:,:, ato_idx['co56']]+num_tot[:,:, ato_idx['ni56']]+num_tot[:,:, ato_idx['x56']]
    sel_phi[8], sel_tht[8] = np.unravel_index(nn_x56.argmax(), nn_x56.shape)
    sel_leg.append('Max X56')
    sel_phi[9], sel_tht[9] = np.unravel_index(nn_x56.argmin(), nn_x56.shape)
    sel_leg.append('Min X56')
    
    sel_col = ['r', 'k',  'g','0.75',   'b', '0.5', 'c', '0.25', 'm', 'y']
    sel_for = ['-', '-', '--',  '--',  '-.',  '-.', ':',    ':', '-', '-']
    return sel_phi.astype('int'), sel_tht.astype('int'), sel_leg, sel_col, sel_for

def check_den():
    wei = np.repeat(mid_tht[:,:,0], phi.size-1, axis=0)
    wei = np.abs(np.cos(wei))
    wei = np.repeat(wei[:,:, np.newaxis], rad.size-1, axis=2)
    wei = np.repeat(wei[:,:,:, np.newaxis], 20, axis=3)
    avg_den = np.average(num_den, axis=(0,1), weights=wei)
    den_leg = ['$^{36}$Ar', '$^{12}$C' , '$^{40}$Ca', '$^{44}$Ca', '$^{56}$Co', '$^{48}$Cr', '$^{52}$Fe', '$^{56}$Fe', '$^{4}$He' , '$^{24}$Mg', '$^{1}$n'   , '$^{20}$Ne', '$^{56}$Ni', '$^{16}$O' , '$^{1}$H'   , '$^{32}$S' , '$^{44}$Sc', '$^{28}$Si', '$^{44}$Ti', '$^{56}$X', '$\Sigma{}$']
    den_leg = [den_leg[ii] for ii in nat_ord + [-1]]

    tempx = np.repeat(mid_rad[0,0][:, np.newaxis], 20, axis=1)*rad2vel
    fig = plt.figure(figsize=(5, 3.75))
    plt.loglog(tempx, avg_den[:, nat_ord]*tim_fra**3)
    plt.xlabel('Velocity (km s$^{-1}$)')
    plt.ylabel('Number density (cm$^{-3}$)')
    plt.xlim([4e0, 3e4])
    plt.ylim([1e-15, 1e6])
    plt.legend(den_leg, ncol=3)
    fig.savefig(figp + 'den_pro.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
    plt.close()
    
    fig = plt.figure(figsize=(5, 3.75))
    plt.loglog(tempx[:, 0], tim_fra**3*np.average(den, axis=(0,1), weights=wei[:,:,:, 0]))
    plt.xlabel('Velocity (km s$^{-1}$)')
    plt.ylabel('Mass density (g cm$^{-3}$)')
    plt.xlim([4e0, 3e4])
    plt.ylim([1e-25, 4e-18])
    fig.savefig(figp + 'mas_den_pro.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
    plt.close()
    
    fig = plt.figure(figsize=(5, 3.75))
    plt.loglog(tempx, tempx**2*avg_den[:, nat_ord]*tim_fra**3)
    plt.xlim([4e0, 3e4])
    plt.ylim([1e-12, 1e18])
    plt.xlabel('Velocity (km s$^{-1}$)')
    plt.ylabel('Scaled number density (km$^{2}$ s$^{-2}$ cm$^{-3}$)')
    plt.legend(den_leg, ncol=3)
    fig.savefig(figp + 'mas_pro.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
    plt.close()
    
# Extrapolate the missing data above 9.8 keV.
# It has been checked with verner95 that this is OK.
def fix_iron(adb):
    ext = adb[:, col.index('FeI')][-1536:]
    ten = ene[-1536:]
    dyd = np.diff(ext[:2])/np.diff(ten[:2])
    adb[:, col.index('FeI')][-1536:] = dyd*(ten[:]-ten[0])+ext[0]
    return adb
    
################################################################
# Check volume and density
mid_rad = (rad[:-1]+rad[1:])/2.
mid_phi = (phi[:-1]+phi[1:])/2.
mid_tht = (tht[:-1]+tht[1:])/2.-np.pi/2

mid_phi = mid_phi[:, np.newaxis, np.newaxis]
mid_tht = mid_tht[np.newaxis, :, np.newaxis]
mid_rad = mid_rad[np.newaxis, np.newaxis, :]

dif_phi = np.diff(phi)[:, np.newaxis, np.newaxis]
dif_tht = np.diff(tht)[np.newaxis, :, np.newaxis]
dif_rad = np.diff(rad)[np.newaxis, np.newaxis, :]

wei = np.repeat(mid_tht[:,:,0], phi.size-1, axis=0)
wei = np.abs(np.cos(wei))

vol = mid_rad**2*np.cos(mid_tht)*dif_rad*dif_tht*dif_phi
mas = np.sum(vol[:,:,:-1]*den[:,:,:-1])/M_Sun
print(name, 'at', tim[name], 's or', tim[name]/(3600*24), 'days')
print('Check volume and mass:', np.sum(vol)/(4*np.pi*(rad[-1]**3-rad[0]**3)/3.), mas)
for itm in sor_ele:
    pri_mas = np.sum(vol[:,:,:-1]*den[:,:,:-1]*dat[itm][:,:,:-1])/M_Sun
    pri_mas = np.round(pri_mas, -np.int(np.log10(pri_mas))+2)
    print('{:16.16s}& ${:12.1E}$ \\\\'.format(lab_pri[itm], pri_mas))
print('{:16.16s}& ${:12.1f}$ \\\\'.format('$\Sigma{}$', np.round(mas, 1)))
    
# Compute some useful quantities
tim_fra = tim[name]/now
rad2vel = 1/(tim[name]*1e5)
num_den = np.zeros(den.shape + (nn,))
num_den[:,:,:, ato_idx['ar36']] = den*dat['ar36']/ato_mas[ato_idx['ar36']]
num_den[:,:,:, ato_idx['c12' ]] = den*dat['c12' ]/ato_mas[ato_idx['c12' ]]
num_den[:,:,:, ato_idx['ca40']] = den*dat['ca40']/ato_mas[ato_idx['ca40']]
num_den[:,:,:, ato_idx['ca44']] = den*dat['ca44']/ato_mas[ato_idx['ca44']]
num_den[:,:,:, ato_idx['co56']] = den*dat['co56']/ato_mas[ato_idx['co56']]
num_den[:,:,:, ato_idx['cr48']] = den*dat['cr48']/ato_mas[ato_idx['cr48']]
num_den[:,:,:, ato_idx['fe52']] = den*dat['fe52']/ato_mas[ato_idx['fe52']]
num_den[:,:,:, ato_idx['fe56']] = den*dat['fe56']/ato_mas[ato_idx['fe56']]
num_den[:,:,:, ato_idx['he4' ]] = den*dat['he4' ]/ato_mas[ato_idx['he4' ]]
num_den[:,:,:, ato_idx['mg24']] = den*dat['mg24']/ato_mas[ato_idx['mg24']]
num_den[:,:,:, ato_idx['n'   ]] = den*dat['n'   ]/ato_mas[ato_idx['n'   ]]
num_den[:,:,:, ato_idx['ne20']] = den*dat['ne20']/ato_mas[ato_idx['ne20']]
num_den[:,:,:, ato_idx['ni56']] = den*dat['ni56']/ato_mas[ato_idx['ni56']]
num_den[:,:,:, ato_idx['o16' ]] = den*dat['o16' ]/ato_mas[ato_idx['o16' ]]
num_den[:,:,:, ato_idx['p'   ]] = den*dat['p'   ]/ato_mas[ato_idx['p'   ]]
num_den[:,:,:, ato_idx['s32' ]] = den*dat['s32' ]/ato_mas[ato_idx['s32' ]]
num_den[:,:,:, ato_idx['sc44']] = den*dat['sc44']/ato_mas[ato_idx['sc44']]
num_den[:,:,:, ato_idx['si28']] = den*dat['si28']/ato_mas[ato_idx['si28']]
num_den[:,:,:, ato_idx['ti44']] = den*dat['ti44']/ato_mas[ato_idx['ti44']]
num_den[:,:,:, ato_idx['x56' ]] = den*dat['x56' ]/ato_mas[ato_idx['x56' ]]

# Check the sum of the mass fractions
fig = plt.figure(figsize=(5, 3.75))
tot_fra = np.array(dat['ar36'])+np.array(dat['c12' ])+np.array(dat['ca40'])+np.array(dat['ca44'])+np.array(dat['co56'])+np.array(dat['cr48'])+np.array(dat['fe52'])+np.array(dat['fe56'])+np.array(dat['he4' ])+np.array(dat['mg24'])+np.array(dat['n'   ])+np.array(dat['ne20'])+np.array(dat['ni56'])+np.array(dat['o16' ])+np.array(dat['p'   ])+np.array(dat['s32' ])+np.array(dat['sc44'])+np.array(dat['si28'])+np.array(dat['ti44'])+np.array(dat['x56' ])
plt.semilogx(mid_rad[0,0]*rad2vel, np.mean(tot_fra, axis=(0,1)))
plt.xlabel('Velocity (km s$^{-1}$)')
plt.ylabel('Sum of fractions')
fig.savefig(figp + 'tot_fra.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close()

# Save velocity profile
fig = plt.figure(figsize=(5, 3.75))
plt.loglog(mid_rad[0,0]*rad2vel, 1e-5*np.average(vex, weights=np.repeat(wei[:,:, np.newaxis], vex.shape[2], axis=2), axis=(0,1)), zorder=1)
plt.loglog([1e1, 3e4], [1e1, 3e4], zorder=0)
plt.ylabel('Velocity (km s$^{-1}$)')
plt.xlabel('Velocity (km s$^{-1}$)')
plt.xlim([1e1, 3e4])
plt.ylim([1e1, 3e4])
fig.savefig(figp + 'vel_pro.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close()

# Save relative radial resolution
fig = plt.figure(figsize=(5, 3.75))
plt.semilogx(mid_rad[0,0]*rad2vel, rad[1:]/rad[:-1])
plt.ylabel('Relative radial resolution')
plt.xlabel('Velocity (km s$^{-1}$)')
plt.xlim([1e0, 3e4])
plt.ylim([0.98, 1.1])
fig.savefig(figp + 'rrr_pro.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close()



################################################################
# Load database
adb = np.array(fits.open(atop)[1].data)
col = adb.dtype.names
sig = np.zeros((adb.size, nn))
ver_fit_par = np.loadtxt(verp)

# Revert into NumPy array
adb = adb.view(('>f4', len(adb.dtype.names)))
ene = adb[:, col.index('Energy')]
adb = fix_iron(adb)
sig[:, ato_idx['ar36']] = adb[:, col.index('ArI')]
sig[:, ato_idx['c12' ]] = adb[:, col.index('CI')]
sig[:, ato_idx['ca40']] = adb[:, col.index('CaI')]
sig[:, ato_idx['ca44']] = adb[:, col.index('CaI')] # Set to 40Ca (40Ca is most abundant)
sig[:, ato_idx['co56']] = verner95(ene, 27, 27)
sig[:, ato_idx['cr48']] = verner95(ene, 24, 24)
sig[:, ato_idx['fe52']] = adb[:, col.index('FeI')] # Set to 56Fe (56Fe is most abundant)
sig[:, ato_idx['fe56']] = adb[:, col.index('FeI')]
sig[:, ato_idx['he4' ]] = adb[:, col.index('HeI')]
sig[:, ato_idx['mg24']] = adb[:, col.index('MgI')]
#sig[:, ato_idx['n'   ]] = adb[:, col.index('')]
sig[:, ato_idx['ne20']] = adb[:, col.index('NeI')]
sig[:, ato_idx['ni56']] = verner95(ene, 28, 28)
sig[:, ato_idx['o16' ]] = adb[:, col.index('OI')]
sig[:, ato_idx['p'   ]] = adb[:, col.index('H')]
sig[:, ato_idx['s32' ]] = adb[:, col.index('SI')]
sig[:, ato_idx['sc44']] = verner95(ene, 21, 21)
sig[:, ato_idx['si28']] = adb[:, col.index('SiI')]
sig[:, ato_idx['ti44']] = verner95(ene, 22, 22)
sig[:, ato_idx['x56' ]] = adb[:, col.index('FeI')] # Set to iron!

# Compare some cross sections
fig = plt.figure(figsize=(5, 3.75))
plt.loglog(ene/1000., sig[:, ato_idx['p']])
plt.loglog(ene/1000., sig[:, ato_idx['he4']])
plt.loglog(ene/1000., sig[:, ato_idx['c12']])
plt.loglog(ene/1000., sig[:, ato_idx['o16']])
plt.loglog(ene/1000., sig[:, ato_idx['mg24']])
plt.loglog(ene/1000., sig[:, ato_idx['si28']])
plt.loglog(ene/1000., sig[:, ato_idx['ar36']])
plt.loglog(ene/1000., sig[:, ato_idx['fe56']])
plt.xlabel('Energy (keV)')
plt.ylabel('Cross section (cm$^{2}$)')
plt.legend(['$^{1}$H', '$^{4}$He', '$^{12}$C', '$^{16}$O', '$^{24}$Mg', '$^{28}$Si', '$^{36}$Ar', '$^{56}$Fe'])
fig.savefig(figp + 'com_sig.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close()

# Compare some similar elements
fig = plt.figure(figsize=(5, 3.75))
plt.loglog(ene/1000., verner95(ene, 20, 20))
plt.loglog(ene/1000., verner95(ene, 21, 21))
plt.loglog(ene/1000., verner95(ene, 22, 22))
plt.loglog(ene/1000., verner95(ene, 26, 26))
plt.loglog(ene/1000., verner95(ene, 27, 27))
plt.loglog(ene/1000., verner95(ene, 28, 28))
plt.xlabel('Energy (keV)')
plt.ylabel('Cross section (cm$^{2}$)')
plt.legend(['Ca', 'Sc', 'Ti', 'Fe', 'Co', 'Ni'])
fig.savefig(figp + 'com_sim.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close()

# Compare some ionization stages
fig = plt.figure(figsize=(5, 3.75))
plt.loglog(ene/1000., verner95(ene,  8,  8))
plt.loglog(ene/1000., verner95(ene,  8,  7))
plt.loglog(ene/1000., verner95(ene,  8,  6))
plt.loglog(ene/1000., verner95(ene, 14, 14))
plt.loglog(ene/1000., verner95(ene, 14, 13))
plt.loglog(ene/1000., verner95(ene, 14, 12))
plt.loglog(ene/1000., verner95(ene, 26, 26))
plt.loglog(ene/1000., verner95(ene, 26, 25))
plt.loglog(ene/1000., verner95(ene, 26, 24))
plt.xlabel('Energy (keV)')
plt.ylabel('Cross section (cm$^{2}$)')
plt.legend(['OI', 'OII', 'OIII', 'SiI', 'SiII', 'SiIII', 'FeI', 'FeII', 'FeIII'])
fig.savefig(figp + 'com_ion.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close()

fig = plt.figure(figsize=(5, 3.75))
plt.loglog(ene/1000., verner95(ene,  8,  8))
plt.loglog(ene/1000., adb[:, col.index('OI')])
plt.loglog(ene/1000., verner95(ene,  8,  6))
plt.loglog(ene/1000., adb[:, col.index('OIII')])
plt.loglog(ene/1000., verner95(ene, 14, 14))
plt.loglog(ene/1000., adb[:, col.index('SiI')])
plt.loglog(ene/1000., verner95(ene, 14, 12))
plt.loglog(ene/1000., adb[:, col.index('SiIII')])
plt.xlabel('Energy (keV)')
plt.ylabel('Cross section (cm$^{2}$)')
plt.legend(['ver\_OI', 'gat\_OI', 'ver\_OIII', 'gat\_OIII', 'ver\_SiI', 'gat\_SiI', 'ver\_SiIII', 'gat\_SiIII'])
fig.savefig(figp + 'com_ion_ver_gat.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close()

# Compare verner95 with gatuzz15
fig = plt.figure(figsize=(5, 3.75))
plt.loglog(ene/1000., sig[:, ato_idx['p']])
plt.loglog(ene/1000., verner95(ene, 1, 1))
plt.loglog(ene/1000., sig[:, ato_idx['he4']])
plt.loglog(ene/1000., verner95(ene, 2, 2))
plt.loglog(ene/1000., sig[:, ato_idx['o16']])
plt.loglog(ene/1000., verner95(ene, 8, 8))
plt.loglog(ene/1000., sig[:, ato_idx['fe56']])
plt.loglog(ene/1000., verner95(ene, 26, 26))
plt.xlabel('Energy (keV)')
plt.ylabel('Cross section (cm$^{2}$)')
plt.legend(['$^{1}$H gatuzz15', '$^{1}$H verner95', '$^{4}$He gatuzz15', '$^{4}$He verner95', '$^{16}$O gatuzz15', '$^{16}$O verner95', '$^{56}$Fe gatuzz15', '$^{56}$Fe verner95'])
fig.savefig(figp + 'ver_gat.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close()

################################################################
# Compute the optical depth at specific energy
ene_cou = 16
ene_idx = np.argmax(ene > 2000)
num_tot = np.sum(dif_rad[:,:,:-1, np.newaxis]*num_den[:,:,:-1,:], axis=2)
tau = num_tot*sig[np.newaxis, np.newaxis, ene_idx, :]
tau = tau*tim_fra**2
plt_hlp(phi, tht, np.sum(tau, axis=2), 'tau', 'Optical depth', 'viridis')
plt_pos(phi, tht, np.sum(tau, axis=2), 'tau_pos', 'PDF of the optical depth at 2 keV\\\\ 10\,000 days (${\sim}$27 years) after explosion')
check_den()

########
# For specific directions, first the sky map then the cross sections
sel_phi, sel_tht, sel_leg, sel_col, sel_for = get_sel()
tau_sel = np.sum(num_tot[sel_phi, sel_tht, np.newaxis, :]*sig[np.newaxis, :, :], axis=2)*tim_fra**2

plt_sel(phi, tht, np.sum(tau, axis=2), 'sel', 'Optical depth', sel_phi, sel_tht, sel_col)
fig = plt.figure(figsize=(5, 3.75))
for idx in range(0, sel_phi.size):
#    plt.loglog(ene[:]/1000., (ene[:]/1000.)**2*tau_sel[idx,:], sel_col[idx], ls=sel_for[idx], lw=2)
    plt.loglog(ene[:]/1000., (ene[:]/1000.)**2*tau_sel[idx,:])

plt.xlim([0.3, 10])
plt.ylim([1e1, 2e3])
plt.xlabel('Energy (keV)')
plt.ylabel('Optical depth')
plt.legend(sel_leg)
fig.savefig(figp + 'sel_sig.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close()

########
# Sigma and tau for many energies
for idx, ene_stp in enumerate(np.logspace(2.5, 4, ene_cou)):
    ene_idx = np.argmax(ene > ene_stp)
    tau = num_tot*sig[np.newaxis, np.newaxis, ene_idx, :]
    tau = tau*tim_fra**2
    tau_rad = np.sum(tau, axis=2)
    ene_sig = tau_rad/(num_tot[:,:, ato_idx['p']]*tim_fra**2)

    ene_str = str(int(ene_stp))
#SLOW    plt_hlp(phi, tht, tau_rad, 'tau_' + ene_str, 'Optical depth at ' + ene_str + ' eV', 'viridis')
#SLOW    plt_hlp(phi, tht, ene_sig, 'sig_' + ene_str, 'Effective cross section at ' + ene_str + ' eV (cm$^2$)', 'viridis')

########
# CDF of tau for different energies
cdf_cou = 15000
ene_ste = [500, 2000, 8000]
ene_ste = [500, 1000, 2000, 4000, 8000]
ene_cou = len(ene_ste)
cdf = np.zeros((ene_cou, cdf_cou))
cdf_bin = np.zeros((ene_cou, cdf_cou+1))
cdf_lim = [1, 4e2]
col = (25/255., 84/255., 166/255.)
col = ['#D3883D', '#D8924B', '#7A5F73', '#5959CC', '#5353A0']
col = plt.cm.viridis_r(np.linspace(0.04, 0.7, 5))

for idx, ene_val in enumerate(ene_ste):
    ene_idx = np.argmax(ene > ene_val)
    tau = num_tot*sig[np.newaxis, np.newaxis, ene_idx, :]
    tau = tau*tim_fra**2
    tau_rad = np.sum(tau, axis=2)
    cdf_val, cdf_bin[idx,:] = np.histogram(tau_rad.ravel(), bins=cdf_cou, normed=True, weights=wei.ravel())
    cdf[idx,:] = np.cumsum(cdf_val)

    if ene_val == 2000:
        mid_bin = (cdf_bin[idx, :-1]+cdf_bin[idx, 1:])/2.
        spr_low = griddata(cdf[idx,:]/np.sum(cdf_val), mid_bin, 0.1, method='linear')
        spr_med = griddata(cdf[idx,:]/np.sum(cdf_val), mid_bin, 0.5, method='linear')
        spr_hig = griddata(cdf[idx,:]/np.sum(cdf_val), mid_bin, 0.9, method='linear')

fig = plt.figure(figsize=(4.5, 1.6))
ax = plt.gca()
temp = np.zeros((cdf.shape[0], cdf.shape[1]+2))
temp[:, 1:-1] = cdf
temp[:, -1] = cdf[:, -1]
cdf = temp
temp = np.zeros((cdf_bin.shape[0], cdf_bin.shape[1]+1))
temp[:, 1:-1] = cdf_bin[:, :-1]
temp[:, 0] = cdf_lim[0]
temp[:, -1] = cdf_lim[1]
cdf_bin = temp
for idx in range(0, cdf.shape[0]):
    plt.semilogx(cdf_bin[idx, :], cdf[idx, :]/cdf[idx,-1], color=col[idx], lw=3)
    
#plt.xlabel('CDF of the optical depth at 10$\,$000 days (${\sim}$27 years) after explosion')
#plt.xlabel('CDF of the optical depth at 10$\,$000 days (~27 years) after explosion')

plt.xlabel('Optical depth')
plt.ylabel('Fraction')
plt.xlim(cdf_lim)
plt.axhline(0.1, c='0.5', lw=1, zorder=0)
plt.axhline(0.5, c='0.5', lw=1, zorder=0)
plt.axhline(0.9, c='0.5', lw=1, zorder=0)
fig.savefig(figp + 'tau_cdf.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close()



########
# Column densities for all elements
for itm in ato_idx:
    ele_tot = np.sum(dif_rad[:,:,:-1]*num_den[:,:,:-1, ato_idx[itm]], axis=2)
    ele_tot = ele_tot*tim_fra**2
#SLOW    plt_hlp(phi, tht, ele_tot, 'num_den_' + itm, 'Column density (cm$^{-2}$)', 'viridis')



########
# ismabs
xsp_ene = np.loadtxt(ismp + 'ismabs.ene')*1000
xsp_flx = np.loadtxt(ismp + 'ismabs.flx')
xsp_pow = np.loadtxt(ismp + 'pow.flx')
xsp_tau = np.log(xsp_pow/xsp_flx)
ism_num = np.array([0.25e16, 33.1e16, 0.22e16, 0., 0., 0., 0., 3.16e16,   1.e20, 3.80e16, 0., 12.0e16, 0., 67.6e16,   1.e21, 2.14e16, 0., 3.35e16, 0., 0.])
ism_ele = ism_num*sig
ism_nh = ism_num[ato_idx['p']]
ism_tau = np.sum(ism_ele, axis=1)
ism_sig = ism_tau/1e21
ism_sel = [0,1,2,7,8,9,11,13,14,15,17] # All in ismabs
#ism_sel = [1,7,8,9,11,13,14,17] # All of significant abundance

# Plot
fig = plt.figure(figsize=(5, 6))
ism_leg = ['$^{36}$Ar', '$^{12}$C' , '$^{40}$Ca', '$^{44}$Ca', '$^{56}$Co', '$^{48}$Cr', '$^{52}$Fe', '$^{56}$Fe', '$^{4}$He' , '$^{24}$Mg', '$^{1}$n', '$^{20}$Ne', '$^{56}$Ni', '$^{16}$O' , '$^{1}$H', '$^{32}$S' , '$^{44}$Sc', '$^{28}$Si', '$^{44}$Ti', '$^{56}$X', '$\Sigma{}$']
ism_ord = []
for ii in nat_ord:
    if ii in ism_sel:
        ism_ord.append(ii)

ism_leg = list(ism_leg[ii] for ii in ism_ord + [-1])
plt.loglog(ene[:]/1000., ism_ele[:, np.array(ism_ord)]/ism_nh, lw=2)
plt.loglog(ene[:]/1000., ism_tau[:]/ism_nh, lw=2, color='k', ls='-')
plt.legend(ism_leg, ncol=3)
plt.xlabel('Energy (keV)')
plt.xlim([0.3, 10])
plt.ylim([1e-26, 1e-20])
plt.ylabel('Effective cross section (cm$^2$)')
fig.savefig(figp + 'ene_tau_ism.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close()

# Sparse version of the contribution to cross section in the ISM
fig = plt.figure(figsize=(5, 6))
ism_leg = []
ism_tmp = (ism_ele[:, ato_idx['p']])/ism_nh
plt.loglog(ene[:]/1000., ism_tmp, lw=2)
ism_leg.append('$^{1}$H')

ism_tmp = (ism_ele[:, ato_idx['he4']])/ism_nh
plt.loglog(ene[:]/1000., ism_tmp, lw=2)
ism_leg.append('$^{4}$He')

ism_tmp = (ism_ele[:, ato_idx['c12']])/ism_nh
plt.loglog(ene[:]/1000., ism_tmp, lw=2)
ism_leg.append('$^{12}$C')

ism_tmp = (ism_ele[:, ato_idx['o16']])/ism_nh
plt.loglog(ene[:]/1000., ism_tmp, lw=2)
ism_leg.append('$^{16}$O')

ism_tmp = (ism_ele[:, ato_idx['ar36']]+ism_ele[:, ato_idx['s32']]+ism_ele[:, ato_idx['ne20']]+ism_ele[:, ato_idx['mg24']])/ism_nh
plt.loglog(ene[:]/1000., ism_tmp, lw=2)
ism_leg.append('NeMgSAr')

ism_tmp = (ism_ele[:, ato_idx['si28']])/ism_nh
plt.loglog(ene[:]/1000., ism_tmp, lw=2)
ism_leg.append('$^{28}$Si')

ism_tmp = (ism_ele[:, ato_idx['ca40']])/ism_nh
plt.loglog(ene[:]/1000., ism_tmp, lw=2)
ism_leg.append('$^{40}$Ca$')

ism_tmp = (ism_ele[:, ato_idx['fe56']])/ism_nh
plt.loglog(ene[:]/1000., ism_tmp, lw=2)
#ism_leg.append('$^{56}$Co$^{56}$Ni$^{52}$Fe$^{56}$FeX')
ism_leg.append('$^{56}$Fe')

plt.loglog(ene[:]/1000., ism_tau[:]/ism_nh, lw=2, color='k', ls='-')
ism_leg.append('Total')

plt.legend(ism_leg, ncol=2)
plt.xlabel('Energy (keV)')
plt.ylabel('Effective cross section (cm$^2$)')
plt.xlim([0.3, 10])
plt.ylim([1e-26, 1e-20])
fig.savefig(figp + 'ene_tau_ism_spa.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close()


################################################################
# Compute average composition over all directions
avg_tot = np.sum(dif_rad[:,:,:-1, np.newaxis]*num_den[:,:,:-1,:], axis=(2))
avg_cum = np.cumsum(dif_rad[:,:,:-1, np.newaxis]*num_den[:,:,:-1,:], axis=(2))
avg_wgh = np.repeat(np.cos(mid_tht), 180, axis=0)
avg_wgh = np.repeat(avg_wgh, 20, axis=2)

# Compare the shape of the energy dependence
ene_low = np.argmax(ene > 500)
ene_hig = np.argmax(ene > 8000)
tau_low = np.sum(avg_tot*tim_fra**2*sig[ene_low], axis=2)
tau_hig = np.sum(avg_tot*tim_fra**2*sig[ene_hig], axis=2)

# Direction median
med_tot = np.zeros(20)
for idx in range(0, 20):
    med_val, med_bin = np.histogram(avg_tot[:,:, idx].ravel(), bins=cdf_cou, normed=True, weights=wei.ravel())
    med_cdf = np.cumsum(med_val)
    med_bin = (med_bin[:-1]+med_bin[1:])/2.
    med_tot[idx] = griddata(med_cdf/np.sum(med_val), med_bin, 0.5, method='linear')*tim_fra**2
    
# Prepare some more stuff, direction average
avg_tot = np.average(avg_tot, weights=avg_wgh, axis=(0,1))*tim_fra**2
avg_cum = np.average(avg_cum, weights=np.repeat(avg_wgh[:, :, np.newaxis, :], avg_cum.shape[2], axis=2), axis=(0,1))*tim_fra**2
avg_tau = avg_tot[np.newaxis, :]*sig
avg_cum = avg_cum*sig[np.argmax(ene > 2000)]

avg_nh = avg_tot[ato_idx['p']]
avg_all = np.sum(avg_tau, axis=1)
avg_sig = avg_all/avg_nh

# Back to comparing the shape
sha_nor = tau_low/avg_all[ene_low]
sha_dis = tau_hig/(sha_nor*avg_all[ene_hig])
fig = plt.figure(figsize=(5, 3.75))
plt.hist(sha_dis.ravel(), 100, normed=True, weights=wei.ravel())
plt.xlabel('Normalization relative to average at 8 keV')
plt.xlim([0, 2])
plt.ylim([0, 3.5])
fig.savefig(figp + 'sha_dis.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close()

# Spread in of optical depth in directions
ene_idx = np.argmax(ene > 2000)
spr_avg = avg_all[ene_idx]
pri_nh  = np.round(avg_nh, -np.int(np.log10(avg_nh))+1)
pri_nhl = np.round(avg_nh*spr_low/spr_avg, -np.int(np.log10(avg_nh*spr_low/spr_avg))+1)
pri_nhm = np.round(avg_nh*spr_med/spr_avg, -np.int(np.log10(avg_nh*spr_med/spr_avg))+1)
pri_nhh = np.round(avg_nh*spr_hig/spr_avg, -np.int(np.log10(avg_nh*spr_hig/spr_avg))+1)
print('Avg N_H: {:7.1E} corresponding to an optical depth of: {:5.1f}'.format(pri_nh,  np.round(spr_avg, 1)))
print('Min N_H: {:7.1E} corresponding to an optical depth of: {:5.1f}'.format(pri_nhl, np.round(spr_low, 1)))
print('Med N_H: {:7.1E} corresponding to an optical depth of: {:5.1f}'.format(pri_nhm, np.round(spr_med, 1)))
print('Max N_H: {:7.1E} corresponding to an optical depth of: {:5.1f}'.format(pri_nhh, np.round(spr_hig, 1)))

# Cumulative optical depth
fig = plt.figure(figsize=(5, 3.75))
avg_leg = ['$^{36}$Ar', '$^{12}$C' , '$^{40}$Ca', '$^{44}$Ca', '$^{56}$Co', '$^{48}$Cr', '$^{52}$Fe', '$^{56}$Fe', '$^{4}$He' , '$^{24}$Mg', '$^{1}$n', '$^{20}$Ne', '$^{56}$Ni', '$^{16}$O' , '$^{1}$H'   , '$^{32}$S' , '$^{44}$Sc', '$^{28}$Si', '$^{44}$Ti', '$^{56}$X', '$\Sigma{}$/5']
avg_leg = [avg_leg[ii] for ii in nat_ord + [-1]]
plt.semilogx(mid_rad[0,0,:-1]*rad2vel, avg_cum[:,nat_ord])
plt.semilogx(mid_rad[0,0,:-1]*rad2vel, np.sum(avg_cum, axis=1)/5., 'k')
plt.legend(avg_leg, ncol=3)
plt.ylabel('Cumulative optical depth at 2 keV')
plt.xlabel('Velocity (km s$^{-1}$)')
plt.ylim([-0.5, 15])
plt.xlim([1e1, 2e4])
fig.savefig(figp + 'cum_tau.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close()

# Differential optical depth
fig = plt.figure(figsize=(5, 3.75))
avg_leg = ['$^{36}$Ar', '$^{12}$C' , '$^{40}$Ca', '$^{44}$Ca', '$^{56}$Co', '$^{48}$Cr', '$^{52}$Fe', '$^{56}$Fe', '$^{4}$He' , '$^{24}$Mg', '$^{1}$n', '$^{20}$Ne', '$^{56}$Ni', '$^{16}$O' , '$^{1}$H'   , '$^{32}$S' , '$^{44}$Sc', '$^{28}$Si', '$^{44}$Ti', '$^{56}$X', '$\Sigma{}$/5']
avg_leg = [avg_leg[ii] for ii in nat_ord + [-1]]
lin_dif = np.gradient(avg_cum[:,nat_ord], np.repeat(dif_rad[0,0,:-1, np.newaxis], len(nat_ord), axis=1), axis=0)
plt.semilogx(mid_rad[0,0,:-1]*rad2vel, lin_dif)
plt.semilogx(mid_rad[0,0,:-1]*rad2vel, np.sum(lin_dif, axis=1)/5., 'k')
plt.legend(avg_leg, ncol=3)
plt.ylabel('Differential optical depth at 2 keV')
plt.xlabel('Velocity (km s$^{-1}$)')
ext = np.amax(lin_dif)
plt.xlim([1e1, 2e4])
plt.ylim([-0.05*ext, 1.75*ext])
plt.ylim([-1e-15, 2e-14])
fig.savefig(figp + 'dif_tau.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close()

# Log differential optical depth
fig = plt.figure(figsize=(5, 3.75))
avg_leg = ['$^{36}$Ar', '$^{12}$C' , '$^{40}$Ca', '$^{44}$Ca', '$^{56}$Co', '$^{48}$Cr', '$^{52}$Fe', '$^{56}$Fe', '$^{4}$He' , '$^{24}$Mg', '$^{1}$n', '$^{20}$Ne', '$^{56}$Ni', '$^{16}$O' , '$^{1}$H'   , '$^{32}$S' , '$^{44}$Sc', '$^{28}$Si', '$^{44}$Ti', '$^{56}$X', '$\Sigma{}$/5']
avg_leg = [avg_leg[ii] for ii in nat_ord + [-1]]
log_dif = np.repeat(mid_rad[0,0,:, np.newaxis], len(nat_ord), axis=1)
log_dif = np.gradient(avg_cum[:,nat_ord], np.diff(np.log10(log_dif), axis=0), axis=0)
#plt.semilogx(mid_rad[0,0,:-1], np.gradient(avg_cum[:,nat_ord], axis=0))
plt.semilogx(mid_rad[0,0,:-1]*rad2vel, log_dif)
plt.semilogx(mid_rad[0,0,:-1]*rad2vel, np.sum(log_dif, axis=1)/5., 'k')
plt.legend(avg_leg, ncol=2)
plt.ylabel('Logarithmic differential optical depth at 2 keV')
plt.xlabel('Velocity (km s$^{-1}$)')
ext = np.amax(log_dif)
plt.xlim([1e1, 2e4])
plt.ylim([-0.05*ext, 1.75*ext])
plt.ylim([-0.3, 30])
fig.savefig(figp + 'dif_tau_log.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close()

# Plot
fig = plt.figure(figsize=(5, 6))
plt.loglog(ene[:]/1000., avg_tau[:, nat_ord]/avg_nh, lw=2)
plt.loglog(ene[:]/1000., avg_all[:]/avg_nh, lw=2, color='k', ls='-')
plt.legend(avg_leg, ncol=3)
plt.xlabel('Energy (keV)')
plt.ylabel('Effective cross section (cm$^2$)')
plt.xlim([0.3, 10])
plt.ylim([1e-25, 1e-18])
fig.savefig(figp + 'ene_tau_avg.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close()

# Sparse version of the contribution to cross section in SN ejecta
fig = plt.figure(figsize=(5, 6))
avg_leg = []
avg_tmp = (avg_tau[:, ato_idx['p']])/avg_nh
plt.loglog(ene[:]/1000., avg_tmp, lw=2)
avg_leg.append('$^{1}$H')

avg_tmp = (avg_tau[:, ato_idx['he4']])/avg_nh
plt.loglog(ene[:]/1000., avg_tmp, lw=2)
avg_leg.append('$^{4}$He')

avg_tmp = (avg_tau[:, ato_idx['c12']])/avg_nh
plt.loglog(ene[:]/1000., avg_tmp, lw=2)
avg_leg.append('$^{12}$C')

avg_tmp = (avg_tau[:, ato_idx['o16']])/avg_nh
plt.loglog(ene[:]/1000., avg_tmp, lw=2)
avg_leg.append('$^{16}$O')

avg_tmp = (avg_tau[:, ato_idx['ar36']]+avg_tau[:, ato_idx['s32']]+avg_tau[:, ato_idx['ne20']]+avg_tau[:, ato_idx['mg24']])/avg_nh
plt.loglog(ene[:]/1000., avg_tmp, lw=2)
avg_leg.append('NeMgSAr')

avg_tmp = (avg_tau[:, ato_idx['si28']])/avg_nh
plt.loglog(ene[:]/1000., avg_tmp, lw=2)
avg_leg.append('$^{28}$Si')

avg_tmp = (avg_tau[:, ato_idx['ca40']]+avg_tau[:, ato_idx['ca44']])/avg_nh
plt.loglog(ene[:]/1000., avg_tmp, lw=2)
avg_leg.append('$^{40}$Ca$^{44}$Ca')

avg_tmp = (avg_tau[:, ato_idx['x56']]+avg_tau[:, ato_idx['fe56']]+avg_tau[:, ato_idx['ni56']]+avg_tau[:, ato_idx['fe52']]+avg_tau[:, ato_idx['co56']])/avg_nh
plt.loglog(ene[:]/1000., avg_tmp, lw=2)
#avg_leg.append('$^{56}$Co$^{56}$Ni$^{52}$Fe$^{56}$FeX')
avg_leg.append('FeCoNiX')

plt.loglog(ene[:]/1000., avg_all[:]/avg_nh, lw=2, color='k', ls='-')
avg_leg.append('Total')

plt.legend(avg_leg, ncol=2)
plt.xlabel('Energy (keV)')
plt.ylabel('Effective cross section (cm$^2$)')
plt.xlim([0.3, 10])
plt.ylim([1e-25, 1e-18])
fig.savefig(figp + 'ene_tau_avg_spa.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close()

# Print abundances
print('N_H:', avg_nh)
for itm in sor_ele:
    idx = ato_idx[itm]
    pri_col_den = avg_tot[idx]
    pri_abu_fra = avg_tot[idx]/avg_nh
    pri_ism_fra = avg_tot[idx]/avg_nh*ism_nh/ism_num[idx]
    pri_col_den = np.round(pri_col_den, -np.int(np.log10(pri_col_den))+1)
    pri_abu_fra = np.round(pri_abu_fra, -np.int(np.log10(pri_abu_fra))+2)
    pri_ism_fra = np.round(pri_ism_fra, 0) if not pri_ism_fra == np.inf else -1
    print('{:16.16s} & ${:12.1E}$ & ${:12.1E}$ & ${:12.0f}$ \\\\'.format(lab_pri[itm], pri_col_den, pri_abu_fra, pri_ism_fra))
print('$^{{1}}\\mathrm{{H}}_{{0.1}}$ & ${:12.1E}$ &  ${:12.2f}$ &  \\nodata{{}} \\\\'.format(pri_nhl, np.round(pri_nhl/avg_nh, 2)))
print('$^{{1}}\\mathrm{{H}}_{{0.9}}$ & ${:12.1E}$ &  ${:12.2f}$ &  \\nodata{{}} \\\\'.format(pri_nhh, np.round(pri_nhh/avg_nh, 2)))

########
# Computed the fudge factor
ene_idx = np.argmax(ene > 2000)
sig_sel = tau_sel/tau_sel[:, ene_idx, np.newaxis]*avg_sig[ene_idx]
tau_all = np.sum(num_tot*sig[np.newaxis, np.newaxis, ene_idx, :], axis=2)
tau_all = tau_all*tim_fra**2
fud_all = tau_all/avg_sig[ene_idx]
fud_ism = tau_all/ism_sig[ene_idx]

fig = plt.figure(figsize=(5, 3.75))
plt.loglog(ene[:]/1000, sig_sel[:,:].T)
plt.legend(sel_leg)
plt.xlabel('Energy (keV)')
plt.ylabel('Normalized cross section (cm$^{2}$)')
plt.xlim([0.3, 10])
fig.savefig(figp + 'sel_nor.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close()

fig = plt.figure(figsize=(5, 3.75))
plt.loglog(ene[:]/1000, avg_sig[:])
plt.loglog(ene[:]/1000., ism_tau[:]/1e21)
plt.legend(['SN', 'ISM'])
plt.xlabel('Energy (keV)')
plt.ylabel('Cross section (cm$^{2}$)')
plt.xlim([0.3, 10])
fig.savefig(figp + 'ene_sig_isn.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close()

fig = plt.figure(figsize=(5, 3.75))
plt.hist(fud_all.ravel(), 36, alpha=0.7)
plt.hist(fud_ism.ravel(), 36, alpha=0.7)
plt.legend(['SN', 'ISM'])
plt.xlabel('Column density (cm$^{-2}$)')
plt.gca().set_xscale('log')
fig.savefig(figp + 'fud_fac.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close()

########
# Compare ISM with SN
fig = plt.figure(figsize=(5, 3.75))
ene_idx = np.argmax(ene > 2000)
sn2ism = avg_all[ene_idx]/ism_tau[ene_idx]
print('ISM scaled by factor', sn2ism)
print('f_Z', sn2ism*1e21/avg_nh)
print('N_H', sn2ism*1e21)
plt.loglog(ene[::10]/1000., avg_all[::10])
plt.loglog(ene[::10]/1000., ism_tau[::10]*sn2ism)
plt.legend(['SN', 'ISM'])
plt.xlabel('Energy (keV)')
plt.ylabel('Optical depth')
fig.savefig(figp + 'ene_tau_isn.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close()

################################################################
# Check script with XSPEC
#plt.figure()
#plt.loglog(xsp_ene[:-1], xsp_tau, lw=2)
#plt.loglog(ene[::10], ism_tau[::10])

########
# Simplify for XSPEC
sne_all = avg_all.copy()
fig = plt.figure(figsize=(5, 3.75))
plt.loglog(ene[:]/1000., sne_all, lw=2, ls='-')

# Load database
adb = np.array(fits.open(atop)[1].data)
col = adb.dtype.names
sig = np.zeros((adb.size, nn))
ver_fit_par = np.loadtxt(verp)

# Revert into NumPy array
adb = adb.view(('>f4', len(adb.dtype.names)))
ene = adb[:, col.index('Energy')]
adb = fix_iron(adb)

sig[:, ato_idx['ar36']] = adb[:, col.index('ArI')]
sig[:, ato_idx['c12' ]] = adb[:, col.index('CI')]
sig[:, ato_idx['ca40']] = adb[:, col.index('CaI')]
sig[:, ato_idx['ca44']] = adb[:, col.index('CaI')] # Set to 40Ca (40Ca is most abundant)
sig[:, ato_idx['co56']] = adb[:, col.index('FeI')] # Co56 set to FeI
sig[:, ato_idx['cr48']] = adb[:, col.index('FeI')] # Cr48 set to FeI
sig[:, ato_idx['fe52']] = adb[:, col.index('FeI')] # Set to 56Fe (56Fe is most abundant)
sig[:, ato_idx['fe56']] = adb[:, col.index('FeI')]
# Because HeI frozen to 0.1H in ISMabs.
avg_nhe = avg_tot[ato_idx['he4']]
num_den_he1 = 0.1*avg_nh
num_den_he2 = avg_nhe - num_den_he1
sig[:, ato_idx['he4' ]] = num_den_he1/avg_nhe*adb[:, col.index('HeI')]+num_den_he2/avg_nhe*adb[:, col.index('HeII')]
sig[:, ato_idx['mg24']] = adb[:, col.index('MgI')]
#sig[:, ato_idx['n'   ]] = adb[:, col.index('')]
sig[:, ato_idx['ne20']] = adb[:, col.index('NeI')]
sig[:, ato_idx['ni56']] = adb[:, col.index('FeI')] # Ni56 set to FeI
sig[:, ato_idx['o16' ]] = adb[:, col.index('OI')]
sig[:, ato_idx['p'   ]] = adb[:, col.index('H')]
sig[:, ato_idx['s32' ]] = adb[:, col.index('SI')]
sig[:, ato_idx['sc44']] = adb[:, col.index('CaI')] # Sc44 set to CaI
sig[:, ato_idx['si28']] = adb[:, col.index('SiI')]
sig[:, ato_idx['ti44']] = adb[:, col.index('CaI')] # Ti44 set to CaI
sig[:, ato_idx['x56' ]] = adb[:, col.index('FeI')] # Set to iron!

avg_tau = avg_tot[np.newaxis, :]*sig
avg_all = np.sum(avg_tau, axis=1)
plt.loglog(ene[:]/1000., avg_all[:], lw=2, ls='-')
plt.legend(['SN', 'XSPEC'])
plt.xlabel('Energy (keV)')
plt.ylabel('Optical depth')
plt.xlim([0.3, 10])
plt.ylim([10**-0.5, 1e4])
fig.savefig(figp + 'com_sne_xsp.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close()

# Ratio to XSPEC
fig = plt.figure(figsize=(5, 2.25))
ene_idx = np.argmax(ene > 300)
fra_sne_xsp = sne_all/avg_all
plt.semilogx(ene[ene_idx:]/1000., fra_sne_xsp[ene_idx:], lw=2, ls='-')
plt.xlim([0.3, 10])
plt.xlabel('Energy (keV)')
plt.ylabel('Optical depth (Supernova/XSPEC)')
fig.savefig(figp + 'fra_sne_xsp.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close()

# Print the simplified XSPEC abundances
num_xsp = []
lab_xsp = ['H', 'HeI', 'HeII', 'C', 'O', 'Ne', 'Mg', 'Si', 'S', 'Ar', 'Ca40, Ca44, Sc44, Ti44', 'Cr48, Fe52, Fe56, Co56, Ni56, X56']
lab_ism_hlp = ['p', 'he4', 'he4', 'c12', 'o16', 'ne20', 'mg24', 'si28', 's32', 'ar36', 'ca40', 'fe56']
num_xsp.append(avg_tot[ato_idx['p']])
num_xsp.append(num_den_he1)
num_xsp.append(num_den_he2)
num_xsp.append(avg_tot[ato_idx['c12']])
num_xsp.append(avg_tot[ato_idx['o16']])
num_xsp.append(avg_tot[ato_idx['ne20']])
num_xsp.append(avg_tot[ato_idx['mg24']])
num_xsp.append(avg_tot[ato_idx['si28']])
num_xsp.append(avg_tot[ato_idx['s32']])
num_xsp.append(avg_tot[ato_idx['ar36']])
num_ca40 = avg_tot[ato_idx['ca40']]+avg_tot[ato_idx['ca44']]+avg_tot[ato_idx['sc44']]+avg_tot[ato_idx['ti44']]
num_xsp.append(num_ca40)
num_fe56 = avg_tot[ato_idx['cr48']] + avg_tot[ato_idx['fe52']] + avg_tot[ato_idx['fe56']] + avg_tot[ato_idx['co56']] + avg_tot[ato_idx['ni56']] + avg_tot[ato_idx['x56']]
num_xsp.append(num_fe56)

# Human readable format
for idx, itm in enumerate(num_xsp):
    print('{:34.34s}{:12.4E}'.format(lab_xsp[idx], itm))
for idx, itm in enumerate(num_xsp):
    pri_col_den = itm
    pri_abu_fra = itm/avg_nh
    pri_ism_fra = itm/avg_nh*ism_nh/ism_num[ato_idx[lab_ism_hlp[idx]]]
    pri_col_den = np.round(pri_col_den, -np.int(np.log10(pri_col_den))+1)
    pri_abu_fra = np.round(pri_abu_fra, -np.int(np.log10(pri_abu_fra))+2)
    pri_ism_fra = np.round(pri_ism_fra, 0) if not pri_ism_fra == np.inf else -1
    print('{:34.34s} & ${:12.1E}$ & ${:12.1E}$ & ${:12.0f}$ \\\\'.format(lab_xsp[idx], pri_col_den, pri_abu_fra, pri_ism_fra))

print('{:34.34s} & ${:12.1E}$ & ${:12.1E}$ & ${:12.0f}$ \\\\'.format('He', avg_nhe, avg_nhe/avg_nh, avg_nhe/avg_nh*ism_nh/ism_num[ato_idx['he4']]))
print('$^{{1}}\\mathrm{{H}}_{{0.1}}$             & ${:12.1E}$ & ${:12.2f}$ &  \\nodata{{}} \\\\'.format(pri_nhl, np.round(pri_nhl/avg_nh, 2)))
print('$^{{1}}\\mathrm{{H}}_{{0.9}}$             & ${:12.1E}$ & ${:12.2f}$ &  \\nodata{{}} \\\\'.format(pri_nhh, np.round(pri_nhh/avg_nh, 2)))


# Format some stuff for XSPEC pasting
low_fra = spr_low/spr_avg
med_fra = spr_med/spr_avg
hig_fra = spr_hig/spr_avg
spr_fra = [1., low_fra, med_fra, hig_fra]
skip = '                                             0 -1 0 0 1e10 1e10'
for fra in spr_fra:
    print('')
    for idx, itm in enumerate(num_xsp):
        if idx == 1: # Skip HeI
            continue
        
        uni = 1e16 if not idx == 0 else 1e22 # Different unit for hydrogen
        print('{:34.34s}{:12.4E} -1 0 0 1e10 1e10'.format(lab_xsp[idx], fra*itm/uni))

        if idx == 0 or idx == 2: # Skip ionization stages of hydrogen and HeII
            continue
        
        if idx == 11: # Quit after FeI
            print(skip) # Redshift
            break
        
        for ii in range(0,2): # Skip higher ionization stages
            print(skip)
            
        if idx == 3: # Skip nitrogen
            for ii in range(0,3):
                print(skip)

# Verify the implemented absorption
#xsp_flx = np.loadtxt(xspp + 'verify.flx')
#xsp_tau = np.log(xsp_pow/xsp_flx)
#plt.loglog(xsp_ene[:-1], xsp_tau, lw=2)
#plt.loglog(ene, avg_all)
#plt.show()

pdb.set_trace()
plt.show()
