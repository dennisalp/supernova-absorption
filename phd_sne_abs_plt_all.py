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
verp = '/Users/silver/box/phd/pro/sne/abs/verner95/fit_par.dat'
atop = '/Users/silver/box/phd/pro/sne/abs/ismabs.v1.2/atomic_data/AtomicData.fits'
ismp = '/Users/silver/box/phd/pro/sne/abs/ismabs.v1.2/sample/'
xspp = '/Users/silver/box/phd/pro/87a/lim/cha/16756/repro/spectra/'
figp = '/Users/silver/box/phd/pro/sne/abs/art/figs_all/'
# Constants, cgs
M_Sun = 1.989e33 # g
uu = 1.660539040e-24 # g

thomson = 6.6524587158e-25 # cm2
me = 510998.9461 # eV

ene_min = 0.1 # keV
ene_max_pha = 10 # keV
ene_max = 1000 # keV
ene_ext = 1000

# THESE MUST BE IN SAME ORDER (times and files)
times = [13489751.4508, 12488928.8787, 12630166.8468, 12755626.5895, 1565943.04402]
files = ['B15', 'N20', 'L15', 'W15', 'IIb']
#times = [1565943.04402]
#files = ['IIb']
#files = ['B15', 'N20']
#times = [1565943.04402, 108152.186024]
#files = ['IIb', 'I1d']
ne = 11 # Number of elements
nf = len(files)
# wilm abundances, Table 2 of wilms00
ism_num = 10**(np.array([12, 10.99, 8.38, 8.69, 7.94, 7.40, 7.27, 7.09, 6.41, 6.20, 7.43])-12)

now = 340*365*24*60*60
now = 10433*24*60*60 # s, time elapsed from SN 1987A (1987-02-23) to Chandra observation 16756 (2015-09-17)
now = float(sys.argv[1])*24*60*60

################################################################
# Allocate space for all arrays
# CDFs
ene_ste = [500, 1000, 2000, 4000, 8000, 99999]
ene_cou = len(ene_ste)

cdf_cou = 10000
cdf = np.zeros((nf, ene_cou, cdf_cou))
cdf_bin = np.zeros((nf, ene_cou, cdf_cou+1))
cdf_lim = [3e-4, 5e2]

# Variance estimate
all_avg = np.zeros((nf, ene_cou))
all_dif = np.zeros((nf, ene_cou, ne))
all_std = np.zeros((nf, ene_cou))
all_med = np.zeros((nf, ene_cou))
all_low = np.zeros((nf, ene_cou))
all_hig = np.zeros((nf, ene_cou))

# Density profiles
den_lim = [3e0, 3e4]
den_cou = 1000
vel_den = np.zeros((nf, den_cou, ne)) # This is just the x-axis for density plots
den_pro = np.zeros((nf, den_cou, ne))
mas_den_pro = np.zeros((nf, den_cou))
mas_pro = np.zeros((nf, den_cou, ne))
all_avg_cum = np.zeros((nf, den_cou))

# Shapes
nsb = 48
sha_bin = np.linspace(-2.8, -1.4, nsb)
all_sha = np.zeros((nf, nsb-1))

zzz_bin = np.linspace(0, 0.4, nsb)
all_zzz = np.zeros((nf, nsb-1))

# Optical depths
tau_cou = 10000
all_nh = np.zeros(nf)
all_ene = np.logspace(np.log10(ene_min), np.log10(ene_max), tau_cou)
all_tau = np.zeros((nf, tau_cou, ne))
all_tot = np.zeros((nf, tau_cou))
all_com_sca = np.zeros((nf, tau_cou))


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

    nn = 1 # principal quantum number
    ll = 0 # orbital quantum number
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
    fy = np.where(ene[-ene_ext:, np.newaxis] > eth, fy, 0)
    return np.sum(s0*fy*1e-18, axis=1)

def fix_all(ene, sig):
    for ii in range(0, ne):
        sig[-ene_ext:, ii] = verner95(ene[-ene_ext:], sca_hlp[ii], sca_hlp[ii])

def plt_hlp(lon, lat, dat, name, lab, cmap):
    # Initialize
    fig = plt.figure()
    plt.gca().axis('off')
    mm = Basemap(projection='hammer', lon_0 = 0, llcrnrlon=True, llcrnrlat=True)
    mm.drawmapboundary(fill_color='w', linewidth=0)
    fig = plt.gcf()
    fig.set_size_inches(5, 3.75)
    
    # Prepare coordinates
    plt_lon, plt_lat = np.rad2deg(lon), -np.rad2deg(lat-np.pi/2) # minus sign to match the Garching group
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

def check_den():
    wei = np.repeat(mid_tht[:,:,0], phi.size-1, axis=0)
    wei = np.abs(np.cos(wei))
    wei = np.repeat(wei[:,:, np.newaxis], rad.size-1, axis=2)
    wei = np.repeat(wei[:,:,:, np.newaxis], ne, axis=3)
    avg_den = np.average(num_den, axis=(0,1), weights=wei)

    tempx = np.repeat(mid_rad[0,0][:, np.newaxis], ne, axis=1)*rad2vel
    tmp_vel_den = np.logspace(np.log10(tempx[0,0]), np.log10(tempx[-1, 0]), den_cou)
    vel_den[ii, :, :] = np.repeat(tmp_vel_den[:, np.newaxis], ne, axis=1)
    
    tmp_den_pro = avg_den*tim_fra**3 # Expands in three dimensions, think volume, also [number density] = L-3
    tmp_mas_den_pro = tim_fra**3*np.average(den, axis=(0,1), weights=wei[:,:,:, 0])
    tmp_mas_pro = tempx**2*avg_den*tim_fra**3

    for jj in range(0, ne):
        den_pro[ii, :, jj] = griddata(tempx[:, jj], tmp_den_pro[:, jj], vel_den[ii, :, jj], method='linear')
        mas_pro[ii, :, jj] = griddata(tempx[:, jj], tmp_mas_pro[:, jj], vel_den[ii, :, jj], method='linear')
    mas_den_pro[ii, :] = griddata(tempx[:, 0], tmp_mas_den_pro, vel_den[ii, :, 0], method='linear')
    
# Extrapolate the missing data above 9.8 keV.
# It has been checked with verner95 that this is OK.
def fix_iron(adb):
    ext = adb[:, col.index('FeI')][-1536:]
    ten = ene[-1536:]
    dyd = np.diff(ext[:2])/np.diff(ten[:2])
    adb[:, col.index('FeI')][-1536:] = dyd*(ten[:]-ten[0])+ext[0]
    return adb

def klein_nishina(ee):
    xx = 1000*ee/me
    return 0.75*thomson*((1+xx)/xx**3*((2*xx*(1+xx))/(1+2*xx)-np.log(1+2*xx))+1/(2*xx)*np.log(1+2*xx)-(1+3*xx)/(1+2*xx)**2)



################################################################
# Some housekeeping
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
    
lab_pri = ['H', 'He', 'C', 'O', 'Ne', 'Mg', 'Si', 'S', 'Ar', 'Ca', 'Fe', 'Sum']

ato_mas = uu*np.array([35.96754510600, 12.00000000000, 39.96259098000, 43.95548180000, 55.93983930000, 47.95403200000, 51.94811400000, 55.93493630000, 4.002603254150, 23.98504170000, 1.008664915880, 19.99244017540, 55.94213200000, 15.99491461956, 1.007276466879, 31.97207100000, 43.95940280000, 27.97692653250, 43.95969010000, 55.93493630000])



################################################################
# Load cross-section database
adb = np.array(fits.open(atop)[1].data)
col = adb.dtype.names
sig = np.zeros((adb.size, ne))
ver_fit_par = np.loadtxt(verp)

sca_hlp = np.array([1, 2, 6, 8, 10, 12, 14, 16, 18, 20, 26])
sca_sig = sca_hlp[np.newaxis, :]*klein_nishina(all_ene)[:, np.newaxis]

# Revert into NumPy array
adb = adb.view(('>f4', len(adb.dtype.names)))
ene = adb[:, col.index('Energy')]
adb = fix_iron(adb)
sig[:,  0] = adb[:, col.index('H')]
sig[:,  1] = adb[:, col.index('HeI')]
sig[:,  2] = adb[:, col.index('CI')]
sig[:,  3] = adb[:, col.index('OI')]
sig[:,  4] = adb[:, col.index('NeI')]
sig[:,  5] = adb[:, col.index('MgI')]
sig[:,  6] = adb[:, col.index('SiI')]
sig[:,  7] = adb[:, col.index('SI')]
sig[:,  8] = adb[:, col.index('ArI')]
sig[:,  9] = adb[:, col.index('CaI')]
sig[:, 10] = adb[:, col.index('FeI')]

ene = np.append(ene, np.logspace(4, 3+np.log10(ene_max), ene_ext))
tmp = np.zeros((ene.size, ne))
tmp[:sig.shape[0], :] = sig
sig = tmp.copy()
fix_all(ene, sig)

# Prepare some ISM-related quantities
tmp = ism_num*sig
ism_ele = np.zeros((tau_cou, ne))
for ii in range(0, ne):
    ism_ele[:, ii] =  griddata(ene/1000., tmp[:, ii], all_ene, method='linear')
ism_sig = np.sum(ism_ele, axis=1)


    
################################################################
# Load the data
for ii, ff in enumerate(files):
    path = '/Users/silver/dat/sne/' + ff + '.h5'
    dat = h5py.File(path, 'r')
    print(ff)

    
    ########
    # Read the geometry
    rad = np.array(dat['radius'])
    tht = np.array(dat['theta'])
    phi = np.array(dat['phi'])
    
    # Read data
    vex = np.array(dat['vex'])
    den = np.array(dat['den'])

    print('Check the following lines in the script (for some temporary notes).')
#    3*0.15*2e33/(4*np.pi*((2.4e8/rad2vel*tim_fra)**3-(2e8/rad2vel*tim_fra)**3))
#    den[:,:,0]*tim_fra**2
    
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

    # Print composition
    vol = mid_rad**2*np.cos(mid_tht)*dif_rad*dif_tht*dif_phi
    mas = np.sum(vol[:,:,:-1]*den[:,:,:-1])/M_Sun
    print(ff, 'at', times[ii], 's or', times[ii]/(3600*24), 'days')
    print('Check volume and mass:', np.sum(vol)/(4*np.pi*(rad[-1]**3-rad[0]**3)/3.), mas)
    pri_lis = [['p'], ['he4'], ['c12'], ['o16'], ['ne20'], ['mg24'], ['si28'], ['s32'], ['ar36'], ['ca40', 'ca44', 'cr48', 'ti44', 'sc44'], ['fe56', 'fe52', 'co56', 'ni56', 'x56']]
    for jj, el in enumerate(pri_lis):
        tmp = 0
        for it in el:
            tmp += np.sum(vol[:,:,:-1]*den[:,:,:-1]*dat[it][:,:,:-1])/M_Sun
        pri_mas = np.round(tmp, -np.int(np.log10(tmp))+2)
        print('{:16.16s}& ${:12.1E}$ \\\\'.format(lab_pri[jj], pri_mas))
    print('{:16.16s}& ${:12.1f}$ \\\\'.format('$\Sigma{}$', np.round(mas, 1)))
    
    # Compute some useful quantities
    tim_fra = times[ii]/now
    rad2vel = 1/(times[ii]*1e5)
    num_den = np.zeros(den.shape + (ne,))
    num_den[:,:,:,  0] = den*(dat['p'   ]/ato_mas[ato_idx['p'   ]])
    num_den[:,:,:,  1] = den*(dat['he4' ]/ato_mas[ato_idx['he4' ]])
    num_den[:,:,:,  2] = den*(dat['c12' ]/ato_mas[ato_idx['c12' ]])
    num_den[:,:,:,  3] = den*(dat['o16' ]/ato_mas[ato_idx['o16' ]])
    num_den[:,:,:,  4] = den*(dat['ne20']/ato_mas[ato_idx['ne20']])
    num_den[:,:,:,  5] = den*(dat['mg24']/ato_mas[ato_idx['mg24']])
    num_den[:,:,:,  6] = den*(dat['si28']/ato_mas[ato_idx['si28']])
    num_den[:,:,:,  7] = den*(dat['s32' ]/ato_mas[ato_idx['s32' ]])
    num_den[:,:,:,  8] = den*(dat['ar36']/ato_mas[ato_idx['ar36']])
    num_den[:,:,:,  9] = den*(dat['ca40']/ato_mas[ato_idx['ca40']]+dat['ca44']/ato_mas[ato_idx['ca44']]+dat['cr48']/ato_mas[ato_idx['cr48']]+dat['ti44']/ato_mas[ato_idx['ti44']]+dat['sc44']/ato_mas[ato_idx['sc44']])
    num_den[:,:,:, 10] = den*(dat['fe56']/ato_mas[ato_idx['fe56']]+dat['fe52']/ato_mas[ato_idx['fe52']]+dat['co56']/ato_mas[ato_idx['co56']]+dat['ni56']/ato_mas[ato_idx['ni56']]+dat['x56' ]/ato_mas[ato_idx['x56' ]])

    
    ########
    # Compute the optical depth at specific energy
    num_tot = np.sum(dif_rad[:,:,:-1, np.newaxis]*num_den[:,:,:-1,:], axis=2)

    # Just print the inner and outer radius
    print('r_0 =', rad[ 0]*rad2vel)
    print('r_1 =', rad[-1]*rad2vel)
    
    ########
    # CDF and vairance of tau for different energies
    for jj, ene_val in enumerate(ene_ste):
        ene_idx = np.argmax(ene > ene_val)
        tmp_sca_sig = griddata(all_ene*1000, sca_sig, ene, method='linear')
        tau = num_tot*(sig[np.newaxis, np.newaxis, ene_idx, :] + tmp_sca_sig[np.newaxis, np.newaxis, ene_idx, :])
        tau = tau*tim_fra**2
        tau_rad = np.sum(tau, axis=2)
        cdf_val, cdf_bin[ii, jj,:] = np.histogram(tau_rad.ravel(), bins=cdf_cou, normed=True, weights=wei.ravel())
        cdf[ii, jj,:] = np.cumsum(cdf_val)

        # Variance
        tau_tot = np.sum(tau, axis=2)
        tau_avg = np.average(tau_tot, weights=wei)
        tau_var = np.average((tau_tot-tau_avg)**2, weights=wei)
        tau_std = np.sqrt(tau_var)
        all_avg[ii, jj] = tau_avg
        all_std[ii, jj] = tau_std

        # 1sigma using quantiles
        mid_bin = (cdf_bin[ii, jj, :-1]+cdf_bin[ii, jj, 1:])/2.
        all_low[ii, jj] = griddata(cdf[ii, jj,:]/cdf[ii, jj, -1], mid_bin, 0.15865525393145702, method='linear')
        all_med[ii, jj] = griddata(cdf[ii, jj,:]/cdf[ii, jj, -1], mid_bin, 0.5, method='linear')
        all_hig[ii, jj] = griddata(cdf[ii, jj,:]/cdf[ii, jj, -1], mid_bin, 0.841344746068543, method='linear')
                    
        # Catch 2 keV for sky image and spread measures
        if ene_val == 2000:
            plt_hlp(phi, tht, np.sum(tau, axis=2), files[ii], 'Optical depth', 'viridis')
            spr_low = griddata(cdf[ii, jj,:]/cdf[ii, jj, -1], mid_bin, 0.1, method='linear')
            spr_med = griddata(cdf[ii, jj,:]/cdf[ii, jj, -1], mid_bin, 0.5, method='linear')
            spr_hig = griddata(cdf[ii, jj,:]/cdf[ii, jj, -1], mid_bin, 0.9, method='linear')

            
    ########
    # Density profiles
    check_den()

    
    ########
    # Shape and averaged quantities
    # Compute average composition over all directions
    avg_tot = np.sum(dif_rad[:,:,:-1, np.newaxis]*num_den[:,:,:-1,:], axis=(2))*tim_fra**2
    avg_cum = np.cumsum(dif_rad[:,:,:-1, np.newaxis]*num_den[:,:,:-1,:], axis=(2))*tim_fra**2
    avg_wgh = np.repeat(np.cos(mid_tht), 180, axis=0)
    avg_wgh = np.repeat(avg_wgh, ne, axis=2)

    # Compare the shape of the energy dependence
    ene_low = np.argmax(ene > 300)
    ene_hig = np.argmax(ene > 9999.9999)-1
    tau_low = np.sum(avg_tot*sig[ene_low], axis=2)
    tau_hig = np.sum(avg_tot*sig[ene_hig], axis=2)
    
    # Prepare some more stuff, direction average
    avg_tot = np.average(avg_tot, weights=avg_wgh, axis=(0,1))
    avg_cum = np.average(avg_cum, weights=np.repeat(avg_wgh[:, :, np.newaxis, :], avg_cum.shape[2], axis=2), axis=(0,1))
    avg_tau = avg_tot[np.newaxis, :]*sig
    dif_tau = avg_cum*sig[np.argmax(ene > 2000)]
    dif_com_sca = avg_cum*sca_sig[np.argmax(all_ene > 2)]
    avg_cum = np.sum(dif_tau, axis=1)
    dif_com_sca = np.sum(dif_com_sca, axis=1)
    all_avg_cum[ii, :] = griddata(mid_rad[0,0,:-1]*rad2vel, avg_cum, vel_den[ii, :, 0], method='linear')
    

    fig = plt.figure(figsize=(5, 3.75))
    log_dif = np.repeat(mid_rad[0,0,:, np.newaxis], ne, axis=1)
    log_dif = np.gradient(dif_tau, np.diff(np.log10(log_dif), axis=0), axis=0)
    sca_log_dif = np.gradient(dif_com_sca, np.diff(np.log10(mid_rad[0,0,:]), axis=0), axis=0)
    plt.semilogx(mid_rad[0,0,:-1]*rad2vel, log_dif)
    plt.semilogx(mid_rad[0,0,:-1]*rad2vel, sca_log_dif*3e2)
    plt.semilogx(mid_rad[0,0,:-1]*rad2vel, np.sum(log_dif, axis=1)/3., '-k')
    tmp_lab = lab_pri.copy()
    tmp_lab.insert(-1, '$\mathrm{CS}\\times 300$')
    tmp_lab[-1] = '$\mathrm{Sum}/3$'
    plt.legend(tmp_lab, ncol=2, loc='upper left')
    plt.ylabel('Optical depth at 2 keV per $\log(\mathrm{velocity})$')
    plt.xlabel('Velocity (km s$^{-1}$)')
    ext = np.amax(log_dif)
    plt.xlim([1e1, 1e4])
    plt.ylim([-1, 15])
    fig.savefig(figp + ff + '_dif_tau_log.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
    plt.close()

    avg_nh = avg_tot[0]
    all_nh[ii] = avg_nh
    avg_sum = np.sum(avg_tau, axis=1)
    avg_sig = avg_sum/avg_nh
    
    # Compare the shape of the energy dependence
    sha_dis = (np.log10(tau_hig)-np.log10(tau_low))/(1-np.log10(0.3))
    all_sha[ii, :], trash = np.histogram(sha_dis.ravel(), bins=sha_bin, weights=wei.ravel(), density=True)

    # Map tau data to common format
    for jj in range(0, ne):
        all_tau[ii, :, jj] = griddata(ene/1000., avg_tau[:, jj], all_ene, method='linear')
    all_tot[ii, :] = griddata(ene/1000., avg_sum, all_ene, method='linear')

    # Compton scattering
    all_com_sca[ii, :] = np.sum(avg_tot[np.newaxis, :]*sca_sig, axis=1)
    
    # Plot optical depth and its composition
    fig = plt.figure(figsize=(5, 3.75))
    plt.loglog(all_ene, all_tau[ii, :, :], lw=2)
    plt.loglog(all_ene, all_com_sca[ii, :], lw=2)
    plt.loglog(all_ene, all_tot[ii, :]+all_com_sca[ii, :], lw=2, color='k', ls='-')
    tmp_lab = lab_pri.copy()
    tmp_lab.insert(-1, 'Scattering')
    plt.legend(tmp_lab, ncol=2, loc='best')
    plt.xlabel('Energy (keV)')
    plt.ylabel('Optical depth')
    plt.xlim([ene_min, ene_max])
    plt.ylim([3e-3, 1e4])
    fig.savefig(figp + ff + '_ene_tau_avg.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
    plt.close()

    # Distribution of metallicity
    tmp = np.sum(num_tot[:,:,2:], axis=2)/num_tot[:,:,0]
    all_zzz[ii, :], trash = np.histogram(tmp.ravel(), bins=zzz_bin, weights=wei.ravel(), density=True)

    # Spread in of optical depth in directions
    ene_idx = np.argmax(ene > 2000)
    spr_avg = avg_sum[ene_idx]
    pri_nha  = np.round(avg_nh, -np.int(np.log10(avg_nh))+1)
    pri_nhl = np.round(avg_nh*spr_low/spr_avg, -np.int(np.log10(avg_nh*spr_low/spr_avg))+1)
    pri_nhm = np.round(avg_nh*spr_med/spr_avg, -np.int(np.log10(avg_nh*spr_med/spr_avg))+1)
    pri_nhh = np.round(avg_nh*spr_hig/spr_avg, -np.int(np.log10(avg_nh*spr_hig/spr_avg))+1)
    print('Avg N_H: {:7.1E} corresponding to an optical depth of: {:5.1f}'.format(pri_nha,  np.round(spr_avg, 1)))
    print('Min N_H: {:7.1E} corresponding to an optical depth of: {:5.1f}'.format(pri_nhl, np.round(spr_low, 1)))
    print('Med N_H: {:7.1E} corresponding to an optical depth of: {:5.1f}'.format(pri_nhm, np.round(spr_med, 1)))
    print('Max N_H: {:7.1E} corresponding to an optical depth of: {:5.1f}'.format(pri_nhh, np.round(spr_hig, 1)))

    # Print abundances
    print('N_H:', avg_nh)
    for jj in range(0, ne):
        pri_col_den = avg_tot[jj]
        pri_abu_fra = avg_tot[jj]/avg_nh
        pri_ism_fra = avg_tot[jj]/(avg_nh*ism_num[jj])
        pri_col_den = np.round(pri_col_den, -np.int(np.log10(pri_col_den))+1)
        pri_abu_fra = np.round(pri_abu_fra, -np.int(np.log10(pri_abu_fra))+2)
        pri_ism_fra = np.round(pri_ism_fra, 0) if not pri_ism_fra == np.inf else -1
        print('{:12.12s} & ${:12.1E}$ & ${:12.1E}$ & ${:12.0f}$ & ${:12.1E}$ \\\\'.format(lab_pri[jj], pri_col_den, pri_abu_fra, pri_ism_fra, pri_col_den*sca_hlp[jj]))

    pri_col_den = np.sum(avg_tot)
    pri_ele_den = np.sum(avg_tot*sca_hlp)
    pri_col_den = np.round(pri_col_den, -np.int(np.log10(pri_col_den))+1)
    pri_ele_den = np.round(pri_ele_den, -np.int(np.log10(pri_ele_den))+1)
    print('{:12.12s} & ${:12.1E}$ &   {:12.12s} &   {:12.12s} & ${:12.1E}$ \\\\'.format('Sum', pri_col_den, '\\nodata{}', '\\nodata{}', pri_ele_den))
    print('H$_{{0.1}}$ & ${:12.1E}$ &  ${:12.2f}$ & \\nodata{{}} & \\nodata{{}} \\\\'.format(pri_nhl, np.round(pri_nhl/avg_nh, 2)))
    print('H$_{{0.9}}$ & ${:12.1E}$ &  ${:12.2f}$ & \\nodata{{}} & \\nodata{{}} \\\\'.format(pri_nhh, np.round(pri_nhh/avg_nh, 2)))

    ########
    # Compare ISM with SN
    ene_idx = np.argmax(ene > 2000)
    all_idx = np.argmax(all_ene > 2)
    sn2ism = avg_sum[ene_idx]/ism_sig[all_idx]
    print('f_Z', sn2ism/avg_nh)
    print('N_ISM', sn2ism)


    ########
    # XSPEC stuff
    # Print the simplified XSPEC abundances
    # Human readable format
    for jj, itm in enumerate(avg_tot):
        print('{:34.34s}{:12.4E}'.format(lab_pri[jj], itm))
    
    # Format some stuff for XSPEC pasting
    low_fra = spr_low/spr_avg
    med_fra = spr_med/spr_avg
    hig_fra = spr_hig/spr_avg
    spr_fra = [1., low_fra, med_fra, hig_fra]
    step = [2, 4, 5, 7, 9]
    skip = '           0 -1 0 0 1e10 1e10'
    ski2 = '           1 -1 0 0    1    1'
    for fra in spr_fra:
        print('BREAK')
        for jj, itm in enumerate(avg_tot):
            out = fra*itm/1e22 if jj == 0 else itm/avg_nh/ism_num[jj] # Different unit for hydrogen
            print('{:12.4E} -1 0 0 1e10 1e10'.format(out))
    
            if jj in step:
                print(skip)

        print(''.join(( 7*[skip + '\n'])) + ''.join((18*[ski2 + '\n'])) + skip)



################################################################
# Plots for all
# Plot CDF
# Pad first and last bin for aesthetics
temp = np.zeros((cdf.shape[0], cdf.shape[1], cdf.shape[2]+2))
temp[:, :, 1:-1] = cdf
temp[:, :, -1] = cdf[:, :, -1]
cdf = temp
temp = np.zeros((cdf_bin.shape[0], cdf_bin.shape[1], cdf_bin.shape[2]+1))
temp[:, :, 1:-1] = cdf_bin[:, :, :-1]
temp[:, :, 0] = cdf_lim[0]
temp[:, :, -1] = cdf_lim[1]
cdf_bin = temp

for jj, ene_val in enumerate(ene_ste):
    fig = plt.figure(figsize=(4.5, 1.6))
    ax = plt.gca()
    for ii in range(0, nf):
        plt.semilogx(cdf_bin[ii, jj, :], cdf[ii, jj, :]/cdf[ii, jj,-1], lw=2)

    if jj == 0:
        plt.legend(files, ncol=2, loc='upper left')
    
    plt.xlabel('Optical depth at ' + str(np.round(ene_val/1000., 1)) + ' keV')
    plt.ylabel('Fraction')
    plt.xlim(cdf_lim)
    plt.axhline(0.1, c='#7f7f7f', lw=1, zorder=0)
    plt.axhline(0.5, c='#7f7f7f', lw=1, zorder=0)
    plt.axhline(0.9, c='#7f7f7f', lw=1, zorder=0)
    fig.savefig(figp + 'tau_cdf_' + str(ene_val) + '.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
    plt.close()

    
########
# Plot density profiles
#fig = plt.figure(figsize=(5, 3.75))
#for ii in range(0, nf):
#    plt.loglog(vel_den[ii, :, :], den_pro[ii, :, :])
#plt.xlabel('Velocity (km s$^{-1}$)')
#plt.ylabel('Number density (cm$^{-3}$)')
#plt.xlim(den_lim)
#plt.ylim([1e-15, 1e6])
#plt.legend(lab_pri, ncol=3, loc='best')
#fig.savefig(figp + 'den_pro.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
#plt.close()

fig = plt.figure(figsize=(5, 3.75))
for ii in range(0, nf):
    plt.loglog(vel_den[ii, :, 0], mas_den_pro[ii, :])
plt.xlabel('Velocity (km s$^{-1}$)')
plt.ylabel('Mass density (g cm$^{-3}$)')
plt.xlim(den_lim)
plt.ylim([1e-25, 4e-18])
plt.legend(files, loc='best')
fig.savefig(figp + 'mas_den_pro.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close()

#fig = plt.figure(figsize=(5, 3.75))
#for ii in range(0, nf):
#    plt.loglog(vel_den[ii, :, :], mas_pro[ii, :, :])
#plt.xlim(den_lim)
#plt.ylim([1e-12, 1e18])
#plt.xlabel('Velocity (km s$^{-1}$)')
#plt.ylabel('Scaled number density (km$^{2}$ s$^{-2}$ cm$^{-3}$)')
#plt.legend(lab_pri, ncol=3, loc='best')
#fig.savefig(figp + 'mas_pro.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
#plt.close()

########
# Plot shape distributions
fig = plt.figure(figsize=(5, 3.75))
for ii in range(0, nf):
    plt.plot((sha_bin[:-1]+sha_bin[1:])/2, all_sha[ii, :])

plt.ylabel('Probability density')
plt.xlabel('Effective slope')
plt.xlim([-2.8, -1.4])
plt.ylim([0, 7.5])
plt.legend(files, loc='upper left')
fig.savefig(figp + 'sha_dis.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close()

########
# Plot metallicity distributions
fig = plt.figure(figsize=(5, 3.75))
for ii in range(0, nf):
    if ii < 4:
        plt.plot((zzz_bin[:-1]+zzz_bin[1:])/2, all_zzz[ii, :])
    else:
        plt.plot((zzz_bin[:-1]+zzz_bin[1:])/2, all_zzz[ii, :]/10.)

plt.ylabel('Probability density')
plt.xlabel('Metallicity')
plt.xlim([0, 0.4])
plt.ylim([0, 25])
plt.legend(files, loc='upper left')
fig.savefig(figp + 'zzz_dis.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close()

# Plot effective cross-section for all models
fig = plt.figure(figsize=(5, 3.75))
for ii in range(0, nf):
    plt.loglog(all_ene, all_tot[ii, :]/all_nh[ii], lw=2)
plt.loglog(all_ene, ism_sig, lw=2)
plt.legend(files + ['ISM'], ncol=3, loc='best')
plt.xlabel('Energy (keV)')
plt.ylabel('Effective cross-section (cm$^2$)')
plt.xlim([ene_min, ene_max_pha])
plt.ylim([1e-25, 1e-17])
fig.savefig(figp + 'ene_sig_avg.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close()

# Plot optical depth for all models
fig = plt.figure(figsize=(5, 3.75))
for ii in range(0, nf):
    plt.loglog(all_ene, all_tot[ii, :] + all_com_sca[ii, :], lw=2, ls='-', color=plt.cm.tab10(np.linspace(0,1,10))[ii])
for ii in range(0, nf):
    plt.loglog(all_ene, all_tot[ii, :], lw=2, ls='--', color=plt.cm.tab10(np.linspace(0,1,10))[ii])
for ii in range(0, nf): # Get the legend in the correct order
    plt.loglog(all_ene, all_com_sca[ii, :], lw=2, ls=':', color=plt.cm.tab10(np.linspace(0,1,10))[ii])


plt.legend(files, ncol=3, loc='best')
plt.xlabel('Energy (keV)')
plt.ylabel('Optical depth')
plt.xlim([ene_min, ene_max])
plt.ylim([1e-4, 1e4])
fig.savefig(figp + 'ene_tau_avg.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close()

fig = plt.figure(figsize=(5, 3.75))
for ii in range(0, nf):
    plt.loglog(all_ene, all_tot[ii, :]/(all_nh[ii]*ism_sig))
plt.legend(files, loc='best')
plt.xlabel('Energy (keV)')
plt.ylabel('$\langle\sigma_\mathrm{SN}\\rangle{}_{\hat n}/\sigma_\mathrm{ISM}$')
plt.xlim([ene_min, ene_max_pha])
fig.savefig(figp + 'ene_tau_isn.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close()

# Table of uncertainties, mean \pm sigma
for ii, ff in enumerate(files):
    out = '{:6.6s} '.format(ff)
    for jj, ene_val in enumerate(ene_ste):
        out += '& ${:8.2f}\\pm{:8.2f}$ '.format(all_avg[ii, jj], all_std[ii, jj])
        
    print(out + '\\\\')

# Table of uncertainties, median -lower +upper
for ii, ff in enumerate(files):
    out = '{:6.6s} '.format(ff)
    for jj, ene_val in enumerate(ene_ste):
        out += '& ${:11.8f}_{{{:11.8f}}}^{{+{:11.8f}}}$ '.format(all_med[ii, jj], all_low[ii, jj]-all_med[ii, jj], all_hig[ii, jj]-all_med[ii, jj])
    for jj, ene_val in enumerate(ene_ste):
        out += '& ${:11.8f}$ '.format(all_hig[ii, jj]/all_low[ii, jj])
        
    print(out + '\\\\')

# Some nasty hax
if len(files) == 5:
    files[4] = '$\mathrm{IIb}\\times 20$'

# Cumulative optical depth
fig = plt.figure(figsize=(5, 3.75))
for ii in range(0, nf):
    scale = 20 if ii == 4 else 1
    plt.semilogx(vel_den[ii, :, 0], scale*all_avg_cum[ii, :])
plt.legend(files, loc='best')
plt.ylabel('Cumulative optical depth at 2 keV')
plt.xlabel('Velocity (km s$^{-1}$)')
plt.ylim([-1.5, 45])
plt.xlim([1e1, 1e4])
fig.savefig(figp + 'cum_tau.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close()

pdb.set_trace()
plt.show()
