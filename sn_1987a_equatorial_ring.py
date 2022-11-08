# phd_87a_abs_plt
# Dennis Alp 2017-10-08
# This runs with numpy 1.11.1, gradient() has been updated.
# The seconds argument int 1.11.1 is the difference between the
# points, but has been updated to the coordinate positions.

import sys
from pdb import set_trace as db
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
figp = '/Users/silver/box/phd/pro/sne/er/fig/'
# Constants, cgs
M_Sun = 1.989e33 # g
uu = 1.660539040e-24 # g
cc = 2.99792458e10 # cm s-1
kpc = 3.086e21 # cm
dd = 51.2*kpc # cm
er = 0.85 # arcsec
er = er/60/60/360*2*np.pi # rad
er = er*dd # cm
incl = np.pi/4 # rad 

thomson = 6.6524587158e-25 # cm2
me = 510998.9461 # eV

ene_min = 0.1 # keV
ene_max_plt = 10 # keV
ene_max = 1000 # keV
ene_ext = 1000

# THESE MUST BE IN SAME ORDER (times and files)
times = [13489751.4508, 13489751.4508, 12488928.8787, 12630166.8468, 12755626.5895, 1565943.04402]
files = ['B15', 'B15_87A', 'N20', 'L15', 'W15', 'IIb']
#times = [1565943.04402]
#files = ['IIb']
#files = ['B15', 'N20']
#times = [1565943.04402, 108152.186024]
#files = ['IIb', 'I1d']
ne = 11 # Number of elements
nf = len(files)
# wilm abundances, Table 2 of wilms00
ism_num = 10**(np.array([12, 10.99, 8.38, 8.69, 7.94, 7.40, 7.27, 7.09, 6.41, 6.20, 7.43])-12)

# now = 340*365*24*60*60 # Cas A
now = 13000*24*60*60 # 87A
# now = float(sys.argv[1])*24*60*60

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
def sph2car(rr, ph, th):
    xx = rr*np.cos(np.deg2rad(ph))*np.sin(np.deg2rad(th))
    yy = rr*np.sin(np.deg2rad(ph))*np.sin(np.deg2rad(th))
    zz = rr*np.cos(np.deg2rad(th))
    return xx, yy, zz

def car2sph(xx, yy, zz):
    return None 
    
    
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
    # vol = mid_rad**2*np.cos(mid_tht)*dif_rad*dif_tht*dif_phi
    # mas = np.sum(vol[:,:,:-1]*den[:,:,:-1])/M_Sun
    # print(ff, 'at', times[ii], 's or', times[ii]/(3600*24), 'days')
    # print('Check volume and mass:', np.sum(vol)/(4*np.pi*(rad[-1]**3-rad[0]**3)/3.), mas)
    # pri_lis = [['p'], ['he4'], ['c12'], ['o16'], ['ne20'], ['mg24'], ['si28'], ['s32'], ['ar36'], ['ca40', 'ca44', 'cr48', 'ti44', 'sc44'], ['fe56', 'fe52', 'co56', 'ni56', 'x56']]
    # for jj, el in enumerate(pri_lis):
    #     tmp = 0
    #     for it in el:
    #         tmp += np.sum(vol[:,:,:-1]*den[:,:,:-1]*dat[it][:,:,:-1])/M_Sun
    #     pri_mas = np.round(tmp, -np.int(np.log10(tmp))+2)
    #     print('{:16.16s}& ${:12.1E}$ \\\\'.format(lab_pri[jj], pri_mas))
    # print('{:16.16s}& ${:12.1f}$ \\\\'.format('$\Sigma{}$', np.round(mas, 1)))
    
    # Compute some useful quantities
    tim_fra = times[ii]/now
    den = den*tim_fra**3
    rad = rad*tim_fra**-1
    mid_rad = mid_rad*tim_fra**-1
    
    rad2vel = 1/(now*1e5)
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

    

    # Just print the inner and outer radius
    print('r_0 =', rad[ 0]*rad2vel)
    print('r_1 =', rad[-1]*rad2vel)

    ########
    # Shape and averaged quantities
    # Compute average composition over all directions
    avg_wgh = np.repeat(np.cos(mid_tht), 180, axis=0)
    avg_wgh = np.repeat(avg_wgh, ne, axis=2)
    avg_den = np.average(num_den, weights=np.repeat(avg_wgh[:, :, np.newaxis, :], num_den.shape[2], axis=2), axis=(0,1))

    bb = np.cos(incl)*er
    dl = np.sin(incl)*er
    ib = np.argmax(mid_rad>bb*1.2)
    avg_den_bb = avg_den[ib,:]
    avg_col_bb = avg_den_bb*dl
    avg_sig_bb = avg_col_bb*sig
    db()

    fig = plt.figure(figsize=(5, 3.75))
    plt.loglog(ene/1e3, avg_sig_bb)
    plt.loglog(ene/1e3, np.sum(avg_sig_bb, axis=1), lw=2, color='k', ls='-')
    plt.xlabel('Energy (keV)')
    plt.ylabel('Optical depth')
    plt.xlim([ene_min, ene_max_plt])
    plt.ylim([1e-3, 1e2])
    fig.savefig(figp + ff + '_avg_sig_bb_r1p2.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
    plt.close()




    ################################################################
    # FULL INTEGRATION
    ################################################################
    # for jj in mid_phi[:,0,0]:
    #     sph2car(er, jj, 0)






    ################################################################
    # OLD
    ################################################################

    
    # # Compton scattering
    # all_com_sca[ii, :] = np.sum(avg_tot[np.newaxis, :]*sca_sig, axis=1)
    
    # # Plot optical depth and its composition
    # fig = plt.figure(figsize=(5, 3.75))
    # plt.loglog(all_ene, all_tot[ii, :]+all_com_sca[ii, :], lw=2, color='k', ls='-')
    # tmp_lab = lab_pri.copy()
    # tmp_lab.insert(-1, 'Scattering')
    # plt.xlabel('Energy (keV)')
    # plt.ylabel('Optical depth')
    # plt.xlim([ene_min, ene_max])
    # plt.ylim([3e-3, 1e4])
    # fig.savefig(figp + ff + '_ene_tau_avg.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
    # plt.close()


    # ########
    # # Compare ISM with SN
    # ene_idx = np.argmax(ene > 2000)
    # all_idx = np.argmax(all_ene > 2)
    # sn2ism = avg_sum[ene_idx]/ism_sig[all_idx]
    # print('f_Z', sn2ism/avg_nh)
    # print('N_ISM', sn2ism)


    # ########
    # # XSPEC stuff
    # # Print the simplified XSPEC abundances
    # # Human readable format
    # for jj, itm in enumerate(avg_tot):
    #     print('{:34.34s}{:12.4E}'.format(lab_pri[jj], itm))
    
    # # Format some stuff for XSPEC pasting
    # low_fra = spr_low/spr_avg
    # med_fra = spr_med/spr_avg
    # hig_fra = spr_hig/spr_avg
    # spr_fra = [1., low_fra, med_fra, hig_fra]
    # step = [2, 4, 5, 7, 9]
    # skip = '           0 -1 0 0 1e10 1e10'
    # ski2 = '           1 -1 0 0    1    1'
    # for fra in spr_fra:
    #     print('BREAK')
    #     for jj, itm in enumerate(avg_tot):
    #         out = fra*itm/1e22 if jj == 0 else itm/avg_nh/ism_num[jj] # Different unit for hydrogen
    #         print('{:12.4E} -1 0 0 1e10 1e10'.format(out))
    
    #         if jj in step:
    #             print(skip)

    #     print(''.join(( 7*[skip + '\n'])) + ''.join((18*[ski2 + '\n'])) + skip)
