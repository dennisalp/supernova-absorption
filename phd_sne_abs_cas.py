# Dennis Alp 2018-02-18
# Generate a plot showing the encircled counts as a funtion of radius using an observation.
# time python -i /Users/silver/box/bin/phd_abs_cas.py

import os
import time
import pdb
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm

from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.interpolate import griddata

#For LaTeX style font in plots
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)



################################################################
# This plots the radial profile
# Parameters
#ff = "/Users/silver/dat/cxo/cas/13783/repro/acisf13783_repro_evt2_03_10_clean.fits"
#WRK_DIR = "/Users/silver/box/phd/pro/abs/cas/"
#os.chdir(WRK_DIR) #Move to designated directory
#
#nn = 100
#x0 = 4107.304
#y0 = 4086.148
#
#xx = fits.open(ff)[1].data['x']
#yy = fits.open(ff)[1].data['y']
#rr = np.linspace(0, 18, nn)
#ee = np.zeros(nn)
#
#for idx in range(1, nn):
#    within = ((xx - x0)**2 + (yy - y0)**2) < rr[idx]**2
#    outside = ((xx - x0)**2 + (yy - y0)**2) >= rr[idx-1]**2
#    ee[idx] = np.sum(np.logical_and(within, outside))
#
#plt.semilogy(rr, ee/rr)
#plt.axhline(y=np.mean(ee[50:]/rr[50:]), linewidth=2, color = '#7f7f7f', zorder=1)
#plt.axvline(x=2, linewidth=2, color = '#7f7f7f', zorder=1)
#plt.axvline(x=3, linewidth=2, color = '#7f7f7f', zorder=1)
#plt.axvline(x=5, linewidth=2, color = '#7f7f7f', zorder=1)
#plt.ylabel('Radial profile [cts]')
#plt.xlabel('Radius [arcsec]')
#plt.show()



################################################################
# Radial profile of simulated PSF
ff = "/Users/silver/dat/cxo/cas/13783/repro/acisf13783_repro_psf.psf"
ff = "/Users/silver/dat/cxo/cas/6690/repro/acisf06690_repro_psf.psf"
ff = "/Users/silver/dat/cxo/cas/16946/repro/acisf16946_repro_psf.psf"
WRK_DIR = "/Users/silver/box/phd/pro/abs/cas/"
os.chdir(WRK_DIR) #Move to designated directory

def help_gauss(dummy, x0, y0, aa, sig):
    xx = np.arange(0, nx)
    yy = np.arange(0, ny)[:,None]
    xx = xx-x0
    yy = yy-y0
    rr = xx**2+yy**2
    last_gau = aa*np.exp(-rr/sig**2)
    return np.reshape(last_gau, nx*ny)
 
img = fits.open(ff)[0].data
nx = img.shape[0]
ny = img.shape[1]
x0 = 157
y0 = 157
guess = np.array([x0, y0, 0.01, 4])
pars, covar = curve_fit(help_gauss, np.arange(0, nx*ny), np.reshape(img, nx*ny), guess)
x0 = pars[0]
y0 = pars[1]
yy, xx = np.indices((img.shape))
tot = np.sum(img)

# Inside source region
ii = np.sqrt((xx-x0)**2+(yy-y0)**2) < 20
aa = np.sum(img[ii])

ii = np.sqrt((xx-x0)**2+(yy-y0)**2) < 30
jj = np.sqrt((xx-x0)**2+(yy-y0)**2) > 20
kk = np.logical_and(ii, jj)
bb = np.sum(img[kk])

ii = np.sqrt((xx-x0)**2+(yy-y0)**2) < 50
jj = np.sqrt((xx-x0)**2+(yy-y0)**2) > 30
kk = np.logical_and(ii, jj)
cc = np.sum(img[kk])

ii = np.sqrt((xx-x0)**2+(yy-y0)**2) > 50
dd = np.sum(img[ii])

print(aa/tot, bb/tot, cc/tot, dd/tot)

pdb.set_trace()
