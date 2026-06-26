from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

# open fits file, get image shape

fits_path = '/Volumes/disks/meerap/data/DOTau/simulation/DOTau_B4_hi.fits'
out_path = '/Volumes/disks/meerap/data/DOTau/simulation/DOTau_B4_hi_gaussian.fits'

HDUlist = fits.open(fits_path)
header = HDUlist[0].header
data = HDUlist[0].data

print("Original shape:", data.shape)

# get pixel scale and build coordinate grid in arcsec
NAXIS1 = header['NAXIS1']
NAXIS2 = header['NAXIS2']

CDELT1 = header['CDELT1']
CDELT2 = header['CDELT2']

RA = (np.arange(NAXIS1) - (NAXIS1 - 1)/2) * CDELT1 * 3600
DEC = (np.arange(NAXIS2) - (NAXIS2 - 1)/2) * CDELT2 * 3600

# make 2D array (meshgrid) - need to define x and y from data to later use in x-x_off in gaussian formula

x, y = np.meshgrid(RA, DEC) # make 2D coordinate grid

# uvmodelfit parameters (given, change for each target)

x_off = 0.0639611 # delta x (arcsec)
y_off = -0.487347 # delta y (arcsec)

A = 0.0484384 # peak flux 

a = 0.393551 # FWHM major axis (arcsec)
r = 0.557551 # axis ratio (min/maj)
p = -66.6743 # rotation angle (deg)

# modifications for formulae

sigma_maj = (a/2.354) # sigma x (maj) 
sigma_min = ((r*a)/2.354) # sigma y (min)
theta = np.deg2rad(p) # rotation angle in rad

# Gaussian formulae

a = (np.cos(theta)**2)/(2*(sigma_maj)**2) + (np.sin(theta)**2)/(2*(sigma_min)**2)
b = -(np.sin(theta)*np.cos(theta))/(2*(sigma_maj)**2) + (np.sin(theta)*np.cos(theta))/(2*(sigma_min)**2)
c = ((np.sin(theta)**2)/(2*(sigma_maj)**2))+ ((np.cos(theta)**2)/(2*(sigma_min)**2))

f = A * np.exp(-(a*(x-x_off)**2 + 2*b*(x-x_off)*(y-y_off) + c*(y-y_off)**2)) #need to change to delta x rather than x_off (and y equiv)

# plt.imshow (to view f)

plt.imshow(f, origin='lower', cmap='inferno')   
plt.colorbar(label='Flux')
plt.xlabel("RA offset (arcsec)")
plt.ylabel("Dec offset (arcsec)")
plt.title("Gaussian Model (arcsec, centered)")
plt.show()

# overwrite fits (replace data to values calculated in f)

HDUlist[0].data = f.reshape(data.shape).astype(data.dtype)

HDUlist.writeto(out_path, overwrite=True)
HDUlist.close()

print("Gaussian model written to:", out_path)
