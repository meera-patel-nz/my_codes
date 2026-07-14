from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

# open fits file, get image shape
trial = '023'
fits_path = '/Volumes/disks/meerap/data/DOTau/simulation/DOTau_B4_hi.fits'
out_path = '/Volumes/disks/meerap/data/DOTau/simulation/DOTau_B4_hi_gaussian.' + trial + '.fits'

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

A = 0.05 # peak flux in ____ units ??

a = 0.393551 # FWHM major axis (arcsec)
r = 0.557551 # axis ratio (min/maj)
p = -66.6743 # rotation angle (deg)
p_cor=-p-90

# modifications for formulae

sigma_maj = (a/2.354) # sigma x (maj) 
sigma_min = ((r*a)/2.354) # sigma y (min)
theta = np.deg2rad(p_cor) # rotation angle in rad

# Gaussian formulae

a = (np.cos(theta)**2)/(2*(sigma_maj)**2) + (np.sin(theta)**2)/(2*(sigma_min)**2)
b = -(np.sin(theta)*np.cos(theta))/(2*(sigma_maj)**2) + (np.sin(theta)*np.cos(theta))/(2*(sigma_min)**2)
c = ((np.sin(theta)**2)/(2*(sigma_maj)**2))+ ((np.cos(theta)**2)/(2*(sigma_min)**2))

f = A * np.exp(-(a*(x-x_off)**2 + 2*b*(x-x_off)*(y-y_off) + c*(y-y_off)**2)) # full gaussian equation

# Code to make peak flux an output of total flux.

current_total_flux = np.nansum(f)
desired_total_flux = A  # Jy

f = f * desired_total_flux / current_total_flux

print("Desired total flux:", desired_total_flux)
print("Actual total flux in model:", np.nansum(f))
print("Peak value in model:", np.nanmax(f))

# plt.imshow (to view f)

pixscale = 0.05  # arcsec/pixel, change if your image uses a different pixel size

ny, nx = f.shape

x_extent = (nx / 2) * pixscale
y_extent = (ny / 2) * pixscale

extent = [-x_extent, x_extent, -y_extent, y_extent]

plt.figure(figsize=(7, 6))

plt.imshow(
    f,
    origin='lower',
    cmap='inferno',
    extent=extent
)

plt.colorbar(label='Flux per pixel')
plt.xlabel('RA offset [arcsec]')
plt.ylabel('Dec offset [arcsec]')
plt.title(
    f'Normalized Gaussian model\n'
    f'Total flux = {np.nansum(f):.4e} Jy, '
    f'Peak = {np.nanmax(f):.4e}'
)

zoom_size = 2.0
plt.xlim(-zoom_size, zoom_size)
plt.ylim(-zoom_size, zoom_size)

plt.tight_layout()
plt.show()

# overwrite fits (replace data to values calculated in f)

HDUlist[0].data = f.reshape(data.shape).astype(data.dtype)

HDUlist.writeto(out_path, overwrite=True)
HDUlist.close()

print("Gaussian model written to:", out_path)
