import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

# ============================================================
# STEP 0: EDIT THESE
# ============================================================
targ = 'DOTau'
trial = '020'
bandwidth_GHz = '7.5'
mtype = 'gauss'

visdir = '/Volumes/disks/meerap/data/' + targ + '/simulation/' + targ + '_sim_' + trial + '/'
postdir = visdir + 'visfit/'
outfile = targ + '.' + mtype + '.trial' + trial + '.' + bandwidth_GHz + 'GHz'
bestfit_file = postdir + outfile + '.bestfit_params.npy'

# !! FILL THESE IN once confirmed for trial 020 (see grep instructions) !!
axis_ratio = 0.557551   # CONFIRM for trial 020
p = -66.6743            # CONFIRM for trial 020 (raw rotation angle, deg)
p_cor = -p - 90         # matches overwritefits_mainsize.py convention

# path to your existing tclean data image (already exported to .fits)
data_fits_path = '/Volumes/disks/meerap/data/' + targ + '/simulation/' + targ + '_B4_hi_gaussian.' + trial + '.fits'

# pixel scale to use for building the best-fit model image (arcsec/pixel)
pixscale = 0.05   # match whatever overwritefits_mainsize.py used
npix = 512        # size of the model image grid (pixels) -- adjust as needed

# ============================================================
# STEP 1: load best-fit parameters
# ============================================================
pars = np.load(bestfit_file)
print('Best-fit params [flux, dx, dy, sigma, logf]:', pars)

A = pars[0]                     # Jy
x_off = pars[1]                 # arcsec
y_off = pars[2]                 # arcsec
sigma_maj = pars[3]             # arcsec -- already sigma, NOT FWHM
sigma_min = sigma_maj * axis_ratio

print(f'A={A}, x_off={x_off}, y_off={y_off}, sigma_maj={sigma_maj}, sigma_min={sigma_min}')

# ============================================================
# STEP 2: build best-fit Gaussian image (same formula as overwritefits_mainsize.py)
# ============================================================
RA = (np.arange(npix) - (npix - 1) / 2) * pixscale
DEC = (np.arange(npix) - (npix - 1) / 2) * pixscale
x, y = np.meshgrid(RA, DEC)

theta = np.deg2rad(p_cor)

a_coef = (np.cos(theta)**2) / (2 * sigma_maj**2) + (np.sin(theta)**2) / (2 * sigma_min**2)
b_coef = -(np.sin(theta) * np.cos(theta)) / (2 * sigma_maj**2) + (np.sin(theta) * np.cos(theta)) / (2 * sigma_min**2)
c_coef = (np.sin(theta)**2) / (2 * sigma_maj**2) + (np.cos(theta)**2) / (2 * sigma_min**2)

f = A * np.exp(-(a_coef * (x - x_off)**2 + 2 * b_coef * (x - x_off) * (y - y_off) + c_coef * (y - y_off)**2))

# normalize total flux to A, matching overwritefits_mainsize.py convention
current_total_flux = np.nansum(f)
f = f * A / current_total_flux

print('Total flux in best-fit model image:', np.nansum(f))
print('Peak value in best-fit model image:', np.nanmax(f))

# ============================================================
# STEP 3: load the data image for comparison
# ============================================================
data_hdu = fits.open(data_fits_path)
data_img = np.squeeze(data_hdu[0].data)
data_header = data_hdu[0].header

data_cdelt = data_header['CDELT2'] * 3600  # arcsec/pixel, assumes square pixels
data_naxis = data_header['NAXIS2']
data_extent_val = (data_naxis / 2) * data_cdelt
data_extent = [-data_extent_val, data_extent_val, -data_extent_val, data_extent_val]

# ============================================================
# STEP 4: plot side by side
# ============================================================
model_extent_val = (npix / 2) * pixscale
model_extent = [-model_extent_val, model_extent_val, -model_extent_val, model_extent_val]

zoom = 2.0  # arcsec, adjust to taste

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

im0 = axes[0].imshow(data_img, origin='lower', cmap='inferno', extent=data_extent)
axes[0].set_title('Data (tclean)')
axes[0].set_xlim(-zoom, zoom)
axes[0].set_ylim(-zoom, zoom)
axes[0].set_xlabel('RA offset [arcsec]')
axes[0].set_ylabel('Dec offset [arcsec]')
plt.colorbar(im0, ax=axes[0], label='Jy/beam')

im1 = axes[1].imshow(f, origin='lower', cmap='inferno', extent=model_extent)
axes[1].set_title('Best-fit model')
axes[1].set_xlim(-zoom, zoom)
axes[1].set_ylim(-zoom, zoom)
axes[1].set_xlabel('RA offset [arcsec]')
axes[1].set_ylabel('Dec offset [arcsec]')
plt.colorbar(im1, ax=axes[1], label='Jy/pixel')

plt.tight_layout()
plt.savefig(postdir + outfile + '.data_vs_bestfit.png')
plt.show(block=True)

print('Saved comparison figure to', postdir + outfile + '.data_vs_bestfit.png')
