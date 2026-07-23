import os
import sys
import numpy as np
from casatools import table
tb = table()

sys.path.append('/Volumes/disks/meerap/codes')
from calduct_tools import imparamcalc, has_corrected_column

# ------------------------------------------------------------------
# User settings
# ------------------------------------------------------------------
targ = 'DOTau'
trial = '029'
tag = 'B4_hi'

vis_ms = '/Volumes/disks/meerap/data/DOTau/simulation/DOTau_sim_029/DOTau_sim_029.alma.cycle5.1.noisy.ms'
outdir = '/Volumes/disks/meerap/data/' + targ + '/simulation/' + targ + '_sim_' + trial + '/see_model/'

bestfit_file = ('/Volumes/disks/meerap/data/' + targ + '/simulation/' + targ + '_sim_' + trial + '/visfit/visfitDOTau.gauss.trial029.1.875GHz.bestfit_params.npy')

os.system('rm -rf ' + outdir)
os.system('mkdir -p ' + outdir)
os.system('cp -r ' + vis_ms + ' ' + outdir)
vis_copy = outdir + os.path.basename(vis_ms)

# Initialize MODEL_DATA / CORRECTED_DATA columns
clearcal(vis=vis_copy, addmodel=True)

# ------------------------------------------------------------------
# Get uv coordinates (meters -> wavelengths)
# ------------------------------------------------------------------
tb.open(vis_copy)
uvw = tb.getcol('UVW')  # 3 x N array (u, v, w)
u_meters = uvw[0, :]
v_meters = uvw[1, :]
tb.close()

tb.open(vis_copy + '/SPECTRAL_WINDOW')
freq_dict = tb.getvarcol('CHAN_FREQ')
tb.close()

all_freqs = []
for row_key in freq_dict.keys():
    all_freqs.extend(freq_dict[row_key].flatten())
mean_freq = np.mean(all_freqs)

c = 299792458.0  # speed of light, m/s
wavelength = c / mean_freq
u_lambda = u_meters / wavelength
v_lambda = v_meters / wavelength

# ------------------------------------------------------------------
# Visibility model -- copied directly from the MCMC fitting script
# (mtype == 'gauss' branch). pars[0:6] = flux, dx, dy, sigma, PA, incl
# ------------------------------------------------------------------
def vis_model(pars, u, v):
    flux = pars[0]
    dx = pars[1]
    dy = pars[2]
    sigma = pars[3]
    PA = pars[4]
    incl = pars[5]

    theta = 0.5 * np.pi - np.radians(PA)
    mu = np.cos(np.radians(incl))
    scl = np.pi / (180 * 3600)  # arcsec -> radians
    sigma_radians = sigma * scl
    uu = (u * np.cos(theta) + v * np.sin(theta)) * sigma_radians
    vv = (-u * np.sin(theta) + v * np.cos(theta)) * sigma_radians * mu

    uuvv = (uu**2 + vv**2)**0.5
    mvis = flux * np.exp(-2 * np.pi**2 * uuvv**2) + 1j * np.zeros_like(uuvv)

    dx_rad = -dx * scl
    dy_rad = -dy * scl
    phase_shift = np.exp(-2 * np.pi * 1j * (u * dx_rad + v * dy_rad))
    mvis *= phase_shift

    return mvis

# ------------------------------------------------------------------
# Load your actual best-fit parameters (only pars[0:6] are used by vis_model)
# ------------------------------------------------------------------
bestfit_pars = np.load(bestfit_file)
print('Best-fit parameters loaded:', bestfit_pars)
pars = bestfit_pars[:6]

model_visibilities = vis_model(pars, u_lambda, v_lambda)

# ------------------------------------------------------------------
# Write model visibilities into MODEL_DATA
# ------------------------------------------------------------------
tb.open(vis_copy, nomodify=False)
existing_data = tb.getcol('DATA')
n_pol, n_chan, n_rows = existing_data.shape
model_3d = np.tile(model_visibilities, (n_pol, n_chan, 1))
tb.putcol('MODEL_DATA', model_3d)
tb.close()

# ------------------------------------------------------------------
# Image the model
# ------------------------------------------------------------------
model_split_MS = outdir + targ + '_model_only.ms'
os.system('rm -rf ' + model_split_MS)
split(vis=vis_copy, outputvis=model_split_MS, datacolumn='model')

cellsize, imsize = imparamcalc(vis_ms)  # your existing helper for cell/imsize

tclean(
        vis=model_split_MS,
        imagename=outdir + targ + '_' + tag + '_model',
        selectdata=True,
        specmode='mfs',
        gridder='standard',
        deconvolver='mtmfs',
        scales=[0],
        pblimit=-0.1,
        weighting='briggs',
        robust=2.0,
        imsize= [1024, 1024],
        cell='0.2arcsec',
        niter=1000,
        nsigma=1.0,
        interactive=False,
        savemodel='modelcolumn'
)

# ------------------------------------------------------------------
# Compute and image the residual: CORRECTED_DATA -= MODEL_DATA
# ------------------------------------------------------------------
if not has_corrected_column(vis_copy):
    raise RuntimeError('CORRECTED_DATA column not found in ' + vis_copy + ' -- clearcal(addmodel=True) should have created it. Check your CASA version/behavior.')

uvsub(vis=vis_copy)

resid_split_MS = outdir + targ + '_resid_only.ms'
os.system('rm -rf ' + resid_split_MS)
split(vis=vis_copy, outputvis=resid_split_MS, datacolumn='corrected')

tclean(
        vis=resid_split_MS,
        imagename=outdir + targ + '_' + tag + '_residual',
        selectdata=True,
        specmode='mfs',
        gridder='standard',
        deconvolver='mtmfs',
        scales=[0],
        pblimit=-0.1,
        weighting='briggs',
        robust=2.0,
        imsize= [1024,1024],
        cell='0.2arcsec',
        niter=1000,
        nsigma=1.0,
        interactive=False,
        savemodel='modelcolumn'
)

print('Model image:    ' + outdir + targ + '_' + tag + '_model.image.tt0')
print('Residual image: ' + outdir + targ + '_' + tag + '_residual.image.tt0')

# View both, e.g.:
# imview(outdir + targ + '_' + tag + '_model.image.tt0')
# imview(outdir + targ + '_' + tag + '_residual.image.tt0')
