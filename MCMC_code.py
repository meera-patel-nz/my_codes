import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

sys.path.append('/opt/anaconda3/lib/python3.7/site-packages')
import emcee
import corner

sys.path.append('/Volumes/disks/meerap/codes') # pulls mcmc_tools.py copied into codes directory 
from mcmc_tools import * 

### User Controls

""" User definitions and setups """

# Choose target
targ = 'DOTau'
trial = '029'
tag = 'B4_hi'
wgt_rescl = 0.33 # Weight rescaling factor - ??? 
mtype = 'gauss' # model type 
bandwidth_GHz = '1.875' # change per run!
outfile = targ + '.' + mtype + '.trial' + trial + '.' + bandwidth_GHz + 'GHz' # Assign the output filename prefix

# simulated data (true inputs)
true_flux_Jy = 0.05
true_FWHM = 0.393551
true_sigma = true_FWHM / 2.354
true_dx = 0.0639611
true_dy = -0.487347
true_angle = -66.6743 
true_PA = (-true_angle-90) % 180.0
true_axis_ratio = 0.557551 
true_inc = np.degrees(np.arccos(true_axis_ratio))

# paths: Location of visibilities, location of posteriors directory 
visdir = '/Volumes/disks/meerap/data/' + targ + '/simulation/' + targ + '_sim_' + trial + '/'

# prior information
uvlim = 1000. # Used to truncate uv data - why would we want to do this? - if data set gets complicated at longer lambda, truncate and fit gaussian at shorter wavelengths. (effectively fitting all baselines as 1000 big number)

pri_type = [
        'uniform', # flux
        'normal', # dx
        'normal', # dy
        'uniform', # sigma
        'uniform', # PA
        'uniform', # inclination
        'uniform', # logf
] # "prior type" 5 entries for gauss

pri_pars = [
        [0, 0.15],
        [true_dx, 0.025],
        [true_dy, 0.025],
        [0.1, 5.0],
        [0.0, 180.0],
        [0.0, 89.0],
        [-20.0, 10.0]] # "prior parameters" Edit for simulated disk !!

""" Prior parameters:
[
    [flux_lo, flux_hi],
    [dx_mean, dx_width],
    [dy_mean, dy_width],
    [sigma_lo, sigma_hi],
    [PA_lo, PA_hi],
    [incl_lo, incl_hi],
    [logf_lo, logf_hi]
]
"""

# MCMC parameters
append = False # Has there been a previous run that this run should add to?
nsteps, ninit = 1000, 100 # Number of steps for the full mcmc, number of steps for the initial run to deal with stray walkers
nwalk, nthread = 64, 8 # Number of walkers, number of threads
maxtau = 500
burnfactor = 10
thinfactor = 0.5
cutfactor = 100

# location of posteriors subdirectory
postdir = visdir + 'visfit/' # Folder from which to start if previous walkers exist, or folder in which to dump everything
postdir = os.path.join(visdir, 'visfit/')
os.makedirs(postdir, exist_ok=True)

""" ======================================================================= """

# prior evaluators
def uniform_prior(par, p):
    if np.logical_and((par >= p[0]), (par <= p[1])):
        return 0
    else:
        return -np.inf

def normal_prior(par, p):
    return -0.5 * ((par - p[0]) / p[1])**2


""" Set function definitions based on model type selection """
if mtype == 'gauss':
    # dimensionality
    ndim = 7

    # notation
    plbls = ['flux', 'dx', 'dy', 'sigma', 'PA', 'incl', 'logf']
    punits = ['mJy', 'arcsec', 'arcsec', 'arcsec', 'deg', 'deg', '']

    # visibility model
    def vis_model(pars, u, v):
        # unpack all fitted parameters:
        flux = pars[0]
        dx = pars[1]
        dy = pars[2]
        sigma = pars[3]
        PA = pars[4]
        incl = pars[5]
        
        #set the projection geometry
        theta = 0.5 * np.pi - np.radians(PA)
        mu = np.cos(np.radians(incl))
        scl = np.pi / (180 * 3600) # Convert arcsec to radians
        sigma_radians = sigma * scl
        uu = (u * np.cos(theta) + v * np.sin(theta)) * sigma_radians
        vv = (-u * np.sin(theta) + v * np.cos(theta)) * sigma_radians * mu # Has been rotated so that all "stretch" is along this axis

        # define the model fixed at the phase center
        uuvv = (uu**2 + vv**2)**0.5
        mvis = flux * np.exp(-2 * np.pi**2 * uuvv**2) + 1j*np.zeros_like(uuvv)

        # phase shift to treat the offsets
        dx_rad = -dx * scl
        dy_rad = -dy * scl
        phase_shift = np.exp(-2 * np.pi * 1j*(u * dx_rad + v * dy_rad))
        mvis *= phase_shift

        return mvis

else:
    raise ValueError(f'Unsupported model type: {mtype}')

""" Probability functions """
# log-prior
def log_prior(pars):
    lnT = 0.0

    for par, prior_type, prior_parameters in zip(
        pars,
        pri_type,
        pri_pars
    ):

        if prior_type == 'uniform':
            lnT += uniform_prior(par, prior_parameters)

        elif prior_type == 'normal':
            lnT += normal_prior(par, prior_parameters)

        else:
            raise ValueError(
                f'Unknown prior type: {prior_type}'
            )

    return lnT

''' updated above - this is original code
def log_prior(pars):
    lnT = 0
    for ii in range(len(pars)):
        cmd = pri_type[ii]+'_prior(pars['+str(ii)+'], '+str(pri_pars[ii])+')' # Allows us to switch between the two prior definitions with minimal code
        lnT += eval(cmd) # Python's "eval()" method, which evaluates a string as code
    return lnT
'''
# log-likelihood
def log_likelihood(pars, u, v, vis, wgt):
    model_vis = vis_model(pars, u, v)
    logf = pars[6]
    var = (1 / wgt) + np.absolute(model_vis)**2 * np.exp(2 * logf)
    return -0.5 * np.sum(np.absolute(vis - model_vis)**2 / var + np.log(var))

# log-posterior
def log_posterior(pars, u, v, vis, wgt):

    lp = log_prior(pars)

    if np.isfinite(lp):
        return log_likelihood(pars, u, v, vis, wgt) + lp
    else:
        return -np.inf

""" Inference """
# caution with internal multithreading
if (nthread > 1): os.environ["OMP_NUM_THREADS"] = "1"


# ------------------------------------------------------------------
# Load visibility data
# ------------------------------------------------------------------
npz_file = os.path.join(visdir, f'exported_vis_data_{trial}.npz')
print('Loading visibility data from:', npz_file)
if not os.path.exists(npz_file):
    raise FileNotFoundError(f'Visibility file does not exist: {npz_file}')

with np.load(npz_file) as data:
    required_keys = {'u', 'v', 'Vis', 'Wgt', 'nu'}
    missing = required_keys.difference(data.files)
    if missing:
        raise KeyError(f'Missing required NPZ keys: {sorted(missing)}')
    u_, v_, vis_, wgt_, nu = (np.asarray(data[k]).copy() for k in ('u', 'v', 'Vis', 'Wgt', 'nu'))

if not (u_.shape == v_.shape == vis_.shape == wgt_.shape == nu.shape):
    raise ValueError('u, v, Vis, Wgt, and nu do not have matching shapes.')
if not np.iscomplexobj(vis_):
    raise TypeError('Vis is not a complex array. Check the NPZ export script.')

# Flatten to 1D vectors
u_, v_, vis_, wgt_, nu = (a.ravel() for a in (u_, v_, vis_, wgt_, nu))
wgt_ *= wgt_rescl  # global weight correction

# Remove unusable points, then apply uv cutoff (uvlim is in klambda)
valid = np.isfinite(u_) & np.isfinite(v_) & np.isfinite(vis_.real) & \
        np.isfinite(vis_.imag) & np.isfinite(wgt_) & np.isfinite(nu) & (wgt_ > 0.0)
uv_cutoff = 1e3 * uvlim
keep = valid & (np.sqrt(u_**2 + v_**2) <= uv_cutoff)

u, v, vis, wgt, nu_fit = u_[keep], v_[keep], vis_[keep], wgt_[keep], nu[keep]
if len(u) == 0:
    raise ValueError('No valid visibility data remain after filtering.')

freq = np.average(nu_fit, weights=wgt) / 1e9
actual_bandwidth_GHz = (np.nanmax(nu_fit) - np.nanmin(nu_fit)) / 1e9

print('Valid fraction:', np.mean(valid), '| Removed:', np.sum(~valid))
print(f'Fitted {len(u)}/{len(u_)} points ({len(u)/len(u_):.2%})')
print('u range:', np.nanmin(u), np.nanmax(u), '| v range:', np.nanmin(v), np.nanmax(v))
print('Max uv distance:', np.nanmax(np.sqrt(u**2 + v**2)), '| UV cutoff:', uv_cutoff, 'lambda')
print('Freq range [Hz]:', np.nanmin(nu_fit), np.nanmax(nu_fit))
print('Weighted avg freq [GHz]:', freq, '| Actual span [GHz]:', actual_bandwidth_GHz,
      '| Requested label [GHz]:', bandwidth_GHz)
print('Vis dtype:', vis.dtype, '| amp range:', np.nanmin(np.abs(vis)), np.nanmax(np.abs(vis)))
print('Weight range:', np.nanmin(wgt), np.nanmedian(wgt), np.nanmax(wgt))

# Initialize the walkers, starting from the previous run
if append:
    if os.path.exists(postdir+outfile+'.post.npz'):
        pre_samples = np.load(postdir+outfile+'.post.npz')['samples']
        pre_logpost = np.load(postdir+outfile+'.post.npz')['logpost']
        p00 = pre_samples[-1,:,:]

        if p00.shape[1] != ndim:
            raise ValueError(
                'Previous chain has '+ str(p00.shape[1]) + ' parameters, but this model expects ' + str(ndim))
    
    else:
        print('I cannot find the file to append samples.  Exiting')
        sys.exit()


# Initialize the walkers, starting from random posterior draws
else:
    p0 = np.empty((nwalk, ndim))
    for ip in range(ndim):
        _ = 'np.random.'+pri_type[ip]+'('+str(pri_pars[ip][0])+', ' # Allows us to switch between random and uniform distribution, with different keywords
        _ += str(pri_pars[ip][1])+', '+str(nwalk)+')'
        p0[:,ip] = eval(_) # Python's "eval()" method

    # Quick initial run to mitigate stray walkers
    with Pool(processes=nthread) as pool:
        isampler = emcee.EnsembleSampler(nwalk, ndim, log_posterior,
                                            pool=pool, args=(u, v, vis, wgt))
        isampler.run_mcmc(p0, ninit, progress=True)
    isamples = isampler.get_chain()
    lop0 = np.quantile(isamples[-1,:,:], 0.25, axis=0)
    hip0 = np.quantile(isamples[-1,:,:], 0.75, axis=0)
    p00 = [np.random.uniform(lop0, hip0, ndim) for iw in range(nwalk)]
    p00 = np.reshape(p00, p0.shape)

# Full MCMC run
with Pool(processes=nthread) as pool:
    sampler = emcee.EnsembleSampler(nwalk, ndim, log_posterior,
                                    pool=pool, args=(u, v, vis, wgt))
    sampler.run_mcmc(p00, nsteps, progress=True)
samples = sampler.get_chain()
logpost = sampler.get_log_prob()
if append: 
    samples = np.concatenate((pre_samples, samples))
    logpost = np.concatenate((pre_logpost, logpost))

# samples is the raw chain
samples_ = mcmc_out(samples, logpost, maxtau=maxtau, cutfactor=cutfactor,
                    burnfactor=burnfactor, thinfactor=thinfactor) # This is the chain with burn removed and thinned out

# Save the outputs
postfile = postdir+outfile+'.post.npz'
print('Posterior samples were saved to '+postfile)
np.savez(postfile, samples=samples, chain=samples_, logpost=logpost)

# Find the best-fit parameter set: highest log-posterior
best_step, best_walker = np.unravel_index(
    np.argmax(logpost),
    logpost.shape
)

bestfit_pars = samples[best_step, best_walker, :]

print('Best-fit parameters:', bestfit_pars)

bestfit_file = postdir + outfile + '.bestfit_params.npy'
np.save(bestfit_file, bestfit_pars)

print('Best-fit parameters saved to ' + bestfit_file)


""" Diagnostics """
### plot the walker traces
# identify outlier walkers (based on lnprob)
nstep, nwalk, ndim = samples.shape
ncut = round(nstep / cutfactor)
dev_ = (np.median(logpost[ncut:,:], axis=0) - \
        np.median(logpost[ncut:,:])) / np.std(logpost[ncut:,:])
out_ix = np.where(np.abs(dev_) >= 2)
_samples  = np.delete(samples, out_ix, axis=1) # _samples is samples thinned in some other way
_logposts = np.delete(logpost, out_ix, axis=1)
_nwalk = _samples.shape[1]

fig, ax = plt.subplots(nrows=ndim+1, ncols=1, figsize=(6, 12),
                        constrained_layout=True, sharex=True)
_samples[:,:,0] *= 1e3
blob = np.dstack((_samples, np.reshape(_logposts, (nstep, _nwalk, 1))))
steps = np.arange(nstep)
for ip in range(ndim+1):
    for iw in range(_nwalk):
        ax[ip].plot(steps, blob[:,iw,ip], '-k', alpha=0.05)
    if ip < ndim:
        ax[ip].set_ylabel(plbls[ip])
    else:
        ax[ip].set_ylabel('log(prob)')
fig.savefig(postdir+outfile+'.traces.png')
plt.show(block=True)


print(_samples.shape, samples_.shape)

### plot the pairwise covariances - corner plot:

'''
1. Make a copy of the processed posterior samples (so that plotting unit convensions don't alter the sample itself)
2. Convert flux from Jy to mJy for corner plot only
3. Confirm posterior has expected number of param
4. Set readable axis labels
5. Set fixed axis ranges - plotting limits so may need to be adjusted if posterior is clipped
6. Define posterior probability levels shown by the contours (enclosed probability fractions)
7. CREATE CORNER PLOT :)
8. Retrieve the individual Matplotlib axes
9. Format every panel consistently
'''

corner_samples = samples_.copy()
corner_samples[:,0] *= 1e3

if corner_samples.shape[1] !=ndim:
    raise ValueError('Corner-plot samples contain ' + str(corner_samples.shape[1]) + ' parameters, but ndim is ' + str(ndim))

print('Corner-plot sample shape:', corner_samples.shape)

corner_labels = [
    'Flux [mJy]',
    r'$\Delta$RA [arcsec]',
    r'$\Delta$Dec [arcsec]',
    r'$\sigma$ [arcsec]',
    'PA [deg]',
    'Inclination [deg]',
    r'$\log f$'
]

corner_ranges = [
    (0.0, 150.0),       # flux [mJy]
    (-0.2, 0.3),        # dx [arcsec]
    (-0.8, -0.2),       # dy [arcsec]
    (0.01, 0.6),        # sigma [arcsec]
    (0.0, 180.0),      # PA [deg]
    (0.0, 89.0),        # inclination [deg]
    (-20.0, 10.0)       # logf
]

corner_levels = (
    1.0 - np.exp(-0.5 * 1.0**2),
    1.0 - np.exp(-0.5 * 2.0**2),
    1.0 - np.exp(-0.5 * 3.0**2)
)


n_params = corner_samples.shape[1]  # should be ndim

fig = corner.corner(
    corner_samples,
    labels=corner_labels,
    range=corner_ranges,
    figsize=(2.6 * n_params, 2.6 * n_params),  # scales with number of params -- key fix
    bins=30,
    smooth=1.0,
    smooth1d=1.0,
    levels=corner_levels,
    show_titles=True,
    title_quantiles=[0.1585, 0.5, 0.8415],
    title_fmt='.2f',                # fewer decimals = shorter titles = less overlap
    title_kwargs={'fontsize': 8, 'pad': 4},
    quantiles=[0.1585, 0.5, 0.8415],
    plot_datapoints=True,
    plot_density=True,
    plot_contours=True,
    fill_contours=True,
    data_kwargs={'alpha': 0.10},
    label_kwargs={'fontsize': 8, 'labelpad': 6},
    hist_kwargs={'linewidth': 1.0},
    contour_kwargs={'linewidths': 1.0},
    max_n_ticks=3,                   # fewer tick marks = less clutter
    use_math_text=True
)

axes = np.array(fig.axes).reshape((n_params, n_params))

for row in range(n_params):
    for col in range(n_params):
        ax = axes[row, col]
        if col > row:
            continue  # upper triangle is empty/unused -- skip formatting it

        ax.tick_params(axis='both', which='major', labelsize=7, pad=3, length=3, width=0.6)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))

        if row == col:
            # diagonal histograms: hide the 0-1 density axis entirely -- not meaningful, was the main source of clutter
            ax.set_yticks([])
            ax.set_yticklabels([])

        if row == n_params - 1:
            for tick_label in ax.get_xticklabels():
                tick_label.set_rotation(45)
                tick_label.set_horizontalalignment('right')
            ax.xaxis.labelpad = 10

        if col == 0 and row > 0:
            ax.yaxis.labelpad = 10

fig.subplots_adjust(left=0.09, right=0.98, bottom=0.09, top=0.93, wspace=0.12, hspace=0.06)

corner_file = postdir + outfile + '.corner.png'
fig.savefig(corner_file, dpi=200, facecolor='white')
print('Corner plot saved to:', corner_file)
plt.show(block=True)
plt.close(fig)

### Save simple marginalized posterior summaries

# Make a copy so that converting flux to mJy does not change samples_
summary_samples = samples_.copy()

# Convert the flux column from Jy to mJy for reporting
summary_samples[:, 0] *= 1e3

# Calculate the 15.85th, 50th, and 84.15th percentiles
clevs = [15.85, 50.0, 84.15]
CI = np.percentile(summary_samples, clevs, axis=0)

# Open the output text file
output = open(
    postdir + outfile + '.output.txt',
    'w'
)

# Write the weighted average frequency
output.write('\nnu = %.2f GHz' % freq)

# Write the median and upper/lower uncertainties for every parameter
for j in range(len(plbls)):
    output.write(
        '\n%s = %.3f +%.3f / -%.3f %s'
        % (
            plbls[j],
            CI[1, j],
            CI[2, j] - CI[1, j],
            CI[1, j] - CI[0, j],
            punits[j]
        )
    )

output.flush()
output.close()

""" True-versus-recovered sanity checks """

print('True input flux (Jy):', true_flux_Jy)
print('Recovered median flux (Jy):', CI[1, 0] * 1e-3)

print('True sigma (arcsec):', true_sigma)
print('Recovered median sigma (arcsec):', CI[1, 3])

print('True dx (arcsec):', true_dx)
print('Recovered median dx (arcsec):', CI[1, 1])

print('True dy (arcsec):', true_dy)
print('Recovered median dy (arcsec):', CI[1, 2])

print('True PA (deg):', true_PA)
print('Recovered median PA (deg):', CI[1, 4])

print('True inclination (deg):', true_inc)
print('Recovered median inclination (deg):', CI[1, 5])

print('Recovered median logf:', CI[1, 6])


true_pars = np.array([true_flux_Jy, true_dx, true_dy, true_sigma, true_PA, true_inc, -15.0])

lp_true = log_posterior(true_pars, u, v, vis, wgt)
lp_best = log_posterior(bestfit_pars, u, v, vis, wgt)

print('log_posterior at TRUE params:     ', lp_true)
print('log_posterior at BEST-FIT params: ', lp_best)
print('difference (best - true):         ', lp_best - lp_true)


