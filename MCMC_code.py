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

"""
sys.path.append('/Volumes/disks/jillian/data/my_code') # edit path 
from calduct_tools import deproject_vis # This uses Jillian's deproject_vis, not Sean's
"""

### User Controls

""" User definitions and setups """

# Choose target
targ = 'DOTau'
trial = '023'
tag = 'B4_hi'
deproj_date = '2026-07-01' # I don't have deproject at the moment
wgt_rescl = 1 # Weight rescaling factor - ??? 
mtype = 'gauss' # model type 
bandwidth_GHz = '7.5' # change per run!
outfile = targ + '.' + mtype + '.trial' + trial + '.' + bandwidth_GHz + 'GHz' # Assign the output filename prefix

''' code for while these were fixed - need to update? 
p = -66.6743                  # raw rotation angle from your sky model (deg)
PA = -p - 90                  # convert to the convention vis_model expects
# PA = -(-66.6743) - 90 = 66.6743 - 90 = -23.3257
axis_ratio = 0.557551          # confirm this is trial 020's actual axis ratio!
'''

# simulated data (true inputs)
true_flux_Jy = 0.05
true_FWHM = 0.393551
true_sigma = true_FWHM / 2.354
true_dx = 0.0639611
true_dy = -0.487347
true_angle = -66.6743 
true_PA = -true_angle-90
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
        [true_dx, 0.1],
        [true_dy, 0.1],
        [0.1, 5.0],
        [-90.0, 90.0],
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
if not append:
    os.system('rm -rf ' + postdir)
os.system('mkdir -p ' + postdir)

if not append:
    os.system('rm -rf ' + postdir)
os.system('mkdir -p ' + postdir)

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
        sigma_radians = pars[3] * scl
        uu = (u * np.cos(theta) + v * np.sin(theta)) * sigma_radians
        vv = (-u * np.sin(theta) + v * np.cos(theta)) * sigma_radians * mu # Has been rotated so that all "stretch" is along this axis

        # define the model fixed at the phase center
        uuvv = (uu**2 + vv**2)**0.5
        mvis = flux * np.exp(-2 * np.pi**2 * uuvv**2) + 1j*np.zeros_like(uuvv)

        # phase shift to treat the offsets
        dx_rad = -dx * scl,
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

    lp = log_priors(pars)

    if np.isfinite(lp):
        return log_likelihood(pars, u, v, vis, wgt) + lp
    else:
        return -np.inf

""" Inference """
# caution with internal multithreading
if (nthread > 1): os.environ["OMP_NUM_THREADS"] = "1"

# Load the visibility data - need to script producing this file!!
_ = np.load(visdir + 'exported_vis_data_' + trial + '.npz')
u_, v_, vis_, wgt_, nu = _['u'], _['v'], _['Vis'], _['Wgt'], _['nu']
wgt_ *= wgt_rescl
freq = np.average(nu, weights=wgt_) / 1e9

""" Unit checks - wavelength should be roughly 1-3mm,"""

print('--- UNIT CHECKS ---')
print('u range:', u_.min(), u_.max())         # meters would be ~1-10000ish; wavelengths ~1e3-1e6ish
print('v range:', v_.min(), v_.max())
print('nu range (Hz):', nu.min(), nu.max())
print('Vis dtype:', vis_.dtype)                # should be complex
print('Vis amplitude range:', np.abs(vis_).min(), np.abs(vis_).max())
print('Wgt range:', wgt_.min(), wgt_.max())
print('number of visibility points:', len(u_))
print('freq (weighted avg, GHz):', freq)

# truncate u,v data if necessary (set uvlim large enough to include everythingm or delete truncation for zero cropping)
uv_data = np.sqrt(u_**2 + v_**2)
u = u_[uv_data <= 1e3 * uvlim]
v = v_[uv_data <= 1e3 * uvlim]
vis = vis_[uv_data <= 1e3 * uvlim]
wgt = wgt_[uv_data <= 1e3 * uvlim]


# Initialize the walkers, starting from the previous run
if append:
    if os.path.exists(postdir+outfile+'.post.npz'):
        pre_samples = np.load(postdir+outfile+'.post.npz')['samples']
        pre_logpost = np.load(postdir+outfile+'.post.npz')['logpost']
        p00 = pre_samples[-1,:,:]

    p00 = pre_samples[-1, :, ;]

    if p00.shape[1] != ndim:
            raise ValueError(
                'Previous chain has '
                + str(p00.shape[1])
                + ' parameters, but this model expects '
                + str(ndim)
            )
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

fig, ax = plt.subplots(nrows=ndim+1, ncols=1, figsize=(5., 8.),
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

corner_samples = samples_.copy
corner_samples[:,0] *= 1e3

if corner_samples.shape[1] !=ndim:
    raise ValueError(
            'Corner-plot samples contain ' + str(corner_samples.shape[1] + ' parameters, but ndim is ' + str(ndim)
)

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
    (-90.0, 90.0),      # PA [deg]
    (0.0, 89.0),        # inclination [deg]
    (-20.0, 10.0)       # logf
]

corner_levels = (
    1.0 - np.exp(-0.5 * 1.0**2),
    1.0 - np.exp(-0.5 * 2.0**2),
    1.0 - np.exp(-0.5 * 3.0**2)
)

fig = corner.corner(
    corner_samples,
    labels=corner_labels,  # Axis labels, in parameter-column order
    range=corner_ranges,   # Keep axis limits consistent across runs
    figsize=(15, 15),      # A square overall figure helps keep each individual panel square
    bins=30,               # Number of histogram bins
    smooth = 1.0,          # Smooth the two-dimensional density and contours slightly
    smooth1d=1.0,          # Smooth the one-dimensional histograms slightly
    smooth1d=1.0,
    levels=corner_levels,  # Draw one-, two-, and three-sigma contour regions
    show_titles=True,      # Show a numerical posterior summary above each diagonal histogram
    title_quantiles=[0.1585, 0.5, 0.8415], # Use the 15.85th, 50th, and 84.15th percentiles for: lower uncertainty, median, and upper uncertainty
    title_fmt='.3f',       # Number of decimal places in the titles
    title_kwargs={
        'fontsize': 11,
        'pad': 12
    },

    # Draw dotted vertical lines on the diagonal histograms at the
    # lower percentile, median, and upper percentile
    quantiles=[0.1585, 0.5, 0.8415],
    # Show the individual posterior samples faintly in the lower triangle
    plot_datapoints=True,
    # Show the estimated two-dimensional density
    plot_density=True,
    # Show contour outlines
    plot_contours=True,
    # Fill the contour regions
    fill_contours=True,
    # Keep the sample points faint so that the contours remain readable
    data_kwargs={'alpha': 0.10, 'markersize': 1.5},
    # Axis-label formatting
    label_kwargs={'fontsize': 12, 'labelpad': 20},
    # One-dimensional histogram formatting
    hist_kwargs={'linewidth': 1.2},
    # Two-dimensional contour formatting
    contour_kwargs={'linewidths': 1.2},
    # Limit the number of displayed tick marks on each axis
    max_n_ticks=4,
    # Display numerical values using normal decimal notation where possible
    use_math_text=True
)

axes = np.array(fig.axes).reshape((ndim, ndim))

for row in range(ndim):
    for col in range(ndim):

        ax = axes[row, col]
        if ax.get_visible():
            ax.set_box_aspect(1)
         ax.tick_params(
            axis='both',
            which='major',
            labelsize=8,
            pad=7,
            length=4,
            width=0.8
        )
         ax.xaxis.set_major_locator(
            plt.MaxNLocator(4)
        )

        ax.yaxis.set_major_locator(
            plt.MaxNLocator(4)
        )
         if row == ndim - 1:
            for tick_label in ax.get_xticklabels():
                tick_label.set_rotation(30)
                tick_label.set_horizontalalignment('right')

            # Increase the distance between the bottom tick values and
            # the x-axis parameter label.
            ax.xaxis.labelpad = 24
        if col == 0 and row > 0:
            ax.yaxis.labelpad = 28
        ax.ticklabel_format(
            axis='both',
            style='plain',
            useOffset=False
        )
fig.subplots_adjust(
    left=0.12,
    right=0.98,
    bottom=0.12,
    top=0.96,
    wspace=0.10,
    hspace=0.10
)

corner_file = postdir + outfile + '.corner.png'

fig.savefig(
    corner_file,
    dpi=300,
    facecolor='white'
)

print('Corner plot saved to:', corner_file)

plt.show(block=True)

plt.close(fig)


### plot the pairwise covariances - original code (edited verison above)]

'''samples_[:,0] *= 1e3
fig = corner.corner(samples_, 
                    levels=(1-np.exp(-0.5*(np.array([1, 2, 3]))**2)), 
                    labels=plbls) # Breaks if number of steps is too small :(
plt.savefig(postdir+outfile+'.corner.png')
plt.show(block=True)


### save simple marginalized posterior summaries # Add printing this out to a file.
output = open(postdir + outfile + '.output.txt', 'w')

clevs = [15.85, 50., 84.15]
CI = np.percentile(samples_, clevs, axis=0)
output.write('\nnu = %.2f GHz' % freq)
for j in range(len(plbls)):
    output.write('\n%s = %.3f +%.3f / -%.3f %s' % \
            (plbls[j], CI[1,j], CI[2,j]-CI[1,j], CI[1,j]-CI[0,j], punits[j]))

output.flush()
output.close()

""" Print flux and sigma checks , orders of magnitude sanity check!!"""

print('True input flux (Jy):', true_flux_Jy)
print('Recovered flux:', CI[1,0]*10^-3)
print('True sigma (arcsec):', true_sigma)
print('Recovered sigma (arcsec):', CI[1,3])
print('True dx (arcsec):', true_dx)
print('Recovered dx (arcsec):', CI[1,1])
print('True dy (arcsec):', true_dy)
print('Recovered dy (arcsec):', CI[1,2])
'''
