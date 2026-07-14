import numpy as np

target = 'DOTau'
trial = '020'
bandwidth = '7.5'
path = '/Volumes/disks/meerap/data/' + target + '/simulation/' + target + '_sim_' + trial + '/' + 'visfit/'

data = np.load(path + target + '.gauss.trial' + trial + '.' + bandwidth + 'GHz.post.npz')
print("Keys:", data.files)
for key in data.files:
    print(key, data[key].shape, data[key].dtype)

# --------------------

infile = path + target + '.gauss.trial' + trial + '.' + bandwidth + 'GHz.post.npz'
data = np.load(infile)

samples = data["samples"]   # shape (nsteps, nwalkers, ndim)
logpost = data["logpost"]   # shape (nsteps, nwalkers)

nsteps, nwalkers, ndim = samples.shape
print(f"samples shape: {samples.shape}")
print(f"logpost shape: {logpost.shape}")

# Flatten logpost to 1D, find index of the maximum
flat_logpost = logpost.ravel()               # shape (nsteps*nwalkers,)
flat_samples = samples.reshape(-1, ndim)     # shape (nsteps*nwalkers, ndim)

best_idx = np.argmax(flat_logpost)
best_fit_params = flat_samples[best_idx]
best_logpost = flat_logpost[best_idx]

# Sanity check: recover (step, walker) location too, just for reference
step_idx, walker_idx = np.unravel_index(np.argmax(logpost), logpost.shape)

print("\n--- Best fit (max logpost) ---")
print(f"logpost value: {best_logpost}")
print(f"found at step {step_idx}, walker {walker_idx}")
print(f"best-fit parameters: {best_fit_params}")

# Save out for use in imview / model-image generation
outfile = infile.replace(".post.npz", ".bestfit_params.npy")
np.save(outfile, best_fit_params)
print(f"\nSaved best-fit params to {outfile}")
