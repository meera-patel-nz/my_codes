from casatasks import simobserve
import os
import subprocess
import numpy as np

trial = '020'
sim_folder = '/Volumes/disks/meerap/data/DOTau/simulation/'
fits_file = sim_folder + 'DOTau_B4_hi_gaussian.' + trial + '.fits'
image_file = sim_folder + 'DOTau_gaussian.image'
project_name = 'DOTau_sim_' + trial
project_folder = sim_folder + project_name + '/'

if os.path.exists(image_file):
    os.system(f'rm -rf {image_file}')
    print(f"Deleted previous image: {image_file}")

importfits(fitsimage=fits_file,
           imagename=image_file,
           overwrite=True)

"""
if os.path.exists(project_name):
    os.system(f'rm -rf {project_name}')
    print(f"Deleted previous project folder: {project_name}")
os.makedirs(project_name, exist_ok=True)

# Create component list
cl = componentlist()

PA = -66.6743 # rotation angle (p in deg)
PA_cor=90-abs(PA)
posang = f'{PA_cor}deg'
majax= 0.393551*0.5
majaxst=f'{majax}arcsec'
minax = (0.393551*0.557551)*0.5
minaxst = f'{minax}arcsec' # FWHM minor axis × axis ratio string
# CHANGING SIZE facTOR!!)
cl.addcomponent(
    flux=0.0484384,                 # peak flux in Jy
    fluxunit='Jy',
    dir='J2000 10h00m00.0s -30d00m00.0s',  # source center
    majoraxis=majaxst,     # FWHM major axis
    minoraxis= minaxst,
    positionangle= posang,         # PA after correction
    shape='Gaussian'
)
cl.rename(cl_file)
cl.close()
"""

antennalist_path = '/soft/casa-6.6.0-20-py3.8.el7/lib/py/lib/python3.8/site-packages/casadata/__data__/alma/simmos/alma.cycle5.1.cfg'
antenna_config_name = 'alma.cycle5.1'
#generate measurement set

simobserve(
        project=project_name,
        #complist=cl_file,
        skymodel=image_file,
        inwidth='7.5GHz',
        antennalist=antennalist_path,
        totaltime='180s',
        obsmode='int',
        mapsize=['10arcsec'],
        integration='10s',
        thermalnoise='tsys-manual',
        tau0=0.9,
        overwrite=True)

# move DOTau_sim into data directory 
subprocess.run(['mv', project_name, sim_folder], check=True)

# --------- generate an npz_outfile to be used for MCMC chain -----------------

ms_file = project_folder + project_name + '.' + antenna_config_name + '.noisy.ms'
npz_outfile = project_folder + 'exported_vis_data_' + trial + '.npz'

print('Reading ms file:', ms_file)

# extract from measurement set
ms.open(ms_file)
d = ms.getdata(['u', 'v', 'data', 'weight', 'axis_info'])
ms.close()

u_m = d['u']
v_m = d['v']
vis = d['data']
wgt = d['weight']
freq_hz = d['axis_info']['freq_axis']['chan_freq']

print('u_m shape:', u_m.shape)
print('v_m shape:', v_m.shape)
print('vis shape:', vis.shape)
print('wgt shape:', wgt.shape)
print('freq_hz shape:', freq_hz.shape)

# average over polarization
vis_avg = np.mean(vis, axis=0)
wgt_avg = np.mean(wgt, axis=0)

nchan = vis_avg.shape[0]
nrow = vis_avg.shape[1]

# build matching-shape grids for u, v, weight, freq
freq_hz_grid = np.tile(freq_hz.reshape(nchan, 1), (1, nrow))
u_m_grid = np.tile(u_m.reshape(1, nrow), (nchan, 1))
v_m_grid = np.tile(v_m.reshape(1, nrow), (nchan, 1))
wgt_grid = np.tile(wgt_avg.reshape(1, nrow), (nchan, 1))

# convert u, v from meters to wavelengths
c = 2.99792458e8
wavelength_m = c / freq_hz_grid
u_wave = u_m_grid / wavelength_m
v_wave = v_m_grid / wavelength_m

# flatten everything to 1D
u_out = u_wave.flatten()
v_out = v_wave.flatten()
vis_out = vis_avg.flatten()
wgt_out = wgt_grid.flatten()
nu_out = freq_hz_grid.flatten()

print('u_out shape:', u_out.shape)
print('v_out shape:', v_out.shape)
print('vis_out shape:', vis_out.shape)
print('wgt_out shape:', wgt_out.shape)
print('nu_out shape:', nu_out.shape)

np.savez(npz_outfile, u=u_out, v=v_out, Vis=vis_out, Wgt=wgt_out, nu=nu_out)
print('Saved to', npz_outfile)

