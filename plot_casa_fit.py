import os, sys
import numpy as np
from casatools import componentlist
from datetime import date
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.coordinates import SkyCoord

sys.path.append('/Volumes/disks/meerap/codes') # This is the path where calduct_tools and targets_dict are stored

from calduct_tools import deproject_vis, export_vis, shift_coords
from targets_dict import targ as targ_dict

top_dir = '/Volumes/disks/meerap/data/'

targ_name = 'FPTau'
run = 'p1' # # tag used in the wtfx ms filename, if required
tag= 'B4_hi'
cf_date = '2026-06-05' # of the casa_fit
wt_date = '2026-06-09' # date of statwt copy 
shape = 'G' # 'G' or 'D' depending on whether you want the Gaussian or disk fit

data_dir = top_dir + targ_name + '/'
wt_dir = data_dir + 'statwt_' + wt_date + '/' # Folder containing the copied .ms directory
cl_dir = data_dir + 'casa_fit_' + cf_date + '/' # Folder containing .cl directories
wtfx_MS = wt_dir + targ_name + '_' + tag + '.contap1_wtfx.ms' 
cl_file = cl_dir + targ_name + '_complist_' + shape + '.cl'
eb_fol = data_dir + 'deproj_files_' + str(date.today()) + '/'

# Remove previous runs from today, and re-make the relevant folder
os.system('rm -rf ' + eb_fol)
os.system('mkdir -p ' + eb_fol)

# Find phase center RA and DEC
direction_info = listobs(vis = wtfx_MS)['field_0']['direction']
pc_RA = direction_info['m0']['value'] * u.rad
pc_DEC = direction_info['m1']['value'] * u.rad

# Get the uvmodefit details
cl.open(cl_file)
clist = cl.getcomponent(0)
cl.close()

# Get RA and DEC from targ_dict and shift_coords
RA_i = targ_dict[targ_name]['RA']
DEC_i = targ_dict[targ_name]['DEC']

propmot_RA = targ_dict[targ_name]['mua']
propmot_DEC = targ_dict[targ_name]['mud']

observTime = listobs(vis = wtfx_MS)['BeginTime']

RA_0, DEC_0 = shift_coords(RA_i, DEC_i, propmot_RA, propmot_DEC, observTime)
disk_coord = SkyCoord(RA_0, DEC_0, unit = (u.hourangle, u.degree))

disk_RA = disk_coord.ra.to(u.rad)
disk_DEC = disk_coord.dec.to(u.rad)

pos_angle = clist['shape']['positionangle']['value'] # in degrees
# disk_RA = clist['shape']['direction']['m0']['value'] # in rad
# disk_DEC = clist['shape']['direction']['m1']['value'] # in rad

# Calculate the RA/DEC offset and convert to arcsec
offset_RA = (disk_RA - pc_RA).to(u.arcsec).value
offset_DEC = (disk_DEC - pc_DEC).to(u.arcsec).value

minoraxis = clist['shape']['minoraxis']['value'] # in arcmin
majoraxis = clist['shape']['majoraxis']['value'] # in arcmin

inclination = (np.arccos(minoraxis/majoraxis) * u.rad).to(u.deg) # use trigonometry to find the inclination

# Use export_vis to pull data needed for deproject_vis
wtfx_ev_path = eb_fol + 'exported_vis_data'
export_vis(wtfx_MS, wtfx_ev_path)
wtfx_ev = np.load(wtfx_ev_path + '.npz')

deprjvs_wtfx = deproject_vis(wtfx_ev, bins = np.arange(0, 600, 10), incl = inclination, PA = pos_angle, offx = offset_RA, offy = offset_DEC)

# Do deproject_vis on the model from uvmodelfit
# Split off the model from the rest of the data first
ft(vis = wtfx_MS, complist = cl_file)

model_split_MS = eb_fol + '/DOTau_model_only.ms'
split(vis = wtfx_MS, outputvis = model_split_MS, datacolumn = 'model')

model_ev_path = eb_fol + 'exported_vis_model'
export_vis(model_split_MS, model_ev_path)
model_ev = np.load(model_ev_path + '.npz')

deprjvs_model = deproject_vis(model_ev, bins = np.arange(0, 600, 10), incl = inclination, PA = pos_angle, offx = offset_RA, offy = offset_DEC)

# Plot the figure
fig = plt.figure()
gs=GridSpec(3, 1)

# Make plot of real twice as tall as plot of imaginary
realax = fig.add_subplot(gs[0:2, 0])
imgax = fig.add_subplot(gs[2, 0])

realax.errorbar(deprjvs_wtfx.rho_uv, abs(deprjvs_wtfx.vis_prof), yerr = deprjvs_wtfx.err_std, fmt = '.', label = 'Data')
realax.plot(deprjvs_model.rho_uv, abs(deprjvs_model.vis_prof), '-', color = 'tab:red', label = 'Model')
realax.set_title(targ_name)
realax.set_ylabel('real amplitude (Jy)') # Not 100% positive about the units
realax.xaxis.set_ticklabels([])
realax.legend()

imgax.errorbar(deprjvs_wtfx.rho_uv/1e3, (1j * deprjvs_wtfx.vis_prof), yerr = deprjvs_wtfx.err_std, fmt = '.')
imgax.plot(deprjvs_model.rho_uv/1e3, (1j * deprjvs_model.vis_prof), '-', color = 'tab:red')
imgax.set_ylabel('img amplitude (Jy)')
imgax.set_xlabel('uv distance (klam)')

# In order to have the img plot have the same y-axis scale as the real plot. (It's not *perfect*, but it's very close.)
ylim = realax.get_ylim()
y_bound = (ylim[1] - ylim[0])/2
imgax.set_ylim([-y_bound/2, y_bound/2])

plt.tight_layout()
plt.savefig(eb_fol + 'amp_v_uvdist_' + shape + '.png')

plt.show(block=True) # IF YOU DELETE THIS CASA WILL CRASH WHEN YOU TRY TO CLOSE THE PLOT.
