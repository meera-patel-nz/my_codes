import os, sys
import numpy as np
from casatools import componentlist
from datetime import date
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.coordinates import SkyCoord
from casatools import table
import astropy.units as u

sys.path.append('/Volumes/disks/meerap/codes') # This is the path where calduct_tools and targets_dict are stored

from calduct_tools import deproject_vis, export_vis, shift_coords
from targets_dict import targ as targ_dict

top_dir = '/Volumes/disks/meerap/data/'

targ_name = 'DOTau'
run = 'p1' # # tag used in the wtfx ms filename, if required
tag= 'B4_hi'
cf_date = '2026-06-11' # of the casa_fit
wt_date = '2026-06-11' # date of statwt copy 
shapes = ['G', 'D'] # 'G' or 'D' depending on whether you want the Gaussian or disk fit


# ----------- Change above this point for each disk ---------- 

data_dir = top_dir + targ_name + '/'
wt_dir = data_dir + 'statwt_' + wt_date + '/' # Folder containing the copied .ms directory
cl_dir = data_dir + 'casa_fit_' + cf_date + '/' # Folder containing .cl directories
wtfx_MS = wt_dir + targ_name + '_' + tag + '.contap1_wtfx.ms' 
cl_file_wo_shape = cl_dir + targ_name + '_complist_' #might have to check this
eb_fol = data_dir + 'deproj_files_' + str(date.today()) + '/'

# Remove previous runs from today, and re-make the relevant folder
os.system('rm -rf ' + eb_fol)
os.system('mkdir -p ' + eb_fol)

# Find phase center RA and DEC

ms_file = data_dir + tag + '/' + targ_name + '_vis_' + tag + '.ms'
direction_info = listobs(vis=ms_file)['field_0']['direction'] # read .ms phase centre with listobs

pc_RA_rad = direction_info['m0']['value']  # phase centre units in rad
pc_RA_arcsec = ((direction_info['m0']['value']) * u.rad).to(u.arcsec)  # phase centre units in arcsec
pc_DEC_rad = direction_info['m1']['value'] # phase centre units in rad
pc_DEC_arcsec = ((direction_info['m1']['value']) * u.rad).to(u.arcsec) # phase centre units in rad

# Use export_vis to pull data needed for deproject_vis (only needed once)

wtfx_ev_path = eb_fol + 'exported_vis_data'
export_vis(wtfx_MS, wtfx_ev_path)
wtfx_ev = np.load(wtfx_ev_path + '.npz')

for shape in shapes:
    cl_file = cl_file_wo_shape + shape + '.cl'
    # Get the uvmodefit details

    cl = componentlist() # Create a casa component list tool
    cl.open(cl_file) # open .cl file
    comp = cl.getcomponent(0) # open the first fitted component: note, there is only one component so input 0
    cl.close()

    direction = comp['shape']['direction'] # fitted sky position of component

    fit_RA_rad = direction['m0']['value']   # units in radians
    fit_RA_arcsec=((direction['m0']['value']) * u.rad).to(u.arcsec) # units in arcsec
    fit_DEC_rad = direction['m1']['value']  # units in radians
    fit_DEC_arcsec=((direction['m1']['value']) * u.rad).to(u.arcsec) # units in arcsec
    
    pos_angle = comp['shape']['positionangle']['value'] # in degrees

    # This code will calculate the offsets as seen when executing uvmodelfit

    xoff_rad = (fit_RA_rad - pc_RA_rad)
    xoff_arcsec=(fit_RA_arcsec - pc_RA_arcsec)* np.cos(pc_DEC_arcsec)

    yoff_rad = (fit_DEC_rad - pc_DEC_rad)
    yoff_arcsec=(fit_DEC_arcsec - pc_DEC_arcsec)

    xoff_RA = xoff_arcsec.value
    yoff_DEC = yoff_arcsec.value

    minoraxis = comp['shape']['minoraxis']['value'] # in arcmin
    majoraxis = comp['shape']['majoraxis']['value'] # in arcmin

    inclination = (np.arccos(minoraxis/majoraxis) * u.rad).to(u.deg) # use trigonometry to find the inclination

    #-------------_NOTE: NEED TO ADD AUTOMATED MAX BASELINE HERE INSTEAD OF 2500

    #Deproject the data based on the fit
    deprjvs_wtfx = deproject_vis(wtfx_ev, bins = np.arange(0, 2500, 25), incl = inclination, PA = pos_angle, offx = xoff_RA, offy = yoff_DEC)

    #----------------------------------

    # Do deproject_vis on the model from uvmodelfit
    # Split off the model from the rest of the data first
    ft(vis = wtfx_MS, complist = cl_file)

    model_split_MS = eb_fol + '/' + targ_name + '_model_only_' + shape +'.ms'
    split(vis = wtfx_MS, outputvis = model_split_MS, datacolumn = 'model')

    model_ev_path = eb_fol + 'exported_vis_model' + shape
    export_vis(model_split_MS, model_ev_path)
    model_ev = np.load(model_ev_path + '.npz')

    #------------ AGAIN NOTE NEW CHANGES NEED TO BE MADE
    deprjvs_model = deproject_vis(model_ev, bins = np.arange(0,2500, 25), incl = inclination, PA = pos_angle, offx = xoff_RA, offy = yoff_DEC)

    #----------------------

    # Plot the figure
    fig = plt.figure()
    gs=GridSpec(3, 1)

    # Make plot of real twice as tall as plot of imaginary
    realax = fig.add_subplot(gs[0:2, 0])
    imgax = fig.add_subplot(gs[2, 0])

    realax.errorbar(deprjvs_wtfx.rho_uv, abs(deprjvs_wtfx.vis_prof), yerr = deprjvs_wtfx.err_std, fmt = '.', label = 'Data')
    realax.plot(deprjvs_model.rho_uv, abs(deprjvs_model.vis_prof), '-', color = 'tab:red', label = 'Model')

    realax.axhline(y=0, color='k', linestyle='--', linewidth=1)

    realax.set_title(targ_name)
    realax.set_ylabel('real amplitude (Jy)') # Not 100% positive about the units
    realax.xaxis.set_ticklabels([])
    realax.legend()
    
    imgax.errorbar(deprjvs_wtfx.rho_uv/1e3, (1j * deprjvs_wtfx.vis_prof), yerr = deprjvs_wtfx.err_std, fmt = '.')
    imgax.plot(deprjvs_model.rho_uv/1e3, (1j * deprjvs_model.vis_prof), '-', color = 'tab:red')
    
    imgax.axhline(y=0, color='k', linestyle='--', linewidth=1)
    
    imgax.set_ylabel('img amplitude (Jy)')
    imgax.set_xlabel('uv distance (klam)')
    
    # In order to have the img plot have the same y-axis scale as the real plot. (It's not *perfect*, but it's very close.)
    ylim = realax.get_ylim()
    y_bound = (ylim[1] - ylim[0])/2
    imgax.set_ylim([-y_bound/2, y_bound/2])
    
    plt.tight_layout()
    plt.savefig(eb_fol + 'amp_v_uvdist_' + shape + '.png')
    
    plt.show(block=True) # IF YOU DELETE THIS CASA WILL CRASH WHEN YOU TRY TO CLOSE THE PLOT.
