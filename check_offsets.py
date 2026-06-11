#This code reads the fitted RA and Dec from .cl

from casatools import componentlist
import numpy as np
import astropy.units as u

top_dir = '/Volumes/disks/meerap/data/' # where the data lives
targ_name = 'FPTau' # change to target name
tag = 'B4_hi'
shape = 'G'  # change to 'D' for disk
cf_date = '2026-06-09' # date of the casa_fit

data_dir = top_dir + targ_name + '/'
cl_dir = data_dir + 'casa_fit_' + cf_date + '/' # Folder containing .cl directories
cl_file = cl_dir + targ_name + '_complist_' + shape + '.cl'

cl = componentlist() # Create a casa component list tool
cl.open(cl_file) # open .cl file
comp = cl.getcomponent(0) # open the first fitted component: note, there is only one component so input 0
direction = comp['shape']['direction'] # fitted sky position of component

fit_RA_rad = direction['m0']['value']   # units in radians
fit_RA_arcsec=((direction['m0']['value']) * u.rad).to(u.arcsec) # units in arcsec
fit_DEC_rad = direction['m1']['value']  # units in radians
fit_DEC_arcsec=((direction['m1']['value']) * u.rad).to(u.arcsec) # units in arcsec

cl.close()

# This code reads the phase centre from .ms

from casatools import table
import astropy.units as u

ms_file = data_dir + tag + '/' + targ_name + '_vis_' + tag + '.ms'
direction_info = listobs(vis=ms_file)['field_0']['direction'] # read .ms phase centre with listobs

pc_RA_rad = direction_info['m0']['value']  # phase centre units in rad
pc_RA_arcsec = ((direction_info['m0']['value']) * u.rad).to(u.arcsec)  # phase centre units in arcsec
pc_DEC_rad = direction_info['m1']['value'] # phase centre units in rad
pc_DEC_arcsec = ((direction_info['m1']['value']) * u.rad).to(u.arcsec) # phase centre units in rad

# This code will calculate the offsets as seen when executing uvmodelfit

xoff_rad = (fit_RA_rad - pc_RA_rad)*np.cos(pc_DEC_rad)
xoff_arcsec=(fit_RA_arcsec - pc_RA_arcsec)

yoff_rad = (fit_DEC_rad - pc_DEC_rad)
yoff_arcsec=(fit_DEC_arcsec - pc_DEC_arcsec)

print('cl_file:',
      cl_file)
print()

print('fit_RA_rad:',
       fit_RA_rad)
print('pc_RA_rad:',
      pc_RA_rad)
print('x offset in radians:',
      xoff_rad)
print()

print('fit_RA_arcsec:',
       fit_RA_arcsec)
print('pc_RA_arcsec:',
      pc_RA_arcsec)
print('x offset in arcsec:',
      xoff_arcsec)
print()

print('fit_DEC_rad:',
       fit_DEC_rad)
print('pc_DEC_rad:',
      pc_DEC_rad)
print('y offset in radians:',
      yoff_rad)
print()

print('fit_DEC_arcsec:',
       fit_DEC_arcsec)
print('pc_DEC_arcsec:',
      pc_DEC_arcsec)
print('y offset in arcsec:',
      yoff_arcsec)
