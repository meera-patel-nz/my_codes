import os

top_dir = '/Volumes/disks/meerap/data/'
targ_name = 'DSTau'
shape = 'D' # G or D based on Gaussian or Disk 
tag = 'B4_hi'
cf_date = '2026-06-09' # date of the casa_fit

data_dir = top_dir + targ_name + '/'
cl_dir = data_dir + 'casa_fit_' + cf_date + '/' # Folder containing .cl directories
band_dir = data_dir + tag +'/' # band directory where .ms directory lives

os.system('mkdir -p ' + cl_dir)

vis = band_dir + targ_name + '_vis_' + tag + '.ms'
outfile = cl_dir + targ_name + '_complist_' + shape + '.cl'

os.system('rm -rf ' + outfile) # delete the old version of the .cl directory before rerunning uvmodelfit

uvmodelfit(
vis= vis,
field='',
comptype=shape,
sourcepar=[0.00244359,0,0,0.419465,0.53724387016,53.1382],
outfile= outfile
)

