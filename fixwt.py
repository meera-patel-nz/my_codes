import os, sys
import numpy as np
from datetime import date

top_dir = '/Volumes/disks/meerap/data/' #this is where the data lives
targ_name = 'FPTau'
tag = 'B4_hi'
input_ms_name = 'FPTau_B4_hi.contap1.ms' # name of measurement set to be copied

data_dir = top_dir + targ_name + '/' # eg. this would give /Volumes/disks/meerap/data/FPTau/
source_ms = data_dir + targ_name + '_' + tag + '.contap1.ms' # eg. this would give /Volumes/disks/meerap/data/FPTau/FPTau_B4_hi.contap1.ms
out_fol = data_dir + 'statwt_' + str(date.today()) + '/' # output folder where copied .ms will go, eg. /Volumes/disks/meerap/data/FPTau/statwt_2026-06-08/
mscpy = out_fol + targ_name + '_' + tag + '.contap1' + '_wtfx.ms'

print('Original MS:')
print(source_ms)

print('Copied/statwt MS:')
print(mscpy)

if not os.path.exists(source_ms):
    raise RuntimeError('Source MS does not exist.')

if os.path.exists(mscpy):
    raise RuntimeError('Copied MS already exists.')

os.system('mkdir -p ' + out_fol)
os.system('cp -r ' + source_ms + ' ' + mscpy)

statwt(vis=mscpy) #run statwt on the copied .ms

print('Complete! statwt was run on: ')
print(mscpy)
