import numpy as np
from casatasks import imstat



targ_name = 'DOTau'
trial = '004'
project_dir = '/Volumes/disks/meerap/data/' + targ_name + '/simulation/' + targ_name + '_sim_' + trial + '/'
ms_noisy = project_dir +  targ_name + '_sim_' + trial + '.alma.cycle5.1.noisy.ms'
imagename_clean = project_dir + targ_name + '_' + trial + '_noisy_clean'
clean_image = imagename_clean + '.image'

# source box (box='left_pixel,bottom_pixel,right_pixel,top_pixel')

Scentre_x = 512.104
Scentre_y = 508.984

Swidth_x = 33.3366
Sheight_y = 38.3686

Slp= Scentre_x - (Scentre_x / 2) # left pixel
Sbp= Scentre_x - (Sheight_y /2) # bottom pixel
Srp= Scentre_x + (Scentre_x / 2) # right pixel
Stp= Scentre_x + (Sheight_y /2) # top pixel

source_box =f'{Slp},{Sbp},{Srp},{Stp}'
print('Source box: ', source_box)

source_stats = imstat(
        imagename = clean_image,
        box = source_box
)

# noise box (box='left_pixel,bottom_pixel,right_pixel,top_pixel')

Ncentre_x = 512.651
Ncentre_y = 448.121

Nwidth_x = 216.881
Nheight_y = 68.262

Nlp= Ncentre_x - (Ncentre_x / 2) # left pixel
Nbp= Ncentre_x - (Nheight_y /2) # bottom pixel
Nrp= Ncentre_x + (Ncentre_x / 2) # right pixel
Ntp= Ncentre_x + (Nheight_y /2) # top pixel

noise_box =f'{Nlp},{Nbp},{Nrp},{Ntp}'
print('Noise box: ', noise_box)

noise_stats = imstat(
        imagename = clean_image,
        box = noise_box
)

peak_flux = source_stats['max'][0]
rms_flux = noise_stats['rms'][0]
SNR = peak_flux / rms_flux

print("Peak flux:", peak_flux)
print("RMS flux:", rms_flux)
print("SNR:", SNR)
