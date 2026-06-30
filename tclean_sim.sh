from casatasks import tclean
from casatools import image as image_tool
import numpy as np
import matplotlib.pyplot as plt
import os

# PATHS KEY

targ_name = 'DOTau'
trial = '004'
project_dir = '/Volumes/disks/meerap/data/' + targ_name + '/simulation/' + targ_name + '_sim_' + trial + '/'
ms_noisy = project_dir +  targ_name + '_sim_' + trial + '.alma.cycle5.1.noisy.ms'
imagename_clean = project_dir + targ_name + '_' + trial + '_noisy_clean'

#tclean

tclean(
    vis= ms_noisy,
    imagename= imagename_clean,
    spw='',
    specmode='mfs',
    gridder='standard',
    imsize=[1024, 1024],
    cell='0.2arcsec',
    weighting="briggs",
    robust=2.0,
    niter=1000,
    interactive=False
)

# view the clean image data

clean_image = imagename_clean + '.image'

rm_command = 'rm -f ' + project_dir + '*.png'

print("Running:", rm_command)
os.system(rm_command)

ia = image_tool()
ia.open(clean_image)
image_data = ia.getchunk()
ia.close()

image_2d = np.squeeze(image_data) # squeeze to 2D
print("Raw squeezed image shape:", image_2d.shape)

image_plot = image_2d.T
ny, nx = image_plot.shape

print("x pixels:", nx)
print("y pixels:", ny)

peak_y, peak_x = np.unravel_index(
    np.nanargmax(image_plot),
    image_plot.shape
)
peak_value = image_plot[peak_y, peak_x]
print("Peak value =", peak_value)

vmin = np.nanpercentile(image_plot, 1) # display scaling?
vmax = np.nanpercentile(image_plot, 99.7)

plt.figure(figsize=(8, 7))

plt.imshow(
    image_plot,
    origin='lower',
    cmap='inferno',
    vmin=vmin,
    vmax=vmax,
    interpolation='nearest'
)

plt.colorbar(label='Flux')
plt.xlabel('x pixel')
plt.ylabel('y pixel')
plt.title('CLEAN image after tclean')

plot_path = imagename_clean + '_diagnostic_full.png'
plt.savefig(plot_path, dpi=200)
plt.show()

print("Full diagnostic plot saved to:")
print(plot_path)



# imstat, defining region for rms


