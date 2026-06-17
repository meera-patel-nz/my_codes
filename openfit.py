#import library
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

fits_path = '/Volumes/disks/meerap/data/modelTau/DOTau_B4_hi.fits'

HDUlist = fits.open(fits_path) # open fits file
HDUlist.info() # print structure of file
HDU=HDUlist[0] # access primary HDU

header = HDU.header
data = HDU.data # get data

print(type(data)) # print data

image=np.squeeze(data)

print('Maximum pixel brightness:', np.nanmax(data))
print('Minimum pixel brightness:', np.nanmin(data))

plt.hist(data.ravel(), bins=300)
plt.show()

c_y = image.shape[0] // 2
c_x = image.shape[1] // 2

half = 500

cutout = image[
    max(c_y-half, 0):min(c_y+half, image.shape[0]),
    max(c_x-half, 0):min(c_x+half, image.shape[1])
]
vmin = np.percentile(image, 5) # vmin (and vmax) from data distribution
vmax = np.percentile(image, 99)

plt.imshow(cutout, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
plt.colorbar()
plt.show()

keys_to_print = [
    "OBJECT", # what source this is
    "TELESCOP", # telescope
    "INSTRUME",
    "DATE-OBS",
    "BUNIT", # units of the image
    "BMAJ", # beam size
    "BMIN",
    "BPA", # beam angle
    "NAXIS",# number of axes
    "NAXIS1",#image size in pixels
    "NAXIS2", # image size in pixels
    "NAXIS3",
    "NAXIS4",
    "CTYPE1",# sky coordinate axes
    "CTYPE2",
    "CTYPE3",
    "CTYPE4",
    "CRVAL1",# sky coordinate of reference pixel
    "CRVAL2",
    "CRVAL3",
    "CDELT1",# angular pixel size
    "CDELT2",
    "CUNIT1",
    "CUNIT2",
    "RESTFRQ",
    "SPECSYS",
]

print(f"{'KEY':<42} {'VALUE':<67} COMMENT")
print("-" * 70)

for key in keys_to_print:
    if key in header:
        value = header[key]
        print(f"{key:<42} {str(value):<67}")

HDUlist.close()






