#import library
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u

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

vmin = np.percentile(image, 5) # vmin (and vmax) from data distribution
vmax = np.percentile(image, 99)

plt.imshow(image, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
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

print(f"{'KEY':<34} {'VALUE':<59} COMMENT")
print("-" * 70)

for key in keys_to_print:
    if key in header:
        value = header[key]
        print(f"{key:<34} {str(value):<59}")

HDUlist.close()

