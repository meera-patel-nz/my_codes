from casatasks import simobserve
from casatools import componentlist
import os

cl.done()

cl_file = '/Volumes/disks/meerap/data/modelTau/DOTau_gaussian.cl'
print('cl file name:', cl_file)
os.system(f'rm -rf {cl_file}')
cl.close()

if os.path.exists("cl_file"):
    rmtables("cl_file")
    cl.done()

# Delete previous CASA image if it exists
image_file = 'DOTau_gaussian.image'
if os.path.exists(image_file):
    os.system(f'rm -rf {image_file}')
    print(f"Deleted previous CASA image: {image_file}")

# Import the Gaussian FITS
importfits(fitsimage='DOTau_B4_hi_gaussian.fits',
           imagename='DOTau_gaussian.image',
           overwrite=True)

# Create component list
cl = componentlist()

PA = -66.6743 # rotation angle (p in deg)
PA_cor=90-abs(PA)
posang = f'{PA_cor}deg'
majax= 0.393551
minax = 0.393551*0.557551
minaxst = f'{minax}arcsec' # FWHM minor axis × axis ratio string

cl.addcomponent(
    flux=0.0484384,                 # peak flux in Jy
    fluxunit='Jy',
    dir='J2000 10h00m00.0s -30d00m00.0s',  # source center
    majoraxis='0.393551arcsec',     # FWHM major axis
    minoraxis= minaxst,
    positionangle= posang,         # PA after correction
    shape='Gaussian'
)


project_folder = '/Volumes/disks/meerap/data/modelTau/DOTau_sim/'
skymodel_path = '/Volumes/disks/meerap/data/modelTau/DOTau_gaussian.image/'
complist_path = '/Volumes/disks/meerap/data/modelTau/DOTau_gaussian.cl/'
antennalist_path = '/soft/casa-6.6.0-20-py3.8.el7/lib/py/lib/python3.8/site-packages/casadata/__data__/alma/simmos/alma.cycle5.1.cfg'

# use os system to copy /soft/casa-6.6.0-20-py3.8.el7/lib/py/lib/python3.8/site-packages/casadata/__data__/alma/simmos/alma.cycle5.1.cfg into the folder, DOTau_sim (project_folder) and .image and .cl are one directory above (in modelTau).


# Delete old folder if exists - may need to move this up or down - will this delete the project i literally just made??

# Delete old folder if it exists using rm -rf
if os.path.exists(project_folder):
    os.system(f'rm -rf {project_folder}')
    print(f"Deleted previous simulation folder: {project_folder}")

# Recreate the folder
os.makedirs(project_folder, exist_ok=True)
print(f"Created simulation folder: {project_folder}")

#if os.path.exists(project_folder):
#    os.system(f'rm -rf {project_folder}')

#generate measurement set

simobserve(
        project=project_folder,
        skymodel=skymodel_path,
        complist=complist_path,
        inbright='0.0484384Jy',
        antennalist=antennalist_path,
        totaltime='180s',
        obsmode='int',
        mapsize='10arcsec',
        incell='0.05arcsec',
        overwrite=True)

from casatools import image
ia = image()
ia.open('DOTau_gaussian.image')
shape = ia.shape()
print("CASA image shape:", shape)
ia.close()

n_pix = mapsize / incell
print("Pixels along one axis:", n_pix)

print("Major axis:", maj)
print("Minor axis:", minr)


