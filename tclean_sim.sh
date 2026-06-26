from casatasks import tclean, imstat, visstat
import os

vis_no_noise = '/Volumes/disks/meerap/data/DOTau/simulation/DOTau_sim/DOTau_sim.alma.cycle5.1.ms'

vis_noisy = '/Volumes/disks/meerap/data/DOTau/simulation/DOTau_sim/DOTau_sim.alma.cycle5.1.noisy.ms'

imagename_no_noise = '/Volumes/disks/meerap/data/DOTau/simulation/DOTau_sim/DOTau_nonoise_dirty'

imagename_noisy = '/Volumes/disks/meerap/data/DOTau/simulation/DOTau_sim/DOTau_noisy_dirty'

# Delete previous tclean products using FULL PATHS
os.system('rm -rf ' + imagename_no_noise + '.*')
os.system('rm -rf ' + imagename_noisy + '.*')

tclean(
    vis= vis_no_noise,
    imagename= imagename_no_noise,
    spw='',
    specmode='mfs',
    gridder='standard',
    imsize=[1024, 1024],
    cell='0.2arcsec',
    weighting="briggs",
    robust=2.0,
    niter=0,
    interactive=False
)

     #pixel size?? - use grid from prev. fits or standard?

