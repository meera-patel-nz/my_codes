from casatasks import tclean, imstat, visstat
import os

vis_no_noise = '/Volumes/disks/meerap/data/DOTau/simulation/DOTau_sim/DOTau_sim.alma.cycle5.1.ms'

vis_noisy = '/Volumes/disks/meerap/data/DOTau/simulation/DOTau_sim/DOTau_sim.alma.cycle5.1.noisy.ms'

imagename_no_noise = '/Volumes/disks/meerap/data/DOTau/simulation/DOTau_sim/DOTau_nonoise_clean'

imagename_noisy = '/Volumes/disks/meerap/data/DOTau/simulation/DOTau_sim/DOTau_noisy_clean'

tclean(
    vis= vis_noisy,
    imagename= imagename_noisy,
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
