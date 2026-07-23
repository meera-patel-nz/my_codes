import casatools
from casatools import table
from casatools import msmetadata
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from casatasks import imhead, imstat # idk guys, at a certain point you don't question things anymore.

# Calibraton and reduction tools. A combination of slightly-edited tools from Sean Andrews, and my own methods.

def has_corrected_column(ms_path):
    tb = table()
    try:
        # Open the main table of the measurement set
        tb.open(ms_path)

        # Get all column names
        colnames = tb.colnames()

        # Check if 'CORRECTED_DATA' is in the list
        return 'CORRECTED_DATA' in colnames

    except Exception as e:
        print(f"Error accessing MS table: {e}")
        return False

    finally:
        # Always close the table tool
        tb.close()

# tool for getting optimal image parameters
def imparamcalc(inpms):
    su = casatools.synthesisutils()

    numSpws = []
    tb = casatools.table()
    tb.open(inpms + '/SPECTRAL_WINDOW') # This is in the .ms file
    maxfreq = max(tb.getcol('REF_FREQUENCY'))
    minfreq = min(tb.getcol('REF_FREQUENCY'))
    numSpws.append(len(tb.getcol('REF_FREQUENCY')))
    tb.close()

    ms = casatools.ms()
    ms.open(inpms)
    uv_range = ms.range(["uvdist"])
    maxuv = (uv_range["uvdist"][1])
    ms.close()

    c = 2.997925e8
    wave = c / maxfreq
    cellsize = 206265. * wave / maxuv / 5.0
    mycell = str(cellsize) + 'arcsec'
    #print("Cell size calculated: ",mycell)

    msmd = casatools.msmetadata()
    msmd.open(inpms)
    antlist = msmd.antennadiameter()
    antsize = antlist['0']['value']
    msmd.close()

    for i in range(len(antlist)):
        if antlist[str(i)]['value'] != antlist['0']['value']:
            raise Exception('Antenna diameters do not all match, please look at listobs manually.')
            break
    
    print("\n...Antenna size: ", antsize)

    fwhm = 206265. * c / minfreq / antsize
    myimsize = max(200, su.getOptimumSize(int(fwhm * 2.0 / cellsize)))
    #print("Imsize calculated: ", myimsize)
    return mycell, myimsize

def estimate_SNR(imagename, disk_mask, noise_mask, f):
    # This version does not print to terminal, it instead takes a file f and writes to it directly.

    headerlist = imhead(imagename, mode = 'list')
    bmaj = headerlist['beammajor']['value']
    bmin = headerlist['beamminor']['value']
    bpa = headerlist['beampa']['value']
    f.write("\n")
    f.write("\n# %s" % imagename)
    f.write("\n# Beam %.3f arcsec x %.3f arcsec (%.2f deg)" % (bmaj, bmin, bpa))
    disk_stats = imstat(imagename = imagename, region = disk_mask)
    disk_flux = disk_stats['flux'][0]
    f.write("\n# Flux inside mask: %.1f uJy" % (disk_flux*1e6,))
    peak_intensity = disk_stats['max'][0]
    f.write("\n# Peak intensity of source: %.1f uJy/beam" % (peak_intensity*1e6,))
    rms = imstat(imagename = imagename, region = noise_mask)['rms'][0]
    f.write("\n# rms: %.1f uJy/beam" % (rms*1e6,))
    SNR = peak_intensity/rms
    f.write("\n# Peak SNR: %.1f" % (SNR,))

def export_vis(visname, outname):
    # get the data tables out of the MS file
    tb = table()
    tb.open(visname)
    data = np.squeeze(tb.getcol("DATA"))
    flag = np.squeeze(tb.getcol("FLAG")) # 2, 16, 29400
    uvw = tb.getcol("UVW")
    weight = tb.getcol("WEIGHT")
    spwid = tb.getcol("DATA_DESC_ID") # 29400,
    tb.close()

    # get the frequency information
    tb.open(visname+'/SPECTRAL_WINDOW')
    freqlist = tb.getvarcol("CHAN_FREQ") # elements of CHAN_FREQ are arrays that can vary in length, so need getvarcol
    tb.close()

    # remove lingering flagged columns

    good = np.squeeze(np.any(flag, axis=0) == False) # 16, 29400 (implicitly assuming that both polarizations have the same flags)

    # Throws error if different channels have flags on different data points.
    same_as_0 = np.zeros(good.shape[0])
    for i in range(good.shape[0]):
        if np.all(good[0, :] == good[i, :]):
            same_as_0[i] = True
    if np.any(same_as_0 != True):
        raise Exception('Flagging differs between channels, please observe data manually.')

    good = good[0, :] # 29400 (We are assuming flagging on integration/baseline basis, not channel basis here!)
    data = data[:, :, good] # 2, 16, 29400
    weight = weight[:, good] # 2, 29400
    uvw = uvw[:, good] # 3, 29400 (because u, v, w for each data point)
    spwid = spwid[good] # 29400,

    # copy weight to match dimensions of data
    polarizations = data.shape[0]
    channels = data.shape[1]
    datapoints = data.shape[2]
    big_weight = np.tile(np.reshape(weight, (polarizations, 1, datapoints)), (1, channels, 1)) # Double check that this reshape is right

    # average the polarizations
    Re = np.sum(data.real * big_weight, axis=0) / np.sum(weight, axis=0)
    Im = np.sum(data.imag * big_weight, axis=0) / np.sum(weight, axis=0)
    vis = Re + 1j*Im
    wgt = np.sum(weight, axis=0)

    # fix format of freqlist to be rectangular
    spwid_short = np.unique(spwid)
    freqlist_neat = np.zeros((1 + np.max(spwid_short), channels))
    for i in spwid_short:
        freqlist_neat[i, :] = np.squeeze(freqlist['r' + str(i + 1)]) # number of spw, 16

    # associate each datapoint with a frequency
    get_freq = lambda ispw: freqlist_neat[ispw]
    freqs = get_freq(spwid)

    # (u,v) positions in wavelengths
    u, v = np.tile(uvw[0,:], (channels, 1)) * freqs.T / 2.9979e8, np.tile(uvw[1,:], (channels, 1)) * freqs.T / 2.9979e8

    # Make wgt match the dimensions of everything else
    wgt = np.tile(wgt, (channels, 1))

    # output to a numpy save file
    np.savez(outname, u=np.ravel(u), v=np.ravel(v), nu=np.ravel(freqs), Vis=np.ravel(vis), Wgt=np.ravel(wgt))

# define a function to generate a 1-D visibility "profile"
def deproject_vis(data, bins=np.array([0]), incl=0., PA=0., offx=0., offy=0.):

    """
    Deprojects and azimuthally averages visibilities 

    Parameters
    ==========
    data: Numpy file output from export_vis 
    bins: 1D array of bins (kilolambda)
    incl: Inclination of disk (degrees)
    PA: Position angle of disk (degrees)
    offx: Horizontal offset of disk center from phase center (arcseconds)
    offy: Vertical offset of disk center from phase center (arcseconds) 

    Returns
    =======
    uv distance bins (1D array), visibilities (1D array), errors on averaged visibilities (1D array) 
    """
        
    # convert keywords into relevant units
    inclr = np.radians(incl)
    PAr = 0.5 * np.pi - np.radians(PA)
    offx *= -np.pi / (180 * 3600) # Convert to radians
    offy *= -np.pi / (180 * 3600) # Convert to radians

    # change to a deprojected, rotated coordinate system
    uprime = (data['u'] * np.cos(PAr) + data['v'] * np.sin(PAr))
    vprime = (-data['u'] * np.sin(PAr) + data['v'] * np.cos(PAr)) * np.cos(inclr)
    rhop = np.sqrt(uprime**2 + vprime**2)

    # phase shifts to account for offsets
    shifts = np.exp(-2 * np.pi * 1.0j * (data['u'] * -offx + data['v'] * -offy))
    visp = data['Vis'] * shifts
    realp = visp.real
    imagp = visp.imag
    
    # create a class to return outputs
    class Vis_profile:
        def __init__(self, vis_prof, rho_uv, err_std, err_scat, nperbin):
            self.vis_prof = vis_prof 
            self.rho_uv = rho_uv 
            self.err_std = err_std
            self.err_scat = err_scat
            self.nperbin = nperbin

    # if requested, return a binned (averaged) representation
    wgt = data['Wgt']
    if (bins.size > 1):
        avbins = 1e3 * bins       # scale to lambda units (input in klambda)
        bwid = 0.5 * (avbins[1] - avbins[0])
        bvis = np.zeros_like(avbins, dtype='complex')
        berr_std = np.zeros_like(avbins, dtype='complex')
        berr_scat = np.zeros_like(avbins, dtype='complex')
        n_in_bin = np.zeros_like(avbins, dtype='int')
        for ib in np.arange(len(avbins)):
            inb = np.where((rhop >= avbins[ib] - bwid) & (rhop < avbins[ib] + bwid))
            if (len(inb[0]) >= 5):
                bRe, eRemu = np.average(realp[inb], weights=wgt[inb], returned=True)
                eRese = np.std(realp[inb])
                bIm, eImmu = np.average(imagp[inb], weights=wgt[inb], returned=True)
                eImse = np.std(imagp[inb])
                bvis[ib] = bRe + 1j*bIm
                berr_scat[ib] = eRese + 1j*eImse # Found from std
                berr_std[ib] = 1 / np.sqrt(eRemu) + 1j / np.sqrt(eImmu) # Found from sum of the weights
                n_in_bin[ib] = np.size(bRe)
            else:
                bvis[ib] = 0 + 1j*0
                berr_scat[ib] = 0 + 1j*0
                berr_std[ib] = 0 + 1j*0
                n_in_bin[ib] = 0
        parser = np.where(berr_std.real != 0)
        output = Vis_profile(bvis[parser], avbins[parser], berr_std[parser], 
                             berr_scat[parser], n_in_bin[parser])
        return output
    
    # if not, returned the unbinned representation
    output = Vis_profile(realp + 1j*imagp, rhop, 1 / np.sqrt(wgt), 
                         1 / np.sqrt(wgt), np.zeros_like(rhop))

    return output

def nearest_factor(dividend, divisor):

    # Will find the value nearest the divisor that divides evenly into the dividend. If two values equidistant
    # from the divisor both divide evenly, it will pick the lower value.
    i = 0

    while (divisor + i) < dividend:
        
        if i < divisor:
            if dividend % (divisor - i) == 0:
                return divisor - i
        
        if dividend % (divisor + i) == 0:
            return divisor + i
        i += 1
    
    return dividend

# Shift the RA/DEC coords based on the proper motion of the object
def shift_coords(RA, DEC, mua, mud, observationTime):
    JulianDate = 2451545.0
    observationTime += 2400000.5 # Because ALMA uses midnight, but Julian date uses noon

    timeDifference = ((observationTime - JulianDate) * u.day).to(u.yr)

    orig_coords = SkyCoord(RA + ' ' + DEC, unit = (u.hourangle, u.deg))

    RA_shift = ((mua * u.mas/u.yr) * timeDifference).to(u.degree)
    new_RA = orig_coords.ra.degree + RA_shift.value

    DEC_shift = ((mud * u.mas/u.yr) * timeDifference).to(u.degree)
    new_DEC = orig_coords.dec.degree + DEC_shift.value

    new_coords = SkyCoord(new_RA, new_DEC, unit = 'deg')
    new_coords_str = new_coords.to_string('hmsdms').split()

    return [new_coords_str[0], new_coords_str[1]]