from __future__ import division

import matplotlib
matplotlib.rcParams.update({'font.size': 14})


# =============================================================================
# IMPORTS
# =============================================================================

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import os

from astropy import units as u
from astropy.coordinates import SkyCoord

from mirage.utils.siaf_interface import sci_subarray_corners

import util


# =============================================================================
# PARAMETERS
# =============================================================================

# Read config.
config = util.config()

# MIRAGE and pyNRC data directories.
mdir = config.paths['wdir']+config.paths['mirage_data_dir']
pdir = config.paths['wdir']+config.paths['pynrc_data_dir']


# =============================================================================
# MAIN
# =============================================================================

# Get MIRAGE and pyNRC files.
mfiles = [f for f in os.listdir(mdir) if f.endswith('_uncal.fits')]
mfiles = sorted(mfiles)
pfiles = [f for f in os.listdir(pdir) if f.endswith('.fits')]
pfiles = sorted(pfiles)

# Go through all MIRAGE files.
for i in range(len(mfiles)):
    print('Modifying '+mfiles[i])
    num = int(mfiles[i][7:10])
    ind = int(mfiles[i][20:25])
    flag = False
    for j in range(len(config.obs['num'])):
        if (num in config.obs['num'][j]):
            flag = True
            break
    if (flag == False):
        raise UserWarning('MIRAGE file '+mfiles[i]+' cannot be matched to an observation in the APT file')
    ww = np.where(num == np.array(config.obs['num'][j]))[0][0]
    
    # Skip Target Acquisition and Astrometric Confirmation images.
    if (config.obs['conf'][j][ww] == False):
        skip = [1]
    else:
        skip = [1, 2, 3]
    if (ind not in skip):
        hdul = pyfits.open(mdir+mfiles[i], do_not_scale_image_data=True)
        filter = hdul[0].header['FILTER']
        vv = np.where(filter == np.array(config.obs['filter'][j][ww]))[0][0]
        
        # Fix incompatibility between MIRAGE v2.1.0 and JWST v1.2.3.
        try:
            if (isinstance(hdul[0].header['NDITHPTS'], str)):
                hdul[0].header['NRIMDTPT'] = int(hdul[0].header['NDITHPTS'])
        except:
            pass
        
        # Find dither pattern.
        if (config.obs['patttype'][j][ww] == 'NONE'):
            pfile = 'obs_%03.0f_filt_' % num+filter+'_ramp.fits'
        elif (config.obs['patttype'][j][ww] == '5-POINT-BOX'):
            dithers = [(0., 0.), (15., 15.), (-15., 15.), (-15., -15.), (15., -15.)] # mas
            temp = [((hdul[0].header['XOFFSET']*1000. == dithers[k][0]) and (hdul[0].header['YOFFSET']*1000. == dithers[k][1])) for k in range(len(dithers))]
            dpos = np.where(np.array(temp) == True)[0][0]
            pfile = 'obs_%03.0f_filt_' % num+filter+'_dpos_%03.0f' % dpos+'_ramp.fits'
        elif (config.obs['patttype'][j][ww] == '5-POINT-DIAMOND'):
            dithers = [(0., 0.), (0., 20.), (0., -20.), (20., 0.), (-20., 0.)] # mas
            temp = [((hdul[0].header['XOFFSET']*1000. == dithers[k][0]) and (hdul[0].header['YOFFSET']*1000. == dithers[k][1])) for k in range(len(dithers))]
            dpos = np.where(np.array(temp) == True)[0][0]
            pfile = 'obs_%03.0f_filt_' % num+filter+'_dpos_%03.0f' % dpos+'_ramp.fits'
        elif (config.obs['patttype'][j][ww] == '9-POINT-CIRCLE'):
            dithers = [(0., 0.), (0., 20.), (-15., 15.), (-20., 0.), (-15., -15.), (0., -20.), (15., -15.), (20., 0.), (15., 15.)] # mas
            temp = [((hdul[0].header['XOFFSET']*1000. == dithers[k][0]) and (hdul[0].header['YOFFSET']*1000. == dithers[k][1])) for k in range(len(dithers))]
            dpos = np.where(np.array(temp) == True)[0][0]
            pfile = 'obs_%03.0f_filt_' % num+filter+'_dpos_%03.0f' % dpos+'_ramp.fits'
        elif (config.obs['patttype'][j][ww] == '3-POINT-BAR'):
            dithers = [(0., 0.), (0., 15.), (0., -15.)] # mas
            temp = [((hdul[0].header['XOFFSET']*1000. == dithers[k][0]) and (hdul[0].header['YOFFSET']*1000. == dithers[k][1])) for k in range(len(dithers))]
            dpos = np.where(np.array(temp) == True)[0][0]
            pfile = 'obs_%03.0f_filt_' % num+filter+'_dpos_%03.0f' % dpos+'_ramp.fits'
        elif (config.obs['patttype'][j][ww] == '5-POINT-BAR'):
            dithers = [(0., 0.), (0., 20.), (0., 10.), (0., -10.), (0., -20.)] # mas
            temp = [((hdul[0].header['XOFFSET']*1000. == dithers[k][0]) and (hdul[0].header['YOFFSET']*1000. == dithers[k][1])) for k in range(len(dithers))]
            dpos = np.where(np.array(temp) == True)[0][0]
            pfile = 'obs_%03.0f_filt_' % num+filter+'_dpos_%03.0f' % dpos+'_ramp.fits'
        else:
            raise UserWarning('Does not support dither pattern '+config.obs['patttype'][j][ww])
        
        # Replace MIRAGE SCI and ZEROFRAME with pyNRC ones.
        print('   Replacing with '+pfile)
        hdul['SCI'].data = pyfits.getdata(pdir+pfile, 'SCI').astype('uint16')
        hdul['ZEROFRAME'].data = pyfits.getdata(pdir+pfile, 'ZEROFRAME').astype('uint16')
        
        # Fix coronagraphy header keywords.
        isref = False
        for k in range(len(config.obs['num'])):
            temp = np.where(np.array(config.obs['tag'][k]) == 'ref')[0].tolist()
            if (num in np.array(config.obs['num'][0])[temp]):
                isref = True
        if (isref == False):
            coord = SkyCoord(config.src[0]['icrs'], unit=(u.hourangle, u.deg))
        else:
            coord = SkyCoord(config.src[1]['icrs'], unit=(u.hourangle, u.deg))
        hdul[0].header['DETECTOR'] = config.obs['detector'][j][ww]
        hdul[0].header['PUPIL'] = config.obs['pupil'][j][ww]
        hdul[0].header['EXP_TYPE'] = 'NRC_CORON'
        hdul[0].header['SUBARRAY'] = config.obs['subarray'][j][ww]
        hdul[0].header['NOUTPUTS'] = config.obs['noutputs'][j][ww]
        hdul[0].header['CORONMSK'] = config.obs['coronmsk'][j][ww]
        hdul[0].header['APERNAME'] = config.obs['apername'][j][ww][vv]
        hdul['SCI'].header['CRPIX1'] = config.obs['crpix'][j][ww][vv][0]
        hdul['SCI'].header['CRPIX2'] = config.obs['crpix'][j][ww][vv][1]
        hdul['SCI'].header['CRVAL1'] = coord.ra.deg
        hdul['SCI'].header['CRVAL2'] = coord.dec.deg
        hdul['SCI'].header['XREF_SCI'] = config.obs['crpix'][j][ww][vv][0]
        hdul['SCI'].header['YREF_SCI'] = config.obs['crpix'][j][ww][vv][1]
        hdul['SCI'].header['RA_REF'] = coord.ra.deg
        hdul['SCI'].header['DEC_REF'] = coord.dec.deg
        
        # Correct mapping to CRDS files, probably a pyNRC error.
        if (hdul[0].header['SUBARRAY'] == 'SUB320A430R'):
            x, y = sci_subarray_corners('nircam', 'NRCA5_MASK430R')
            substrt1, substrt2 = x[0]+1, y[0]+1
            hdul[0].header['SUBSTRT1'] = substrt1+1
            hdul[0].header['SUBSTRT2'] = substrt2+11
        elif (hdul[0].header['SUBARRAY'] == 'SUB320ALWB'):
            x, y = sci_subarray_corners('nircam', 'NRCA5_MASKLWB')
            substrt1, substrt2 = x[0]+1, y[0]+1
            hdul[0].header['SUBSTRT1'] = substrt1-1
            hdul[0].header['SUBSTRT2'] = substrt2+15
        else:
            raise UserWarning('Mapping correction unknown')
        
        hdul.writeto(mdir+mfiles[i], overwrite=True)
        hdul.close()
    else:
        print('   Skipping because Target Acquisition or Astrometric Confirmation image')

print('DONE')
