from __future__ import division

import matplotlib
matplotlib.rcParams.update({'font.size': 14})


# =============================================================================
# IMPORTS
# =============================================================================

import yaml
config = yaml.load(open('config.yaml'), Loader=yaml.BaseLoader)

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import os

import utils

from mirage.utils.siaf_interface import sci_subarray_corners


# =============================================================================
# PARAMETERS
# =============================================================================

# Observing setup.
xml_path = config['paths']['wdir']+config['apt']['xml_path']
keys = config['observation']['sequences'].keys()
seqs = []
sets = []
for key in keys:
    seqs += [config['observation']['sequences'][key].split(',')]
    sets += [utils.read_from_xml(xml_path, seqs[-1])]

# Find sci and ref observations.
scis = []
refs = []
for i in range(len(sets)):
    scis += sets[i]['nums'][0:2]
    refs += [sets[i]['nums'][2]]

# MIRAGE and pyNRC data directories.
mdir = config['paths']['wdir']+config['paths']['mirage_data_dir']
pdir = config['paths']['wdir']+config['paths']['pynrc_data_dir']


# =============================================================================
# MAIN
# =============================================================================

# Get MIRAGE and pyNRC files.
mfiles = [f for f in os.listdir(mdir) if f.endswith('nrca5_uncal.fits')]
mfiles = sorted(mfiles)
pfiles = [f for f in os.listdir(pdir) if f.endswith('.fits')]
pfiles = sorted(pfiles)

# Go through all MIRAGE files.
for i in range(len(mfiles)):
    print('Modifying '+mfiles[i])
    num = int(mfiles[i][7:10])
    
    # Science data.
    if (num in scis):
        for j in range(len(sets)):
            if (num in sets[j]['nums']):
                break
        ww = np.where(num == np.array(sets[j]['nums']))[0][0]
        
        # Skip Target Acquisition and Astrometric Confirmation images.
        if (sets[j]['astro'][ww] == 'false'):
            skip = [1]
        else:
            skip = [1, 2, 3]
        if (int(mfiles[i][20:25]) not in skip):
            hdul = pyfits.open(mdir+mfiles[i], do_not_scale_image_data=True)
            filt = hdul[0].header['FILTER']
            
            # Replace MIRAGE SCI and ZEROFRAME with pyNRC ones.
            pfile = 'obs_%03.0f_filt_' % num+filt+'_ramp.fits'
            print('   Replacing with '+pfile)
            hdul['SCI'].data = pyfits.getdata(pdir+pfile, 'SCI').astype('uint16')
            hdul['ZEROFRAME'].data = pyfits.getdata(pdir+pfile, 'ZEROFRAME').astype('uint16')
            
            # Fix incompatibility between MIRAGE v2.1.0 and JWST v1.2.3.
            if (isinstance(hdul[0].header['NDITHPTS'], str)):
                hdul[0].header['NRIMDTPT'] = int(hdul[0].header['NDITHPTS'])
            
            # Fix coronagraphy header keywords.
            ww = np.where(filt == np.array(sets[j]['filts']))[0][0]
            if ((sets[j]['masks'][ww] == 'MASK210R') and (sets[j]['subs'][ww] == 640)):
                hdul[0].header['DETECTOR'] = 'NRCA2'
                hdul[0].header['PUPIL'] = 'MASKRND'
                hdul[0].header['EXP_TYPE'] = 'NRC_CORON'
                hdul[0].header['SUBARRAY'] = 'SUB640A210R'
                hdul[0].header['CORONMSK'] = 'MASKA210R'
            elif ((sets[j]['masks'][ww] == 'MASK335R') and (sets[j]['subs'][ww] == 320)):
                hdul[0].header['DETECTOR'] = 'NRCALONG'
                hdul[0].header['PUPIL'] = 'MASKRND'
                hdul[0].header['EXP_TYPE'] = 'NRC_CORON'
                hdul[0].header['SUBARRAY'] = 'SUB320A335R'
                hdul[0].header['CORONMSK'] = 'MASKA335R'
            elif ((sets[j]['masks'][ww] == 'MASK430R') and (sets[j]['subs'][ww] == 320)):
                hdul[0].header['DETECTOR'] = 'NRCALONG'
                hdul[0].header['PUPIL'] = 'MASKRND'
                hdul[0].header['EXP_TYPE'] = 'NRC_CORON'
                hdul[0].header['SUBARRAY'] = 'SUB320A430R'
                hdul[0].header['CORONMSK'] = 'MASKA430R'
            elif ((sets[j]['masks'][ww] == 'MASKSWB') and (sets[j]['subs'][ww] == 640)):
                hdul[0].header['DETECTOR'] = 'NRCB3'
                hdul[0].header['PUPIL'] = 'MASKBAR'
                hdul[0].header['EXP_TYPE'] = 'NRC_CORON'
                hdul[0].header['SUBARRAY'] = 'SUB640ASWB'
                hdul[0].header['CORONMSK'] = 'MASKASWB'
            elif ((sets[j]['masks'][ww] == 'MASKLWB') and (sets[j]['subs'][ww] == 320)):
                hdul[0].header['DETECTOR'] = 'NRCBLONG'
                hdul[0].header['PUPIL'] = 'MASKBAR'
                hdul[0].header['EXP_TYPE'] = 'NRC_CORON'
                hdul[0].header['SUBARRAY'] = 'SUB320ALWB'
                hdul[0].header['CORONMSK'] = 'MASKALWB'
            elif ((sets[j]['masks'][ww] == 'MASK430R') and (sets[j]['subs'][ww] == 2048)):
                hdul[0].header['DETECTOR'] = 'NRCALONG'
                hdul[0].header['PUPIL'] = 'MASKRND'
                hdul[0].header['EXP_TYPE'] = 'NRC_CORON'
                hdul[0].header['SUBARRAY'] = 'FULL'
                hdul[0].header['CORONMSK'] = 'MASKA430R'
            else:
                raise UserWarning('Does not support pupil mask '+sets[j]['masks'][ww]+' with subarray '+str(sets[j]['subs'][ww]))
            x, y = sci_subarray_corners('nircam', 'NRCA5_'+sets[j]['masks'][ww])
            substrt1, substrt2 = x[0]+1, y[0]+1
            hdul[0].header['SUBSTRT1'] = substrt1
            hdul[0].header['SUBSTRT2'] = substrt2
            # print('   '+str(substrt1), str(substrt2))
            
            hdul.writeto(mdir+mfiles[i], overwrite=True)
            hdul.close()
        
        else:
            print('   Skipping because Target Acquisition or Astrometric Confirmation image')
    
    # Reference data.
    elif (num in refs):
        for j in range(len(sets)):
            if (num in sets[j]['nums']):
                break
        ww = np.where(num == np.array(sets[j]['nums']))[0][0]
        
        # Skip Target Acquisition and Astrometric Confirmation images
        if (sets[j]['astro'][ww] == 'false'):
            skip = [1]
        else:
            skip = [1, 2, 3]
        if (int(mfiles[i][20:25]) not in skip):
            hdul = pyfits.open(mdir+mfiles[i], do_not_scale_image_data=True)
            filt = hdul[0].header['FILTER']
            
            # Fix incompatibility between MIRAGE v2.1.0 and JWST v1.2.3.
            if (isinstance(hdul[0].header['NDITHPTS'], str)):
                hdul[0].header['NRIMDTPT'] = int(hdul[0].header['NDITHPTS'])
            
            # Find dither pattern.
            if (sets[j]['dithers'][2] == 'NONE'):
                pfile = 'obs_%03.0f_filt_' % num+filt+'_ramp.fits'
            elif (sets[j]['dithers'][2] == '5-POINT-BOX'):
                dithers = [(0., 0.), (15., 15.), (-15., 15.), (-15., -15.), (15., -15.)] # mas
                temp = [((hdul[0].header['XOFFSET']*1000. == dithers[k][0]) and (hdul[0].header['YOFFSET']*1000. == dithers[k][1])) for k in range(len(dithers))]
                dpos = np.where(np.array(temp) == True)[0][0]
                pfile = 'obs_%03.0f_filt_' % num+filt+'_dpos_%03.0f' % dpos+'_ramp.fits'
            elif (sets[j]['dithers'][2] == '5-POINT-DIAMOND'):
                dithers = [(0., 0.), (0., 20.), (0., -20.), (20., 0.), (-20., 0.)] # mas
                temp = [((hdul[0].header['XOFFSET']*1000. == dithers[k][0]) and (hdul[0].header['YOFFSET']*1000. == dithers[k][1])) for k in range(len(dithers))]
                dpos = np.where(np.array(temp) == True)[0][0]
                pfile = 'obs_%03.0f_filt_' % num+filt+'_dpos_%03.0f' % dpos+'_ramp.fits'
            elif (sets[j]['dithers'][2] == '9-POINT-CIRCLE'):
                dithers = [(0., 0.), (0., 20.), (-15., 15.), (-20., 0.), (-15., -15.), (0., -20.), (15., -15.), (20., 0.), (15., 15.)] # mas
                temp = [((hdul[0].header['XOFFSET']*1000. == dithers[k][0]) and (hdul[0].header['YOFFSET']*1000. == dithers[k][1])) for k in range(len(dithers))]
                dpos = np.where(np.array(temp) == True)[0][0]
                pfile = 'obs_%03.0f_filt_' % num+filt+'_dpos_%03.0f' % dpos+'_ramp.fits'
            elif (sets[j]['dithers'][2] == '3-POINT-BAR'):
                dithers = [(0., 0.), (0., 15.), (0., -15.)] # mas
                temp = [((hdul[0].header['XOFFSET']*1000. == dithers[k][0]) and (hdul[0].header['YOFFSET']*1000. == dithers[k][1])) for k in range(len(dithers))]
                dpos = np.where(np.array(temp) == True)[0][0]
                pfile = 'obs_%03.0f_filt_' % num+filt+'_dpos_%03.0f' % dpos+'_ramp.fits'
            elif (sets[j]['dithers'][2] == '5-POINT-BAR'):
                dithers = [(0., 0.), (0., 20.), (0., 10.), (0., -10.), (0., -20.)] # mas
                temp = [((hdul[0].header['XOFFSET']*1000. == dithers[k][0]) and (hdul[0].header['YOFFSET']*1000. == dithers[k][1])) for k in range(len(dithers))]
                dpos = np.where(np.array(temp) == True)[0][0]
                pfile = 'obs_%03.0f_filt_' % num+filt+'_dpos_%03.0f' % dpos+'_ramp.fits'
            else:
                raise UserWarning('Does not support dither pattern '+sets[j]['dithers'][2]+' for reference')
            
            # Replace MIRAGE SCI and ZEROFRAME with pyNRC ones.
            print('   Replacing with '+pfile)
            hdul['SCI'].data = pyfits.getdata(pdir+pfile, 'SCI').astype('uint16')
            hdul['ZEROFRAME'].data = pyfits.getdata(pdir+pfile, 'ZEROFRAME').astype('uint16')
            
            # Fix coronagraphy header keywords.
            ww = np.where(filt == np.array(sets[j]['filts']))[0][0]
            if ((sets[j]['masks'][ww] == 'MASK210R') and (sets[j]['subs'][ww] == 640)):
                hdul[0].header['DETECTOR'] = 'NRCA2'
                hdul[0].header['PUPIL'] = 'MASKRND'
                hdul[0].header['EXP_TYPE'] = 'NRC_CORON'
                hdul[0].header['SUBARRAY'] = 'SUB640A210R'
                hdul[0].header['CORONMSK'] = 'MASKA210R'
            elif ((sets[j]['masks'][ww] == 'MASK335R') and (sets[j]['subs'][ww] == 320)):
                hdul[0].header['DETECTOR'] = 'NRCALONG'
                hdul[0].header['PUPIL'] = 'MASKRND'
                hdul[0].header['EXP_TYPE'] = 'NRC_CORON'
                hdul[0].header['SUBARRAY'] = 'SUB320A335R'
                hdul[0].header['CORONMSK'] = 'MASKA335R'
            elif ((sets[j]['masks'][ww] == 'MASK430R') and (sets[j]['subs'][ww] == 320)):
                hdul[0].header['DETECTOR'] = 'NRCALONG'
                hdul[0].header['PUPIL'] = 'MASKRND'
                hdul[0].header['EXP_TYPE'] = 'NRC_CORON'
                hdul[0].header['SUBARRAY'] = 'SUB320A430R'
                hdul[0].header['CORONMSK'] = 'MASKA430R'
            elif ((sets[j]['masks'][ww] == 'MASKSWB') and (sets[j]['subs'][ww] == 640)):
                hdul[0].header['DETECTOR'] = 'NRCB3'
                hdul[0].header['PUPIL'] = 'MASKBAR'
                hdul[0].header['EXP_TYPE'] = 'NRC_CORON'
                hdul[0].header['SUBARRAY'] = 'SUB640ASWB'
                hdul[0].header['CORONMSK'] = 'MASKASWB'
            elif ((sets[j]['masks'][ww] == 'MASKLWB') and (sets[j]['subs'][ww] == 320)):
                hdul[0].header['DETECTOR'] = 'NRCBLONG'
                hdul[0].header['PUPIL'] = 'MASKBAR'
                hdul[0].header['EXP_TYPE'] = 'NRC_CORON'
                hdul[0].header['SUBARRAY'] = 'SUB320ALWB'
                hdul[0].header['CORONMSK'] = 'MASKALWB'
            elif ((sets[j]['masks'][ww] == 'MASK430R') and (sets[j]['subs'][ww] == 2048)):
                hdul[0].header['DETECTOR'] = 'NRCALONG'
                hdul[0].header['PUPIL'] = 'MASKRND'
                hdul[0].header['EXP_TYPE'] = 'NRC_CORON'
                hdul[0].header['SUBARRAY'] = 'FULL'
                hdul[0].header['CORONMSK'] = 'MASKA430R'
            else:
                raise UserWarning('Does not support pupil mask '+sets[j]['masks'][ww]+' with subarray '+str(sets[j]['subs'][ww]))
            x, y = sci_subarray_corners('nircam', 'NRCA5_'+sets[j]['masks'][ww])
            substrt1, substrt2 = x[0]+1, y[0]+1
            hdul[0].header['SUBSTRT1'] = substrt1
            hdul[0].header['SUBSTRT2'] = substrt2
            # print('   '+str(substrt1), str(substrt2))
            
            hdul.writeto(mdir+mfiles[i], overwrite=True)
            hdul.close()
        else:
            print('   Skipping because Target Acquisition or Astrometric Confirmation image')
    else:
        raise UserWarning(mfiles[i]+' cannot be identified with a sci or ref observation')

print('DONE')
