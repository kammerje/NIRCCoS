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
    if (num in scis):
        for j in range(len(sets)):
            if (num in sets[j]['nums']):
                break
        ww = np.where(num == np.array(sets[j]['nums']))[0][0]
        if (sets[j]['astro'][ww] == 'false'):
            skip = [1]
        else:
            skip = [1, 2, 3]
        if (int(mfiles[i][20:25]) not in skip):
            hdul = pyfits.open(mdir+mfiles[i], scale_back=True)
            filt = hdul[0].header['FILTER']
            
            pfile = 'obs_%03.0f_filt_' % num+filt+'.fits'
            print('   Replacing with '+pfile)
            hdul['SCI'].data = pyfits.getdata(pdir+pfile, 'SCI').astype('uint16')
            hdul['ZEROFRAME'].data = pyfits.getdata(pdir+pfile, 'ZEROFRAME').astype('uint16')
            
            ww = np.where(filt == np.array(sets[j]['filts']))[0][0]
            if (sets[j]['masks'][ww] in ['MASK210R', 'MASK335R', 'MASK430R']):
                hdul[0].header['PUPIL'] = 'MASKRND'
                hdul[0].header['EXP_TYPE'] = 'NRC_CORON'
            elif (sets[j]['masks'][ww] in ['MASKSWB', 'MASKLWB']):
                hdul[0].header['PUPIL'] = 'MASKBAR'
                hdul[0].header['EXP_TYPE'] = 'NRC_CORON'
            else:
                raise UserWarning('Does not support pupil mask '+sets[j]['masks'][ww])
            
            hdul.writeto(mdir+mfiles[i], overwrite=True)
            hdul.close()
        else:
            print('   Skipping because Target Acquisition or Astrometric Confirmation image')
    elif (num in refs):
        for j in range(len(sets)):
            if (num in sets[j]['nums']):
                break
        ww = np.where(num == np.array(sets[j]['nums']))[0][0]
        if (sets[j]['astro'][ww] == 'false'):
            skip = [1]
        else:
            skip = [1, 2, 3]
        if (int(mfiles[i][20:25]) not in skip):
            hdul = pyfits.open(mdir+mfiles[i], scale_back=True)
            filt = hdul[0].header['FILTER']
            
            if (sets[j]['dithers'][2] == 'NONE'):
                pfile = 'obs_%03.0f_filt_' % num+filt+'.fits'
            elif (sets[j]['dithers'][2] == '5-POINT-BOX'):
                dithers = [(0., 0.), (15., 15.), (-15., 15.), (-15., -15.), (15., -15.)] # mas
                temp = [((hdul[0].header['XOFFSET']*1000. == dithers[k][0]) and (hdul[0].header['YOFFSET']*1000. == dithers[k][1])) for k in range(len(dithers))]
                dpos = np.where(np.array(temp) == True)[0][0]
                pfile = 'obs_%03.0f_filt_' % num+filt+'_dpos_%03.0f' % dpos+'.fits'
            elif (sets[j]['dithers'][2] == '5-POINT-DIAMOND'):
                dithers = [(0., 0.), (0., 20.), (0., -20.), (20., 0.), (-20., 0.)] # mas
                temp = [((hdul[0].header['XOFFSET']*1000. == dithers[k][0]) and (hdul[0].header['YOFFSET']*1000. == dithers[k][1])) for k in range(len(dithers))]
                dpos = np.where(np.array(temp) == True)[0][0]
                pfile = 'obs_%03.0f_filt_' % num+filt+'_dpos_%03.0f' % dpos+'.fits'
            elif (sets[j]['dithers'][2] == '9-POINT-CIRCLE'):
                dithers = [(0., 0.), (0., 20.), (-15., 15.), (-20., 0.), (-15., -15.), (0., -20.), (15., -15.), (20., 0.), (15., 15.)] # mas
                temp = [((hdul[0].header['XOFFSET']*1000. == dithers[k][0]) and (hdul[0].header['YOFFSET']*1000. == dithers[k][1])) for k in range(len(dithers))]
                dpos = np.where(np.array(temp) == True)[0][0]
                pfile = 'obs_%03.0f_filt_' % num+filt+'_dpos_%03.0f' % dpos+'.fits'
            elif (sets[j]['dithers'][2] == '3-POINT-BAR'):
                dithers = [(0., 0.), (0., 15.), (0., -15.)] # mas
                temp = [((hdul[0].header['XOFFSET']*1000. == dithers[k][0]) and (hdul[0].header['YOFFSET']*1000. == dithers[k][1])) for k in range(len(dithers))]
                dpos = np.where(np.array(temp) == True)[0][0]
                pfile = 'obs_%03.0f_filt_' % num+filt+'_dpos_%03.0f' % dpos+'.fits'
            elif (sets[j]['dithers'][2] == '5-POINT-BAR'):
                dithers = [(0., 0.), (0., 20.), (0., 10.), (0., -10.), (0., -20.)] # mas
                temp = [((hdul[0].header['XOFFSET']*1000. == dithers[k][0]) and (hdul[0].header['YOFFSET']*1000. == dithers[k][1])) for k in range(len(dithers))]
                dpos = np.where(np.array(temp) == True)[0][0]
                pfile = 'obs_%03.0f_filt_' % num+filt+'_dpos_%03.0f' % dpos+'.fits'
            else:
                raise UserWarning('Does not support dither pattern '+sets[j]['dithers'][2]+' for reference')
            print('   Replacing with '+pfile)
            hdul['SCI'].data = pyfits.getdata(pdir+pfile, 'SCI').astype('uint16')
            hdul['ZEROFRAME'].data = pyfits.getdata(pdir+pfile, 'ZEROFRAME').astype('uint16')
            
            ww = np.where(filt == np.array(sets[j]['filts']))[0][0]
            if (sets[j]['masks'][ww] in ['MASK210R', 'MASK335R', 'MASK430R']):
                hdul[0].header['PUPIL'] = 'MASKRND'
                hdul[0].header['EXP_TYPE'] = 'NRC_CORON'
            elif (sets[j]['masks'][ww] in ['MASKSWB', 'MASKLWB']):
                hdul[0].header['PUPIL'] = 'MASKBAR'
                hdul[0].header['EXP_TYPE'] = 'NRC_CORON'
            else:
                raise UserWarning('Does not support pupil mask '+sets[j]['masks'][ww])
            
            hdul.writeto(mdir+mfiles[i], overwrite=True)
            hdul.close()
        else:
            print('   Skipping because Target Acquisition or Astrometric Confirmation image')
    else:
        raise UserWarning(mfiles[i]+' cannot be identified with a sci or ref observation')

print('DONE')
