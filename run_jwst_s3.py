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
os.environ['CRDS_PATH'] = 'crds_cache'
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

import utils

from jwst.pipeline import Coron3Pipeline


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

# JWST data directories.
jdir = config['paths']['wdir']+config['paths']['jwst_s1s2_data_dir']
odir = config['paths']['wdir']+config['paths']['jwst_s3_data_dir']
if (not os.path.exists(odir)):
    os.makedirs(odir)


# =============================================================================
# RUN JWST
# =============================================================================

# Get JWST files.
jfiles = [f for f in os.listdir(jdir) if f.endswith('nrca5_calints.fits')]
jfiles = sorted(jfiles)

# # Go through all JWST files.
# for i in range(len(jfiles)):
#     print('Reducing '+jfiles[i])
#     num = int(jfiles[i][7:10])
#     for j in range(len(sets)):
#         if (num in sets[j]['nums']):
#             break
#     ww = np.where(num == np.array(sets[j]['nums']))[0][0]
    
#     # Skip Target Acquisition and Astrometric Confirmation images.
#     if (sets[j]['astro'][ww] == 'false'):
#         skip = [1]
#     else:
#         skip = [1, 2, 3]
#     if (int(jfiles[i][20:25]) not in skip):
#         hdul = pyfits.open(jdir+jfiles[i])
#         filt = hdul[0].header['FILTER']
#         ww = np.where(filt == np.array(sets[j]['filts']))[0][0]
        
#         print('   Updating header keywords')
        
#         # Fix coronagraphy header keywords.
#         if ((sets[j]['masks'][ww] == 'MASK210R') and (sets[j]['subs'][ww] == 640)):
#             hdul[0].header['SUBARRAY'] = 'SUB640A210R'
#         elif ((sets[j]['masks'][ww] == 'MASK335R') and (sets[j]['subs'][ww] == 320)):
#             hdul[0].header['SUBARRAY'] = 'SUB320A335R'
#         elif ((sets[j]['masks'][ww] == 'MASK430R') and (sets[j]['subs'][ww] == 320)):
#             hdul[0].header['SUBARRAY'] = 'SUB320A430R'
#         elif ((sets[j]['masks'][ww] == 'MASKSWB') and (sets[j]['subs'][ww] == 640)):
#             hdul[0].header['SUBARRAY'] = 'SUB640ASWB'
#         elif ((sets[j]['masks'][ww] == 'MASKLWB') and (sets[j]['subs'][ww] == 320)):
#             hdul[0].header['SUBARRAY'] = 'SUB320ALWB'
#         else:
#             raise UserWarning('Does not support pupil mask '+sets[j]['masks'][ww]+' with '+sets[j]['subs'][ww]+' subarray')
        
#         hdul.writeto(jdir+jfiles[i], output_verify='fix', overwrite=True)
#         hdul.close()
    
#     else:
#         print('   Skipping because Target Acquisition or Astrometric Confirmation image')

# Make ASN files.
afiles = []
program = jfiles[0][2:7]
asn_id = 'c0000'
target = 't000'
for i in range(len(seqs)):
    for j in range(len(sets[i]['filts'])):
        if (sets[i]['pups'][j] == 'CIRCLYOT'):
            mask = 'MASKRND'
        elif (sets[i]['pups'][j] == 'WEDGELYOT'):
            mask = 'MASKBAR'
        else:
            raise UserWarning('Does not support pupil mask '+sets[i]['pups'][j])
        if ((sets[i]['masks'][j] == 'MASK210R') and (sets[i]['subs'][j] == 640)):
            subarray = 'SUB640A210R'
        elif ((sets[i]['masks'][j] == 'MASK335R') and (sets[i]['subs'][j] == 320)):
            subarray = 'SUB320A335R'
        elif ((sets[i]['masks'][j] == 'MASK430R') and (sets[i]['subs'][j] == 320)):
            subarray = 'SUB320A430R'
        elif ((sets[i]['masks'][j] == 'MASKSWB') and (sets[i]['subs'][j] == 640)):
            subarray = 'SUB640ASWB'
        elif ((sets[i]['masks'][j] == 'MASKLWB') and (sets[i]['subs'][j] == 320)):
            subarray = 'SUB320ALWB'
        else:
            raise UserWarning('Does not support pupil mask '+sets[i]['masks'][j]+' with subarray '+str(sets[i]['subs'][j]))
        roll1files = [f for f in os.listdir(jdir) if (f.endswith('nrca5_calints.fits') and 'jw0119400'+str(seqs[i][0])+'001' in f and pyfits.getheader(jdir+f, 0)['FILTER'] == sets[i]['filts'][j])]
        roll2files = [f for f in os.listdir(jdir) if (f.endswith('nrca5_calints.fits') and 'jw0119400'+str(seqs[i][1])+'001' in f and pyfits.getheader(jdir+f, 0)['FILTER'] == sets[i]['filts'][j])]
        reffiles = [f for f in os.listdir(jdir) if (f.endswith('nrca5_calints.fits') and 'jw0119400'+str(seqs[i][2])+'001' in f and pyfits.getheader(jdir+f, 0)['FILTER'] == sets[i]['filts'][j])]
        afiles += ['seq_%03.0f' % (i+1)+'_filt_'+sets[i]['filts'][j]+'.asn']
        f = open(odir+afiles[-1], 'w')
        f.write('{"asn_type": "coron3",\n')
        f.write(' "asn_rule": "candidate_Asn_Coron",\n')
        f.write(' "program": "'+program+'",\n')
        f.write(' "asn_id": "'+asn_id+'",\n')
        f.write(' "target": "'+target+'",\n')
        f.write(' "asn_pool": "jw'+program+'_00000000T000000_pool",\n')
        f.write(' "products": [\n')
        name = 'jw'+program+'-'+asn_id+'_'+target+'_nircam_'+sets[i]['filts'][j]+'-'+mask+'-'+subarray
        f.write('     {"name": "'+name.lower()+'",\n')
        f.write('      "members": [\n')
        for k in range(len(roll1files)):
            f.write('          {"expname": "'+jdir+roll1files[k]+'",\n')
            f.write('           "exptype": "science"\n')
            f.write('          },\n')
        for k in range(len(roll2files)):
            f.write('          {"expname": "'+jdir+roll2files[k]+'",\n')
            f.write('           "exptype": "science"\n')
            f.write('          },\n')
        for k in range(len(reffiles)):
            f.write('          {"expname": "'+jdir+reffiles[k]+'",\n')
            f.write('           "exptype": "psf"\n')
            f.write('          },\n')
        f.write('      ]\n')
        f.write('     }\n')
        f.write(' ]\n')
        f.write('}\n')
        f.close()

# Go through all ASN files.
for i in range(len(afiles)):
    
    # Initialize Coron3Pipeline.
    result3 = Coron3Pipeline()
    
    # 1 Assemble reference PSFs.
    result3.stack_refs.skip = False
    result3.stack_refs.save_results = False
    
    # 2 Outlier detection.
    result3.outlier_detection.skip = True
    result3.outlier_detection.save_results = False
    result3.outlier_detection.weight_type = 'exptime'
    result3.outlier_detection.pixfrac = 1.
    result3.outlier_detection.kernel = 'square'
    result3.outlier_detection.fillval = 'INDEF'
    result3.outlier_detection.nlow = 0
    result3.outlier_detection.nhigh = 0
    result3.outlier_detection.maskpt = 0.7
    result3.outlier_detection.grow = 1
    result3.outlier_detection.snr = '4.0 3.0'
    result3.outlier_detection.scale = '0.5 0.4'
    result3.outlier_detection.backg = 0.
    result3.outlier_detection.save_intermediate_results = False
    result3.outlier_detection.resample_data = True
    result3.outlier_detection.good_bits = '~DO_NOT_USE'
    result3.outlier_detection.scale_detection = False
    result3.outlier_detection.allowed_memory = None
    
    # 3 Align reference PSFs.
    result3.align_refs.skip = False
    result3.align_refs.save_results = False
    result3.align_refs.median_box_length = 3
    # result3.align_refs.bad_bits = '0'
    result3.align_refs.bad_bits = 'HOT, UNRELIABLE_BIAS'
    
    # 4 Reference PSF subtraction.
    result3.klip.skip = False
    result3.klip.save_results = False
    result3.klip.truncate = 50
    
    # 5 Image combination.
    result3.resample.skip = False
    result3.save_results = True
    result3.resample.pixfrac = 1.
    result3.resample.kernel = 'square'
    result3.resample.pixel_scale_ratio = 1.
    result3.resample.fillval = 'INDEF'
    result3.resample.weight_type = 'exptime'
    result3.resample.single = False
    result3.resample.blendheaders = True
    result3.resample.allowed_memory = None
    
    # Run Coron3Pipeline.
    result3.output_dir = odir
    result3.run(odir+afiles[i])

print('DONE')
