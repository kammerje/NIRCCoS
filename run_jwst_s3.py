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
os.environ['CRDS_PATH'] = 'crds_cache'
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

import util

from jwst.pipeline import Coron3Pipeline


# =============================================================================
# PARAMETERS
# =============================================================================

# Read config.
config = util.config()

# JWST data directories.
jdir = config.paths['wdir']+config.paths['jwst_s1s2_data_dir']
odir = config.paths['wdir']+config.paths['jwst_s3_data_dir']
if (not os.path.exists(odir)):
    os.makedirs(odir)


# =============================================================================
# RUN JWST
# =============================================================================

# Get JWST files.
jfiles = [f for f in os.listdir(jdir) if f.endswith('_calints.fits')]
jfiles = sorted(jfiles)

# Make ASN files.
afiles = []
program = jfiles[0][2:7]
asn_id = 'c0000'
target = 't000'
for i in range(len(config.obs['filter'])):
    for j in range(len(config.obs['filter'][i][0])):
        roll1files = [f for f in os.listdir(jdir) if (f.endswith('_calints.fits') and '%03.0f' % config.obs['num'][i][0]+'001' in f and pyfits.getheader(jdir+f, 0)['FILTER'] == config.obs['filter'][i][0][j])]
        roll2files = [f for f in os.listdir(jdir) if (f.endswith('_calints.fits') and '%03.0f' % config.obs['num'][i][1]+'001' in f and pyfits.getheader(jdir+f, 0)['FILTER'] == config.obs['filter'][i][0][j])]
        reffiles = [f for f in os.listdir(jdir) if (f.endswith('_calints.fits') and '%03.0f' % config.obs['num'][i][2]+'001' in f and pyfits.getheader(jdir+f, 0)['FILTER'] == config.obs['filter'][i][0][j])]
        afiles += ['seq_%03.0f' % (i+1)+'_filt_'+config.obs['filter'][i][0][j]+'.asn']
        f = open(odir+afiles[-1], 'w')
        f.write('{"asn_type": "coron3",\n')
        f.write(' "asn_rule": "candidate_Asn_Coron",\n')
        f.write(' "program": "'+program+'",\n')
        f.write(' "asn_id": "'+asn_id+'",\n')
        f.write(' "target": "'+target+'",\n')
        f.write(' "asn_pool": "jw'+program+'_00000000T000000_pool",\n')
        f.write(' "products": [\n')
        name = 'jw'+program+'-'+asn_id+'_'+target+'_nircam_'+config.obs['filter'][i][0][j]+'-'+config.obs['pupil'][i][0]+'-'+config.obs['subarray'][i][0]
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
    result3.align_refs.median_box_length = 5
    result3.align_refs.bad_bits = 'DO_NOT_USE'
    
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
