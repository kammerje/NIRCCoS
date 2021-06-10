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

from jwst.pipeline import Detector1Pipeline, Image2Pipeline


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

# MIRAGE and JWST data directories.
mdir = config['paths']['wdir']+config['paths']['mirage_data_dir']
odir = config['paths']['wdir']+config['paths']['jwst_data_dir']
if (not os.path.exists(odir)):
    os.makedirs(odir)


# =============================================================================
# RUN JWST
# =============================================================================

# Get MIRAGE files.
mfiles = [f for f in os.listdir(mdir) if f.endswith('nrca5_uncal.fits')]
mfiles = sorted(mfiles)

# Go through all MIRAGE files.
for i in range(len(mfiles)):
    print('Reducing '+mfiles[i])
    num = int(mfiles[i][7:10])
    for j in range(len(sets)):
        if (num in sets[j]['nums']):
            break
    ww = np.where(num == np.array(sets[j]['nums']))[0][0]
    if (sets[j]['astro'][ww] == 'False'):
        skip = [1]
    else:
        skip = [1, 2, 3]
    if (int(mfiles[i][20:25]) not in skip):
        
        print('   Running through JWST data reduction pipeline')
        
        result1 = Detector1Pipeline()
        result1.save_results = True
        result1.output_dir = odir
        result1.run(mdir+mfiles[i])
        
        result2 = Image2Pipeline()
        result2.save_results = True
        result2.output_dir = odir
        result2.run(odir+mfiles[i].replace('uncal', 'rateints'))
    
    else:
        print('   Skipping because Target Acquisition or Astrometric Confirmation image')

print('DONE')
