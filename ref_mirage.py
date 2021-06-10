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

from mirage.reference_files import downloader


# =============================================================================
# REF MIRAGE
# =============================================================================

download_path = config['paths']['mirage_refs_dir']
if (not os.path.exists(download_path)):
    os.makedirs(download_path)
downloader.download_reffiles(download_path,
                             instrument='nircam',
                             dark_type='linearized',
                             skip_darks=False,
                             single_dark=True,
                             skip_cosmic_rays=False,
                             skip_psfs=False)

os.rename(download_path+'mirage_data/nircam/darks/linearized/ALONG/', download_path+'mirage_data/nircam/darks/linearized/A5/')
os.rename(download_path+'mirage_data/nircam/darks/linearized/BLONG/', download_path+'mirage_data/nircam/darks/linearized/B5/')

print('DONE')
