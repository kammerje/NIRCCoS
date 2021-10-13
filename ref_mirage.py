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

from mirage.reference_files import downloader

import util


# =============================================================================
# REF MIRAGE
# =============================================================================

# Read config.
config = util.config()

# Download MIRAGE reference files.
download_path = config.paths['mirage_refs_dir']
if (not os.path.exists(download_path)):
    os.makedirs(download_path)
downloader.download_reffiles(download_path,
                             instrument='nircam',
                             dark_type='linearized',
                             skip_darks=False,
                             single_dark=True,
                             skip_cosmic_rays=False,
                             skip_psfs=False)

# Fix issue with folder name occuring with old version of MIRAGE.
try:
    os.rename(download_path+'mirage_data/nircam/darks/linearized/ALONG/', download_path+'mirage_data/nircam/darks/linearized/A5/')
except:
    pass
try:
    os.rename(download_path+'mirage_data/nircam/darks/linearized/BLONG/', download_path+'mirage_data/nircam/darks/linearized/B5/')
except:
    pass

print('DONE')
