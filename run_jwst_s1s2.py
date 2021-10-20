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

from astropy import units as u
from astropy.coordinates import SkyCoord

from jwst.pipeline import Detector1Pipeline, Image2Pipeline


# =============================================================================
# PARAMETERS
# =============================================================================

# Read config.
config = util.config()

# MIRAGE and JWST data directories.
mdir = config.paths['wdir']+config.paths['mirage_data_dir']
odir = config.paths['wdir']+config.paths['jwst_s1s2_data_dir']
if (not os.path.exists(odir)):
    os.makedirs(odir)


# =============================================================================
# RUN JWST
# =============================================================================

# Get MIRAGE files.
mfiles = [f for f in os.listdir(mdir) if f.endswith('_uncal.fits')]
mfiles = sorted(mfiles)

# Go through all MIRAGE files.
for i in range(len(mfiles)):
    print('Reducing '+mfiles[i])
    num = int(mfiles[i][7:10])
    ind = int(mfiles[i][20:25])
    for j in range(len(config.obs['num'])):
        if (num in config.obs['num'][j]):
            break
    if (j >= len(config.obs['num'])):
        raise UserWarning('MIRAGE file '+mfiles[i]+' cannot be matched to an observation in the APT file')
    ww = np.where(num == np.array(config.obs['num'][j]))[0][0]
    
    # Skip Target Acquisition and Astrometric Confirmation images.
    if (config.obs['conf'][j][ww] == False):
        skip = [1]
    else:
        skip = [1, 2, 3]
    if (ind not in skip):
        print('   Running through JWST data reduction pipeline')
        
        # Initialize Detector1Pipeline.
        result1 = Detector1Pipeline()
        
        # 1 Data quality initialization.
        # - SCI remains unchanged.
        # - ERR is added (empty).
        # - PIXELDQ is added.
        # - NOTE: PIXELDQ has shape (sz, sz) and contains bad pixels that
        #         affect each frame (e.g., dead pixels).
        # - GROUPDQ is added (empty).
        # - NOTE: GROUPDQ has shape (nint, ngroup, sz, sz).
        result1.dq_init.skip = False
        result1.dq_init.save_results = False
        
        # 2 Saturation check.
        # - SCI remains unchanged.
        # - ERR remains unchanged.
        # - PIXELDQ flags pixels with value 0 in CRDS saturation reference
        #   file.
        # - GROUPDQ flags saturated, zero, and negative pixels.
        result1.saturation.skip = False
        result1.saturation.save_results = False
        
        # 3 Superbias subtraction.
        # - SCI gets superbias subtracted.
        # - NOTE: this step does more than superbias subtraction.
        # - ERR remains unchanged.
        # - PIXELDQ combines science and superbias PIXELDQs.
        # - GROUPDQ remains unchanged.
        result1.superbias.skip = False
        result1.superbias.save_results = False
        
        # 4 Reference pixel correction.
        # - DOES NOTHING.
        result1.refpix.skip = False
        result1.refpix.save_results = False
        result1.refpix.odd_even_columns = True
        result1.refpix.use_side_ref_pixels = True
        result1.refpix.side_smoothing_length = 11
        result1.refpix.side_gain = 1.
        result1.refpix.odd_even_rows = True
        
        # 5 Linearity correction.
        # - SCI gets linearity corrected.
        # - NOTE: this step can create bad pixels with negative or highly
        #         positive values.
        # - ERR remains unchanged.
        # - PIXELDQ combines science and linearity PIXELDQs.
        # - GROUPDQ remains unchanged.
        result1.linearity.skip = False
        result1.linearity.save_results = False
        
        # 6 Persistence correction.
        # - DOES NOTHING.
        result1.persistence.skip = False
        result1.persistence.save_results = False
        result1.persistence.input_trapsfilled = None
        result1.persistence.flag_pers_cutoff = 40.
        result1.persistence.save_persistence = None
        
        # 7 Dark subtraction.
        # - SCI gets dark subtracted.
        # - NOTE: this step should be skipped for now because the available
        #         dark reference files are bad.
        # - ERR remains unchanged.
        # - PIXELDQ remains unchanged.
        # - GROUPDQ remains unchanged.
        result1.dark_current.skip = True
        result1.dark_current.save_results = False
        result1.dark_current.dark_output = None
        
        # 8 Jump detection.
        # - DOES NOTHING.
        result1.jump.skip = False
        result1.jump.save_results = False
        # result1.jump.rejection_threshold = 4.
        result1.jump.rejection_threshold = 200.
        result1.jump.maximum_cores = 'none'
        result1.jump.flag_4_neighbors = True
        # result1.jump.max_jump_to_flag_neighbors = 200.
        # result1.jump.min_jump_to_flag_neighbors = 10.
        
        # 9 Slope fitting.
        # - SCI gets collapsed along ngroup axis.
        # - ERR gets collapsed along ngroup axis.
        # - NOTE: ERR is not sqrt(VAR_POISSON+VAR_RNOISE) in rateints data
        #         product.
        # - VAR_POISSON is added.
        # - NOTE: VAR_POISSON contains photon noise.
        # - VAR_RNOISE is added.
        # - NOTE: VAR_RNOISE contains read noise.
        # - DQ is added.
        # - NOTE: DQ combines PIXELDQ and GROUPDQ bad pixels.
        result1.ramp_fit.skip = False
        result1.save_results = True
        result1.ramp_fit.save_opt = False
        result1.ramp_fit.opt_name = None
        result1.ramp_fit.int_name = None
        result1.ramp_fit.maximum_cores = 'none'
        
        # Run Detector1Pipeline.
        result1.output_dir = odir
        result1.run(mdir+mfiles[i])
        
        # Initialize Image2Pipeline.
        result2 = Image2Pipeline()
        
        # 1 WCS information.
        # - DOES NOTHING.
        result2.assign_wcs.skip = False
        result2.assign_wcs.save_results = False
        
        # 2 Background subtraction.
        # - DOES NOTHING.
        result2.bkg_subtract.skip = False
        result2.bkg_subtract.save_results = False
        
        # 3 Flat field correction.
        # - SCI gets flat field corrected.
        # - ERR gets flat field corrected.
        # - VAR_POISSON gets flat field corrected.
        # - VAR_RNOISE gets flat field corrected.
        # - DQ combines science and flat field DQs.
        result2.flat_field.skip = False
        result2.flat_field.save_results = False
        result2.flat_field.save_interpolated_flat = False
        
        # 4 Flux calibration.
        # - SCI gets flux calibrated.
        # - ERR gets flux calibrated.
        # - VAR_POISSON gets flux calibrated.
        # - VAR_RNOISE gets flux calibrated.
        # - DQ remains unchanged.
        result2.photom.skip = False
        result2.photom.save_results = False
        
        # Rectified 2D product.
        # - DOES NOTHING.
        result2.resample.skip = False
        result2.save_results = True
        result2.resample.pixfrac = 1.
        result2.resample.kernel = 'square'
        result2.resample.pixel_scale_ratio = 1.
        result2.resample.fillval = 'INDEF'
        result2.resample.weight_type = 'exptime'
        result2.resample.single = False
        result2.resample.blendheaders = True
        result2.resample.allowed_memory = None
        
        # Run Image2Pipeline.
        result2.output_dir = odir
        result2.run(odir+mfiles[i].replace('uncal', 'rateints'))
    
    else:
        print('   Skipping because Target Acquisition or Astrometric Confirmation image')

print('DONE')
