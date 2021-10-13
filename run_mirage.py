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
import yaml

import util

from mirage import imaging_simulator
from mirage.catalogs import catalog_generator
from mirage.yaml import yaml_generator


# =============================================================================
# PARAMETERS
# =============================================================================

# Read config.
config = util.config()

# Append MIRAGE and CRDS paths.
os.environ['MIRAGE_DATA'] = config.paths['mirage_refs_dir']+'mirage_data/'
os.environ['CRDS_PATH'] = 'crds_cache'
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

# APT files.
xml_path = config.paths['wdir']+config.paths['xml_path']
pointing_path = config.paths['wdir']+config.paths['pointing_path']

# Observation date and roll angles.
dates = config.obs['date'][:10]
roll_angle = {}
for i in range(len(config.obs['num'])):
    nums = config.obs['num'][i]
    roll_angle['%03.0f' % nums[0]] = config.obs['pa'][i][0] # roll 1
    roll_angle['%03.0f' % nums[1]] = config.obs['pa'][i][1] # roll 2
    roll_angle['%03.0f' % nums[2]] = config.obs['pa'][i][2] # ref

# Find used filters.
filters = []
for i in range(len(config.obs['num'])):
    filters += config.obs['filter'][i][0]
filters += ['F335M']
filters = np.unique(filters)

# Set output directories.
cdir = config.paths['wdir']+config.paths['mirage_cats_dir']
if (not os.path.exists(cdir)):
    os.makedirs(cdir)
ldir = config.paths['wdir']+config.paths['mirage_logs_dir']
if (not os.path.exists(ldir)):
    os.makedirs(ldir)
ddir = config.paths['wdir']+config.paths['mirage_data_dir']
if (not os.path.exists(ddir)):
    os.makedirs(ddir)


# =============================================================================
# RUN MIRAGE
# =============================================================================

# Create source catalogs of random source. The MIRAGE ramp images will be
# replaced by pyNRC ramp images anyways.
ra = ['00h00m00s']
dec = ['+00d00m00s']
ptsrc = catalog_generator.PointSourceCatalog(ra=ra,
                                             dec=dec,
                                             starting_index=1)
for i in range(len(filters)):
    ptsrc.add_magnitude_column([20.], instrument='nircam', filter_name=filters[i], magnitude_system='vegamag')
ptsrc.save(cdir+'sci_ptsrc.cat')
ra = ['00h00m00s']
dec = ['+00d00m00s']
ptsrc = catalog_generator.PointSourceCatalog(ra=ra,
                                             dec=dec,
                                             starting_index=1)
for i in range(len(filters)):
    ptsrc.add_magnitude_column([20.], instrument='nircam', filter_name=filters[i], magnitude_system='vegamag')
ptsrc.save(cdir+'ref_ptsrc.cat')
catalogs = {config.src[0]['name']: {'point_source': cdir+'sci_ptsrc.cat'},
            config.src[1]['name']: {'point_source': cdir+'ref_ptsrc.cat'}}

# Create YAML files.
temp = yaml_generator.SimInput(xml_path,
                               pointing_path,
                               catalogs=catalogs,
                               verbose=False,
                               output_dir=ldir,
                               simdata_output_dir=ddir,
                               cosmic_rays=None,
                               background=None,
                               roll_angle=roll_angle,
                               dates=dates,
                               datatype='raw',
                               dateobs_for_background=True,
                               reffile_defaults='crds',
                               reffile_overrides=None,
                               segmap_flux_limit=None, # for spectroscopic data only
                               segmap_flux_limit_units=None, # for spectroscopic data only
                               add_ghosts=False, # for NIRISS only
                               convolve_ghosts_with_psf=False) # for NIRISS only
temp.use_linearized_darks = True
temp.create_inputs()

# Run MIRAGE.
yaml_files = [ldir+f for f in os.listdir(ldir) if f.endswith('.yaml')]
yaml_files.remove(ldir+'observation_list.yaml')
nums = []
inds = []
for yaml_file in yaml_files:
    
    # MIRAGE does not support coronagraphic observations, therefore change the
    # observing mode to imaging and fix unknown dither patterns.
    with open(yaml_file, 'r') as file:
        temp = yaml.safe_load(file)
    
    temp['Inst']['mode'] = 'imaging'
    temp['Readout']['pupil'] = 'CLEAR'
    if (temp['Output']['total_primary_dither_positions'] == 'NONE'):
        temp['Output']['total_primary_dither_positions'] = 1
    
    num = int(yaml_file[-29:-26])
    ind = int(yaml_file[-16:-11])
    for j in range(len(config.obs['num'])):
        if (num in config.obs['num'][j]):
            break
    if (j >= len(config.obs['num'])):
        raise UserWarning('YAML file '+yaml_file+' cannot be matched to an observation in the APT file')
    ww = np.where(num == np.array(config.obs['num'][j]))[0][0]
    
    # Skip Target Acquisition and Astrometric Confirmation images.
    if (config.obs['conf'][j][ww] == False):
        skip = [1]
    else:
        skip = [1, 2, 3]
    if (ind not in skip):
        filter = temp['Readout']['filter']
        vv = np.where(filter == np.array(config.obs['filter'][j][ww]))[0][0]
        if ('MASKSWB' in config.obs['apername'][j][ww][vv]):
            uu = config.obs['apername'][j][ww][vv].find('MASKSWB')
            temp['Readout']['array_name'] = config.obs['apername'][j][ww][vv][:uu+7]
        elif ('MASKLWB' in config.obs['apername'][j][ww][vv]):
            uu = config.obs['apername'][j][ww][vv].find('MASKLWB')
            temp['Readout']['array_name'] = config.obs['apername'][j][ww][vv][:uu+7]
        else:
            temp['Readout']['array_name'] = config.obs['apername'][j][ww][vv]
    
    with open(yaml_file, 'w') as file:
        yaml.safe_dump(temp, file, default_flow_style=False)
    
    # Simulate the images.
    im = imaging_simulator.ImgSim()
    im.paramfile = yaml_file
    im.create()

print('DONE')
