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
os.environ['MIRAGE_DATA'] = config['paths']['mirage_refs_dir']+'mirage_data/'
os.environ['CRDS_PATH'] = 'crds_cache'
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

import utils

from mirage import imaging_simulator
from mirage.catalogs import catalog_generator, create_catalog
from mirage.yaml import yaml_generator


# =============================================================================
# PARAMETERS
# =============================================================================

# APT files.
xml_path = config['paths']['wdir']+config['apt']['xml_path']
pointing_path = config['paths']['wdir']+config['apt']['pointing_path']

# Observing setup.
keys = config['observation']['sequences'].keys()
seqs = []
sets = []
for key in keys:
    seqs += [config['observation']['sequences'][key].split(',')]
    sets += [utils.read_from_xml(xml_path, seqs[-1])]

# Observation date and roll angles.
dates = config['observation']['date'][:10]
PA1 = float(config['observation']['pa1'])
PA2 = float(config['observation']['pa2'])
roll_angle = {}
for i in range(len(sets)):
    nums = sets[i]['nums']
    roll_angle['%03.0f' % nums[0]] = PA1 # roll 1
    roll_angle['%03.0f' % nums[1]] = PA2 # roll 2
    roll_angle['%03.0f' % nums[2]] = PA1 # ref

# Find used filters.
filts = []
for i in range(len(sets)):
    filts += sets[i]['filts']
filts = np.unique(filts)

# Set output directories.
cdir = config['paths']['wdir']+config['paths']['mirage_cats_dir']
if (not os.path.exists(cdir)):
    os.makedirs(cdir)
ldir = config['paths']['wdir']+config['paths']['mirage_logs_dir']
if (not os.path.exists(ldir)):
    os.makedirs(ldir)
ddir = config['paths']['wdir']+config['paths']['mirage_data_dir']
if (not os.path.exists(ddir)):
    os.makedirs(ddir)


# =============================================================================
# RUN MIRAGE
# =============================================================================

# Create source catalogs of random source. The MIRAGE ramp images will be
# replaced by pyNRC ramp images anyway.
ra = ['00h00m00s']
dec = ['+00d00m00s']
ptsrc = catalog_generator.PointSourceCatalog(ra=ra,
                                             dec=dec,
                                             starting_index=1)
for i in range(len(filts)):
    ptsrc.add_magnitude_column([20.], instrument='nircam', filter_name=filts[i], magnitude_system='vegamag')
ptsrc.save(cdir+'sci_ptsrc.cat')
ra = ['00h00m00s']
dec = ['+00d00m00s']
ptsrc = catalog_generator.PointSourceCatalog(ra=ra,
                                             dec=dec,
                                             starting_index=1)
for i in range(len(filts)):
    ptsrc.add_magnitude_column([20.], instrument='nircam', filter_name=filts[i], magnitude_system='vegamag')
ptsrc.save(cdir+'ref_ptsrc.cat')
catalogs = {config['sources']['sci']['name']: {'point_source': cdir+'sci_ptsrc.cat'},
            config['sources']['ref']['name']: {'point_source': cdir+'ref_ptsrc.cat'}}

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
for yaml_file in yaml_files:
    
    # MIRAGE does not support coronagraphic observations, therefore change the
    # observing mode to imaging and fix unknown dither patterns.
    with open(yaml_file, 'r') as file:
        temp = yaml.safe_load(file)
    temp['Inst']['mode'] = 'imaging'
    temp['Readout']['pupil'] = 'CLEAR'
    if (temp['Output']['total_primary_dither_positions'] == 'NONE'):
        temp['Output']['total_primary_dither_positions'] = 1
    with open(yaml_file, 'w') as file:
        yaml.safe_dump(temp, file, default_flow_style=False)
    
    im = imaging_simulator.ImgSim()
    im.paramfile = yaml_file
    im.create()

print('DONE')
