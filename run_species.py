from __future__ import division

import matplotlib
matplotlib.rcParams.update({'font.size': 14})


# =============================================================================
# IMPORTS
# =============================================================================

import yaml
config = yaml.load(open('config.yaml'), Loader=yaml.BaseLoader)

import sys
sys.path.append(config['paths']['species_dir'])

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import os

import species
import utils


# =============================================================================
# PARAMETERS
# =============================================================================

# Companion tags.
keys = config['companions'].keys()
tags_witp = []
tags_spec = []
for key in keys:
    tags_witp += [config['companions'][key]['name_witp']]
    tags_spec += [config['companions'][key]['name_spec']]

# Directory for companion magnitudes.
pmdir = config['paths']['wdir']+config['paths']['pmdir']
if (not os.path.exists(pmdir)):
    os.makedirs(pmdir)

# Parameters for species.
model = config['pipeline']['model_spec']
teff_range = config['pipeline']['teff_range']
teff_range = teff_range.split(',')
teff_range = (float(teff_range[0]), float(teff_range[1]))
dist = float(config['sources']['sci']['dist'])

# Observing setup.
xml_path = config['paths']['wdir']+config['apt']['xml_path']
keys = config['observation']['sequences'].keys()
seqs = []
sets = []
for key in keys:
    seqs += [config['observation']['sequences'][key].split(',')]
    sets += [utils.read_from_xml(xml_path, seqs[-1])]

# Find used filters.
filts = []
for i in range(len(sets)):
    filts += sets[i]['filts']
filts = np.unique(filts)
filts = ['JWST/NIRCam.'+filts[i] for i in range(len(filts))]


# =============================================================================
# MAIN
# =============================================================================

# Initialize species.
try:
    os.remove('species_database.hdf5')
except:
    pass
species.SpeciesInit()
database = species.Database()

# Add companions and model to database.
for i in range(len(tags_spec)):
    database.add_companion(tags_spec[i])
database.add_model(model=model,
                   wavel_range=(0.8, 5.2),
                   teff_range=teff_range)

# Go through all companions.
for i in range(len(tags_spec)):
    
    # Fit model to companion photometry and spectra.
    fit = species.FitModel(object_name=tags_spec[i],
                           model=model,
                           bounds={'teff': teff_range,
                                   'radius': (0.3, 3.)},
                           inc_phot=True,
                           inc_spec=False)
    fit.run_multinest(tag=tags_witp[i],
                      n_live_points=1000,
                      output='multinest/')
    species.plot_posterior(tag=tags_witp[i],
                           burnin=None,
                           title=None,
                           offset=(-0.25, -0.25),
                           title_fmt='.2f',
                           inc_luminosity=True,
                           inc_mass=True,
                           output=pmdir+'posterior_'+tags_witp[i]+'.pdf')
    model_param = database.get_median_sample(tag=tags_witp[i],
                                             burnin=None)
    
    # Plot median model and synthetic NIRCam photometry.
    readmodel = species.ReadModel(model, wavel_range=(0.8, 5.2))
    modelbox = readmodel.get_model(model_param, spec_res=100., smooth=True)
    objectbox = database.get_object(object_name=tags_spec[i])
    synphotbox = species.multi_photometry(datatype='model',
                                          spectrum=model,
                                          filters=filts,
                                          parameters=model_param)
    res_box = species.get_residuals(datatype='model',
                                    spectrum=model,
                                    parameters=model_param,
                                    objectbox=objectbox,
                                    inc_phot=True,
                                    inc_spec=False)
    species.plot_spectrum(boxes=[modelbox, objectbox, synphotbox],
                          filters=filts,
                          residuals=res_box,
                          plot_kwargs=[{'ls': '-', 'lw': 1., 'color': 'black'},
                                       None,
                                       None],
                          xlim=(0.8, 5.2),
                          scale=('linear', 'linear'),
                          title=tags_spec[i],
                          offset=(-0.45, -0.04),
                          legend={'loc': 'upper right', 'frameon': False, 'fontsize': 12},
                          figsize=(9., 4),
                          quantity='flux',
                          output=pmdir+'spectrum_'+tags_witp[i]+'.pdf')
    
    # Save synthetic NIRCam photometry.
    for j in range(len(filts)):
        synphot = species.SyntheticPhotometry(filts[j])
        app_mag, abs_mag = synphot.flux_to_magnitude(synphotbox.flux[filts[j]], error=0., distance=(dist, 0.))
        temp = np.array([app_mag[0]])
        np.save(pmdir+tags_witp[i]+'_'+filts[j][-5:]+'.npy', temp)

print('DONE')
