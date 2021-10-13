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
import sys

import util


# =============================================================================
# PARAMETERS
# =============================================================================

# Read config.
config = util.config()

# Append species path.
sys.path.append(config.paths['species_dir'])
import species

# Directory for companion magnitudes.
pmdir = config.paths['wdir']+config.paths['pmdir']
if (not os.path.exists(pmdir)):
    os.makedirs(pmdir)

# Find used filters.
filters = []
for i in range(len(config.obs['filter'])):
    for j in range(len(config.obs['filter'][i])):
        for k in range(len(config.obs['filter'][i][j])):
            filters += [config.obs['filter'][i][j][k]]
filters = np.unique(filters)
filters = ['JWST/NIRCam.'+filters[i] for i in range(len(filters))]


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
for i in range(len(config.cmp)):
    database.add_companion(config.cmp[i]['name_spec'])
database.add_model(model=config.pip['model_spec'],
                   wavel_range=(0.8, 5.2),
                   teff_range=tuple(config.pip['teff_range']))

# Go through all companions.
for i in range(len(config.cmp)):
    
    # Fit model to companion photometry and spectra.
    fit = species.FitModel(object_name=config.cmp[i]['name_spec'],
                           model=config.pip['model_spec'],
                           bounds={'teff': tuple(config.pip['teff_range']),
                                   'radius': (0.3, 3.)},
                           inc_phot=True,
                           inc_spec=False)
    fit.run_multinest(tag=config.cmp[i]['name_witp'],
                      n_live_points=1000,
                      output='multinest/')
    species.plot_posterior(tag=config.cmp[i]['name_witp'],
                           burnin=None,
                           title=None,
                           offset=(-0.25, -0.25),
                           title_fmt='.2f',
                           inc_luminosity=True,
                           inc_mass=True,
                           output=pmdir+'posterior_'+config.cmp[i]['name_witp']+'.pdf')
    model_param = database.get_median_sample(tag=config.cmp[i]['name_witp'],
                                             burnin=None)
    
    # Plot median model and synthetic NIRCam photometry.
    readmodel = species.ReadModel(config.pip['model_spec'], wavel_range=(0.8, 5.2))
    modelbox = readmodel.get_model(model_param, spec_res=100., smooth=True)
    objectbox = database.get_object(object_name=config.cmp[i]['name_spec'])
    synphotbox = species.multi_photometry(datatype='model',
                                          spectrum=config.pip['model_spec'],
                                          filters=filters,
                                          parameters=model_param)
    res_box = species.get_residuals(datatype='model',
                                    spectrum=config.pip['model_spec'],
                                    parameters=model_param,
                                    objectbox=objectbox,
                                    inc_phot=True,
                                    inc_spec=False)
    species.plot_spectrum(boxes=[modelbox, objectbox, synphotbox],
                          filters=filters,
                          residuals=res_box,
                          plot_kwargs=[{'ls': '-', 'lw': 1., 'color': 'black'},
                                       None,
                                       None],
                          xlim=(0.8, 5.2),
                          scale=('linear', 'linear'),
                          title=config.cmp[i]['name_spec'],
                          offset=(-0.45, -0.04),
                          legend={'loc': 'upper right', 'frameon': False, 'fontsize': 12},
                          figsize=(9., 4),
                          quantity='flux',
                          output=pmdir+'spectrum_'+config.cmp[i]['name_witp']+'.pdf')
    
    # Save synthetic NIRCam photometry.
    for j in range(len(filters)):
        synphot = species.SyntheticPhotometry(filters[j])
        app_mag, abs_mag = synphot.flux_to_magnitude(synphotbox.flux[filters[j]], error=0., distance=(config.src[0]['dist'], 0.))
        temp = np.array([app_mag[0]])
        np.save(pmdir+config.cmp[i]['name_witp']+'_'+filters[j][-5:]+'.npy', temp)

print('DONE')
