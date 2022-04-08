from __future__ import division

import matplotlib
matplotlib.rcParams.update({'font.size': 14})


# =============================================================================
# IMPORTS
# =============================================================================

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import datetime
import os
import sys

from astropy import units as u
from astropy.coordinates import SkyCoord, Distance
from astropy.time import Time
from tqdm.auto import trange, tqdm

import pynrc
from pynrc.simul.ngNRC import make_gaia_source_table
from pynrc.simul.ngNRC import create_level1b_FITS
pynrc.setup_logging('WARN', verbose=False)

import util

if __name__ == '__main__':
    
    # =============================================================================
    # PARAMETERS
    # =============================================================================
    
    # Read config.
    config = util.config()
    
    # Append whereistheplanet path.
    sys.path.append(config.paths['whereistheplanet_dir'])
    import whereistheplanet
    
    # Source parameters.
    # Name, dist (pc), age (Myr), sptype, teff (K), feh (dex), logg, kmag, band.
    bp_k = pynrc.bp_2mass('k')
    sci_source = (config.src[0]['name'],
                  config.src[0]['dist'],
                  config.src[0]['age'],
                  config.src[0]['sptype'],
                  config.src[0]['teff'],
                  config.src[0]['feh'],
                  config.src[0]['logg'],
                  config.src[0]['kmag'],
                  bp_k)
    ref_source = (config.src[1]['name'],
                  config.src[1]['sptype'],
                  config.src[1]['teff'],
                  config.src[1]['feh'],
                  config.src[1]['logg'],
                  config.src[1]['kmag'],
                  bp_k)
    
    # Observation date.
    time = config.obs['date']
    t = Time(time, format='isot', scale='utc')
    obs_time = datetime.datetime(int(time[0:4]),
                                 int(time[5:7]),
                                 int(time[8:10]),
                                 int(time[11:13]),
                                 int(time[14:16]),
                                 int(time[17:19]))
    
    # Companion locations on observation date.
    tags = []
    locs = []
    for i in range(len(config.cmp)):
        tags += [config.cmp[i]['name_witp']]
        if ((config.cmp[i]['ra_off'] is None) or (config.cmp[i]['de_off'] is None) or (config.cmp[i]['ra_off'] == 0. and config.cmp[i]['de_off'] == 0.)):
            temp = whereistheplanet.predict_planet(tags[-1], time_mjd=t.mjd)
            locs += [(temp[0][0]/1e3, temp[1][0]/1e3)]
            print('Companion '+tags[-1]+' found in whereistheplanet!')
        else:
            locs += [(config.cmp[i]['ra_off']/1e3, config.cmp[i]['de_off']/1e3)]
            print('Companion '+tags[-1]+' found in config file')
            print('RA = %.3f arcsec, DEC = %.3f arcsec' % (locs[-1][0], locs[-1][1]))
    
    # Companion magnitudes.
    pmdir = config.paths['wdir']+config.paths['pmdir']
    
    # Set output directory and enable or disable plots.
    odir = config.paths['wdir']+config.paths['pynrc_data_dir']
    if (not os.path.exists(odir)):
        os.makedirs(odir)
    make_plots = config.pip['make_plots']
    if (make_plots == True):
        fdir = config.paths['wdir']+config.paths['pynrc_figs_dir']
        if (not os.path.exists(fdir)):
            os.makedirs(fdir)
    
    # APT input files.
    json_file = config.paths['wdir']+config.paths['xml_path'][:-4]+'.timing.json'
    sm_acct_file = config.paths['wdir']+config.paths['xml_path'][:-4]+'.smart_accounting'
    pointing_file = config.paths['wdir']+config.paths['xml_path'][:-4]+'.pointing'
    xml_file = config.paths['wdir']+config.paths['xml_path']
    
    
    # =============================================================================
    # SOURCES
    # =============================================================================
    
    # Make source spectra.
    name_sci, dist_sci, age_sci, sptype_sci, teff_sci, feh_sci, logg_sci, kmag_sci, bp_sci = sci_source
    vot = config.paths['wdir']+config.src[0]['vot']
    args = (name_sci, sptype_sci, kmag_sci, bp_sci, vot)
    kwargs = {'Teff': teff_sci, 'metallicity': feh_sci, 'log_g': logg_sci}
    src = pynrc.source_spectrum(*args, **kwargs)
    src.fit_SED(use_err=False, robust=False, wlim=[0.5, 10.0])
    sp_sci = src.sp_model
    name_ref, sptype_ref, teff_ref, feh_ref, logg_ref, kmag_ref, bp_ref = ref_source
    vot = config.paths['wdir']+config.src[1]['vot']
    args = (name_ref, sptype_ref, kmag_ref, bp_ref, vot)
    kwargs = {'Teff': teff_ref, 'metallicity': feh_ref, 'log_g': logg_ref}
    ref = pynrc.source_spectrum(*args, **kwargs)
    ref.fit_SED(use_err=False, robust=False, wlim=[0.5, 10.0])
    sp_ref = ref.sp_model
    
    # Plot source spectra.
    if (make_plots == True):
        f, ax = plt.subplots(1, 2, figsize=(2*6.4, 1*4.8))
        src.plot_SED(xr=[0.3, 10.0], ax=ax[0])
        ref.plot_SED(xr=[0.3, 10.0], ax=ax[1])
        ax[0].set_title('Science spectrum -- {} ({})'.format(src.name, sptype_sci))
        ax[1].set_title('Reference spectrum -- {} ({})'.format(ref.name, sptype_ref))
        plt.tight_layout()
        plt.savefig(fdir+'source_spectra.pdf')
        plt.close()
    
    
    # =============================================================================
    # RAMPS
    # =============================================================================
    
    # Go through all observations and filters.
    print('Simulating DMS files')
    nobs = len(config.obs['filter'])
    for i in range(nobs):
        nfilts = len(config.obs['filter'][i][0])
        for j in range(nfilts):
            
            # Initialize target dictionary.
            targ_dict = {}
            
            # Add science target.
            params_companions = {}
            bp_pl_norm = pynrc.read_filter(config.obs['filter'][i][0][j])
            for k, loc in enumerate(locs):
                temp = np.load(pmdir+tags[k]+'_'+config.obs['filter'][i][0][j]+'.npy')[0]
                params_companions[tags[k]] = {'xy': (-locs[k][0], locs[k][1]),
                                              'runits': 'arcsec',
                                              'mass': config.cmp[k]['mass'],
                                              'renorm_args': (temp, 'vegamag', bp_pl_norm)}
            targ_dict[name_sci] = {'type': 'FixedTargetType',
                                   'TargetName': name_sci,
                                   'TargetArchiveName': name_sci,
                                   'EquatorialCoordinates': config.src[0]['icrs'],
                                   'RAProperMotion': config.src[0]['pmra']*u.mas/u.yr,
                                   'DecProperMotion': config.src[0]['pmde']*u.mas/u.yr,
                                   'parallax': 1e3/config.src[0]['dist']*u.mas,
                                   'age_Myr': age_sci,
                                   'params_star': {'sp': sp_sci},
                                   'params_companions': params_companions,
                                   'params_disk_model': None,
                                   'src_tbl' : None,
                                   }
            
            # Add reference target.
            targ_dict[name_ref] = {'type': 'FixedTargetType',
                                   'TargetName': name_ref,
                                   'TargetArchiveName': name_ref,
                                   'EquatorialCoordinates': config.src[1]['icrs'],
                                   'RAProperMotion': config.src[1]['pmra']*u.mas/u.yr,
                                   'DecProperMotion': config.src[1]['pmde']*u.mas/u.yr,
                                   'parallax': 1e3/config.src[1]['dist']*u.mas,
                                   'age_Myr': None,
                                   'params_star': {'sp': sp_ref},
                                   'params_companions': None,
                                   'params_disk_model': None,
                                   'src_tbl' : None,
                                   }
            
            # Add Gaia sources.
            for k in targ_dict.keys():
                d = targ_dict[k]
                dist = Distance(parallax=d['parallax']) if d['parallax'] is not None else None
                c = SkyCoord(d['EquatorialCoordinates'],
                             frame='icrs',
                             unit=(u.hourangle, u.deg),
                             pm_ra_cosdec=d['RAProperMotion'],
                             pm_dec=d['DecProperMotion'],
                             distance=dist,
                             obstime='J2000')
                d['sky_coords'] = c
                d['ra_J2000'], d['dec_J2000'] = (c.ra.deg, c.dec.deg)
                d['dist_pc'] = c.distance.value if dist is not None else None
                src_tbl = make_gaia_source_table(c)
                d['src_tbl'] = src_tbl if len(src_tbl) > 0 else None
            
            # Set simulation parameters.
            sim_config = {'json_file': json_file,
                          'sm_acct_file': sm_acct_file,
                          'pointing_file': pointing_file,
                          'xml_file': xml_file,
                          'save_dir': odir,
                          'rand_seed_init': 1234,
                          'obs_date': config.obs['date'][:10],
                          'obs_time': config.obs['date'][11:],
                          'pa_v3': np.mean(np.unique(config.obs['pa'][0])),
                          'params_targets': targ_dict,
                          'params_webbpsf': {'fov_pix': None, 'oversample': int(config.obs['oversample'])},
                          'params_psfconv': {'npsf_per_full_fov': 9, 'osamp': 1, 'sptype': 'G0V'},
                          'params_wfedrift': {'case': 'BOL', 'slew_init': 10, 'plot': True, 'figname': 'wfe.pdf'},
                          'large_grid': True,
                          'large_slew': 100.0, # mas
                          'ta_sam': 5.0, # mas
                          'std_sam': 5.0, # mas
                          'sgd_sam': 2.5, # mas
                          'save_slope': True,
                          'save_dms': True,
                          'dry_run': False,
                          'params_noise': {'include_poisson': True, # photon noise
                                           'include_dark': True, # dark current
                                           'include_bias': True, # bias offset
                                           'include_ktc': True, # kTC noise
                                           'include_rn': True, # read noise
                                           'include_cpink': True, # correlated 1/f noise between channels
                                           'include_upink': True, # channel-dependent 1/f noise
                                           'include_acn': True, # alternating column noise
                                           'apply_ipc': True, # interpixel capacitance
                                           'apply_ppc': True, # post-pixel coupling
                                           'amp_crosstalk': True, # amplifier crosstalk
                                           'include_refoffsets': True, # reference offsets
                                           'include_refinst': True, # reference pixel instabilities
                                           'include_colnoise': True, # transient detector column noise
                                           'add_crs': True, # cosmic rays
                                           'cr_model': 'SUNMAX', # cosmic ray model ('SUNMAX', 'SUNMIN', or 'FLARES')
                                           'cr_scale': 1, # cosmic ray probability scaling
                                           'apply_nonlinearity': True, # apply non-linearity
                                           'random_nonlin': True, # add randomness to non-linearity
                                           'apply_flats': True, # pixel-to-pixel QE variations and field-dependent illumination
                                           },
                          }
            
            # Create slope and DMS files.
            for k in range(len(config.obs['ind'][i])):
                visit_id = '%03.0f' % int(config.obs['ind'][i][k])+':001'
                print('   Visit '+visit_id)
                
                det = config.obs['detector'][i][k]
                if (det == 'NRCALONG'):
                    det = 'NRCA5'
                apname = det+'_TA'+config.obs['mask'][i][k]
                create_level1b_FITS(sim_config,
                                    dry_run=False,
                                    save_slope=True,
                                    save_dms=True,
                                    visit_id=visit_id,
                                    apname=apname)
                
                if (config.obs['conf'][i][k] == True):
                    apname = det+'_FULL_TA'+config.obs['mask'][i][k]
                    create_level1b_FITS(sim_config,
                                        dry_run=False,
                                        save_slope=True,
                                        save_dms=True,
                                        visit_id=visit_id,
                                        apname=apname)
                    apname = det+'_FULL_'+config.obs['mask'][i][k]
                    create_level1b_FITS(sim_config,
                                        dry_run=False,
                                        save_slope=True,
                                        save_dms=True,
                                        visit_id=visit_id,
                                        apname=apname)
                
                create_level1b_FITS(sim_config,
                                    dry_run=False,
                                    save_slope=True,
                                    save_dms=True,
                                    visit_id=visit_id,
                                    filter=config.obs['filter'][i][0][j])
    
    print('DONE')
