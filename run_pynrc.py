from __future__ import division

import matplotlib
matplotlib.rcParams.update({'font.size': 14})


# =============================================================================
# IMPORTS
# =============================================================================

import yaml
config = yaml.load(open('config.yaml'), Loader=yaml.BaseLoader)

import sys
sys.path.append(config['paths']['whereistheplanet_dir'])
sys.path.append(config['paths']['webbpsf_ext_dir'])
sys.path.append(config['paths']['pynrc_dir'])

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import datetime
import os

from astropy.time import Time
from copy import deepcopy
from matplotlib.patches import Circle

import pynrc
import pynrc.nrc_utils as nrc_utils
import utils
import whereistheplanet

from pynrc.nb_funcs import plot_hdulist, plot_contrasts, plot_planet_patches, plot_contrasts_mjup
from pynrc.simul.ngNRC import slope_to_ramp


# =============================================================================
# PARAMETERS
# =============================================================================

# Source parameters.
# Name, dist (pc), age (Myr), sptype, teff (K), feh (dex), logg, kmag, band.
bp_k = pynrc.bp_2mass('k')
sci_source = (config['sources']['sci']['name'],
              float(config['sources']['sci']['dist']),
              float(config['sources']['sci']['age']),
              config['sources']['sci']['sptype'],
              float(config['sources']['sci']['teff']),
              float(config['sources']['sci']['feh']),
              float(config['sources']['sci']['logg']),
              float(config['sources']['sci']['kmag']),
              bp_k)
ref_source = (config['sources']['ref']['name'],
              config['sources']['ref']['sptype'],
              float(config['sources']['ref']['teff']),
              float(config['sources']['ref']['feh']),
              float(config['sources']['ref']['logg']),
              float(config['sources']['ref']['kmag']),
              bp_k)

# Observation date.
time = config['observation']['date']
t = Time(time, format='isot', scale='utc')
obs_time = datetime.datetime(int(time[0:4]),
                             int(time[5:7]),
                             int(time[8:10]),
                             int(time[11:13]),
                             int(time[14:16]),
                             int(time[17:19]))

# Companion locations on observation date.
keys = config['companions'].keys()
tags = []
locs = []
for key in keys:
    tags += [config['companions'][key]['name_witp']]
    try:
        temp = whereistheplanet.predict_planet(tags[-1], time_mjd=t.mjd)
        # locs += [(-temp[0][0]/1e3, temp[1][0]/1e3)]
        locs += [(temp[0][0]/1e3, temp[1][0]/1e3)]
    except:
        locs += [(float(config['companions'][key]['ra_off'])/1e3, float(config['companions'][key]['de_off'])/1e3)]
        print('Companion '+tags[-1]+' is not in whereistheplanet, using offset from config file')
        print('RA = %.3f mas, DEC = %.3f mas' % (locs[-1][0], locs[-1][1]))

# Companion magnitudes.
pmdir = config['paths']['wdir']+config['paths']['pmdir']

# Roll angles.
PA1 = float(config['observation']['pa1'])
PA2 = float(config['observation']['pa2'])

# Wavefront drifts.
wfe_drift0 = float(config['observation']['wfe_drift0'])
wfe_ref_drift = float(config['observation']['wfe_ref_drift'])
wfe_roll_drift = float(config['observation']['wfe_roll_drift'])
wfe_drifts = [wfe_drift0, wfe_ref_drift, wfe_roll_drift]

# Observing setup.
xml_path = config['paths']['wdir']+config['apt']['xml_path']
keys = config['observation']['sequences'].keys()
seqs = []
sets = []
for key in keys:
    seqs += [config['observation']['sequences'][key].split(',')]
    sets += [utils.read_from_xml(xml_path, seqs[-1])]
oversample = int(config['observation']['oversample'])

# Detection significance for contrast curves and companion properties.
nsig = 5
keys = config['companions'].keys()
mass = []
sent = []
for key in keys:
    mass += [float(config['companions'][key]['mass'])]
    sent += [float(config['companions'][key]['sent'])]

# Set output directory and enable or disable plots.
odir = config['paths']['wdir']+config['paths']['pynrc_data_dir']
if (not os.path.exists(odir)):
    os.makedirs(odir)
make_plots = config['pipeline']['make_plots'] == 'True'
if (make_plots == True):
    fdir = config['paths']['wdir']+config['paths']['pynrc_figs_dir']
    if (not os.path.exists(fdir)):
        os.makedirs(fdir)


# =============================================================================
# SOURCES
# =============================================================================

# Make source spectra.
name_sci, dist_sci, age_sci, sptype_sci, teff_sci, feh_sci, logg_sci, kmag_sci, bp_sci = sci_source
name_ref, sptype_ref, teff_ref, feh_ref, logg_ref, kmag_ref, bp_ref = ref_source
sp_sci = pynrc.stellar_spectrum(sptype_sci, kmag_sci, 'vegamag', bp_sci, Teff=teff_sci, metallicity=feh_sci, log_g=logg_sci)
sp_sci.name = name_sci
sp_ref = pynrc.stellar_spectrum(sptype_ref, kmag_ref, 'vegamag', bp_ref, Teff=teff_ref, metallicity=feh_ref, log_g=logg_ref)
sp_ref.name = name_ref

# # Ramp optimization.
# pl = pynrc.planets_sb12(atmo='hy3s', mass=mass[0], age=age_sci, entropy=sent[0], distance=dist_sci)
# sp_pl = pl.export_pysynphot()
# bp_pl = pynrc.read_filter(sets[0]['filts'][0])
# temp = np.load(pmdir+tags[0]+'_'+sets[0]['filts'][0]+'.npy')[0]
# sp_pl = sp_pl.renorm(temp, 'vegamag', bp_pl)
# nrc = pynrc.NIRCam(sets[0]['filts'][0], pupil=sets[0]['pups'][0], mask=sets[0]['masks'][0], wind_mode=sets[0]['wind'][0], xpix=sets[0]['subs'][0], ypix=sets[0]['subs'][0])
# res = nrc.ramp_optimize(sp_pl, sp_bright=sp_sci, tacq_max=3600, tacq_frac=0.05, even_nints=True, verbose=True)
# import pdb; pdb.set_trace()

# Plot source spectra.
if (make_plots == True):
    bp = pynrc.read_filter(sets[0]['filts'][0], sets[0]['pups'][0], sets[0]['masks'][0])
    f, ax = plt.subplots(1, 1)
    xr = [2.5, 5.5]
    for sp in [sp_sci, sp_ref]:
        wave = sp.wave/1e4 # microns
        ww = (wave >= xr[0]) & (wave <= xr[1])
        sp.convert('Jy')
        flux = sp.flux/np.interp(4., wave, sp.flux) # normalize source spectra at 4 microns
        ax.semilogy(wave[ww], flux[ww], lw=1.5, label=sp.name)
        sp.convert('flam')
    ax.set_xlim(xr)
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel('Flux (Jy) normalized at 4 μm')
    ax.set_title('Source spectra')
    ax2 = ax.twinx()
    ax2.plot(bp.wave/1e4, bp.throughput, color='C2', label=bp.name+' bandpass')
    ax2.set_xlim(xr)
    ax2.set_ylim([0, 0.8])
    ax2.set_ylabel('Bandpass throughput')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(fdir+'source_spectra.pdf')
    plt.close()


# =============================================================================
# RAMPS
# =============================================================================

# Go through all observations and filters.
nobs = len(sets)
ctr1 = 0
for i in range(nobs):
    print('Simulating sequence '+str(i+1))
    nfilts = len(sets[i]['filts'])
    ctr2 = 0
    for j in range(nfilts):
        print('   Filter '+sets[i]['filts'][j])
        
        # Initialize nrc_hci object.
        obs = pynrc.obs_hci(sp_sci,
                            sp_ref,
                            dist_sci,
                            filter=sets[i]['filts'][j],
                            mask=sets[i]['masks'][j],
                            pupil=sets[i]['pups'][j],
                            wfe_ref_drift=wfe_drift0,
                            fov_pix=sets[i]['fovp'][j],
                            oversample=oversample,
                            wind_mode=sets[i]['wind'][j],
                            xpix=sets[i]['subs'][j],
                            ypix=sets[i]['subs'][j],
                            verbose=False,
                            bar_offset=sets[i]['boff'][j])
        
        # Update readout mode.
        obs.update_detectors(read_mode=sets[i]['read_sci'][j], ngroup=sets[i]['ngrp_sci'][j], nint=sets[i]['nint_sci'][j], verbose=False)
        obs.nrc_ref.update_detectors(read_mode=sets[i]['read_ref'][j], ngroup=sets[i]['ngrp_ref'][j], nint=sets[i]['nint_ref'][j])
        
        # Add companions to observation class.
        obs.kill_planets()
        for k, loc in enumerate(locs):
            temp = np.load(pmdir+tags[k]+'_'+sets[i]['filts'][j]+'.npy')[0]
            obs.add_planet(mass=mass[k], entropy=sent[k], age=age_sci, xy=loc, runits='arcsec', renorm_args=(temp, 'vegamag', obs.bandpass))
        
        if (make_plots == True):
            
            # Generate image.
            im_planets = obs.gen_planets_image(PA_offset=PA1)
            
            # Plot image.
            f, ax = plt.subplots(1, 1, figsize=(6.4, 6.4))
            xasec = obs.det_info['xpix']*obs.pix_scale # arcsec
            yasec = obs.det_info['ypix']*obs.pix_scale # arcsec
            # extent = [-xasec/2., xasec/2., -yasec/2., yasec/2.] # arcsec
            extent = [xasec/2., -xasec/2., -yasec/2., yasec/2.] # arcsec
            xylim = 3. # arcsec
            xlim = ylim = np.array([-1, 1])*xylim
            ax.imshow(im_planets, extent=extent, vmin=0., vmax=0.75*np.nanmax(im_planets))
            
            detid = obs.Detectors[0].detid
            im_mask = obs.mask_images[detid]
            masked = np.ma.masked_where(im_mask > 0.975, im_mask)
            ax.imshow(masked, extent=extent, alpha=1./3., cmap='Greys', vmin=-0.5)
            
            xc_off = obs.bar_offset
            for loc in locs:
                xc, yc = deepcopy(loc)
                xc, yc = nrc_utils.xy_rot(xc, yc, PA1)
                xc += xc_off
                # circle = Circle((xc, yc), radius=xylim/15., lw=1., edgecolor='red', facecolor='none')
                circle = Circle((-xc, yc), radius=xylim/15., lw=1., edgecolor='red', facecolor='none')
                ax.add_artist(circle)
            # ax.set_xlim(xlim+xc_off)
            ax.set_xlim(xlim-xc_off)
            ax.set_ylim(ylim)
            ax.set_xlabel('Arcsec')
            ax.set_ylabel('Arcsec')
            ax.set_title('{} planets - {} {}'.format(name_sci, obs.filter, obs.mask))
            ax.tick_params(axis='both', color='white', which='both')
            for k in ax.spines.keys():
                ax.spines[k].set_color('white')
            # nrc_utils.plotAxes(ax, width=1., headwidth=5., alength=0.15, angle=PA1, position=(0.9, 0.9), label1='E', label2='N')
            nrc_utils.plotAxes(ax, width=1., headwidth=5., alength=0.15, angle=PA1, position=(0.7, 0.7), label1='E', label2='N', dir1=[1, 0], dir2=[0, -1])
            plt.tight_layout()
            plt.savefig(fdir+'seq_%03.0f_filt_' % i+sets[i]['filts'][j]+'_planets.pdf')
            plt.close()
            
            # Generate images.
            hdul_dict = {}
            for k, wfe_drift in enumerate(wfe_drifts):
                obs.wfe_ref_drift = wfe_drift
                hdul = obs.gen_roll_image(PA1=PA1, PA2=PA2)
                hdul_dict[wfe_drift] = hdul
            
            # Plot images.
            f, ax = plt.subplots(1, len(wfe_drifts), figsize=(4.8*len(wfe_drifts), 4.8))
            if (len(wfe_drifts) == 1):
                ax = [ax]
            xasec = obs.det_info['xpix']*obs.pix_scale # arcsec
            yasec = obs.det_info['ypix']*obs.pix_scale # arcsec
            # extent = [-xasec/2., xasec/2., -yasec/2., yasec/2.] # arcsec
            extent = [xasec/2., -xasec/2., -yasec/2., yasec/2.] # arcsec
            xylim = 2.5 # arcsec
            xlim = ylim = np.array([-1, 1])*xylim
            for k, wfe_drift in enumerate(wfe_drifts):
                hdul = hdul_dict[wfe_drift]
                plot_hdulist(hdul, xr=xlim, yr=ylim, ax=ax[k], vmin=0., vmax=0.75*np.nanmax(hdul[0].data))
                for loc in locs:
                    xc, yc = deepcopy(loc)
                    circle = Circle((xc, yc), radius=xylim/15., lw=1., edgecolor='red', facecolor='none')
                    ax[k].add_artist(circle)
                # ax[k].set_xlim(xlim)
                ax[k].set_xlim(-xlim)
                ax[k].set_ylim(ylim)
                ax[k].set_xlabel('Arcsec')
                ax[k].set_ylabel('Arcsec')
                ax[k].set_title('$\Delta$WFE = {:.0f} nm'.format(wfe_drift))
                ax[k].tick_params(axis='both', color='white', which='both')
                for l in ax[k].spines.keys():
                    ax[k].spines[l].set_color('white')
                nrc_utils.plotAxes(ax[k], width=1., headwidth=5., alength=0.15, position=(0.9, 0.7), label1='E', label2='N')
            plt.suptitle('{} planets - {} {}'.format(name_sci, obs.filter, obs.mask))
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
            plt.savefig(fdir+'seq_%03.0f_filt_' % i+sets[i]['filts'][j]+'_roll.pdf')
            plt.close()
            
            # Calculate contrast curves.
            curves = []
            for k, wfe_drift in enumerate(wfe_drifts):
                obs.wfe_ref_drift = wfe_drift
                curves += [obs.calc_contrast(roll_angle=np.abs(PA1-PA2), nsig=nsig)]
            
            # Plot contrast curves.
            f, ax = plt.subplots(1, 2, figsize=(6.4*2, 4.8*1))
            xr = [0., 5.] # arcsec
            yr = [24., 8.] # mag
            plot_contrasts(curves, nsig, wfe_drifts, obs=obs, xr=xr, yr=yr, ax=ax[0], return_axes=False)
            seps = [np.sqrt(x**2+y**2) for x, y in locs]
            mags = []
            for k in range(len(tags)):
                mags += [np.load(pmdir+tags[k]+'_'+sets[i]['filts'][j]+'.npy')[0]]
            ax[0].plot(seps, mags, marker='o', ls='None', label='Companions ({})'.format(obs.filter), color='black', zorder=10)
            plot_planet_patches(ax[0], obs, age=age_sci, entropy=np.mean(sent), av_vals=None)
            ax[0].legend(ncol=2, fontsize=10)
            plot_contrasts_mjup(curves, nsig, wfe_drifts, obs=obs, age=age_sci, ax=ax[1], twin_ax=True, xr=xr, yr=None, linder_models=False)
            ax[1].set_yscale('log')
            ax[1].set_ylim([0.05, 100.]) # Jupiter masses
            ax[1].legend(loc='upper right', title='COND ({:.0f} Myr)'.format(age_sci), fontsize=10)
            plt.suptitle('{} planets - {} {}'.format(name_sci, obs.filter, obs.mask))
            plt.tight_layout()
            plt.subplots_adjust(top=0.80)
            plt.savefig(fdir+'seq_%03.0f_filt_' % i+sets[i]['filts'][j]+'_contrast.pdf')
            plt.close()
        
        # Activate wavefront drifts.
        obs.wfe_drift = True
        
        # Roll 1.
        if (sets[i]['dithers'][0] == 'NONE'):
            
            # Generate noiseless slope image.
            im_slope = obs.gen_slope_image(PA=PA1,
                                           exclude_disk=True,
                                           exclude_planets=False,
                                           exclude_noise=True,
                                           zfact=2.5,
                                           do_ref=False,
                                           do_roll2=False,
                                           im_star=None,
                                           wfe_drift0=wfe_drift0,
                                           wfe_ref_drift=wfe_ref_drift,
                                           wfe_roll_drift=wfe_roll_drift)
            det = obs.Detectors[0]
            im_slope = nrc_utils.sci_to_det(im_slope, det.detid)
            
            # Convert slope to ramp (the ramp will be saved automatically).
            file_out = odir+'obs_%03.0f_filt_' % sets[i]['nums'][0]+sets[i]['filts'][j]+'_ramp.fits'
            hdul = slope_to_ramp(det,
                                 im_slope=im_slope,
                                 out_ADU=True,
                                 file_out=file_out,
                                 filter=obs.filter,
                                 pupil=obs.pupil,
                                 obs_time=obs_time,
                                 targ_name=name_sci,
                                 DMS=True,
                                 dark=True,
                                 bias=True,
                                 det_noise=True,
                                 return_results=True)
            
            # Save noiseless slope image.
            hdul['SCI'].data = im_slope
            hdul.pop(2)
            hdul.writeto(file_out[:-9]+'nlslope.fits', output_verify='fix', overwrite=True)
        
        else:
            raise UserWarning('Does not support dither pattern '+sets[i]['dithers'][0]+' for roll 1')
        
        # Roll 2.
        if (sets[i]['dithers'][1] == 'NONE'):
            
            # Generate noiseless slope image.
            im_slope = obs.gen_slope_image(PA=PA2,
                                           exclude_disk=True,
                                           exclude_planets=False,
                                           exclude_noise=True,
                                           zfact=2.5,
                                           do_ref=False,
                                           do_roll2=True,
                                           im_star=None,
                                           wfe_drift0=wfe_drift0,
                                           wfe_ref_drift=wfe_ref_drift,
                                           wfe_roll_drift=wfe_roll_drift)
            det = obs.Detectors[0]
            im_slope = nrc_utils.sci_to_det(im_slope, det.detid)
            
            # Convert slope to ramp (the ramp will be saved automatically).
            file_out = odir+'obs_%03.0f_filt_' % sets[i]['nums'][1]+sets[i]['filts'][j]+'_ramp.fits'
            hdul = slope_to_ramp(det,
                                 im_slope=im_slope,
                                 out_ADU=True,
                                 file_out=file_out,
                                 filter=obs.filter,
                                 pupil=obs.pupil,
                                 obs_time=obs_time,
                                 targ_name=name_sci,
                                 DMS=True,
                                 dark=True,
                                 bias=True,
                                 det_noise=True,
                                 return_results=True)
            
            # Save noiseless slope image.
            hdul['SCI'].data = im_slope
            hdul.pop(2)
            hdul.writeto(file_out[:-9]+'nlslope.fits', output_verify='fix', overwrite=True)
        
        else:
            raise UserWarning('Does not support dither pattern '+sets[i]['dithers'][1]+' for roll 2')
        
        # Reference.
        if (sets[i]['dithers'][2] == 'NONE'):
            
            # Generate noiseless slope image.
            im_slope = obs.gen_slope_image(PA=PA1,
                                           exclude_disk=True,
                                           exclude_planets=True,
                                           exclude_noise=True,
                                           zfact=2.5,
                                           do_ref=True,
                                           do_roll2=False,
                                           im_star=None,
                                           wfe_drift0=wfe_drift0,
                                           wfe_ref_drift=wfe_ref_drift,
                                           wfe_roll_drift=wfe_roll_drift)
            det = obs.Detectors[0]
            im_slope = nrc_utils.sci_to_det(im_slope, det.detid)
            
            # Convert slope to ramp (the ramp will be saved automatically).
            file_out = odir+'obs_%03.0f_filt_' % sets[i]['nums'][2]+sets[i]['filts'][j]+'_ramp.fits'
            hdul = slope_to_ramp(det,
                                 im_slope=im_slope,
                                 out_ADU=True,
                                 file_out=file_out,
                                 filter=obs.filter,
                                 pupil=obs.pupil,
                                 obs_time=obs_time,
                                 targ_name=name_sci,
                                 DMS=True,
                                 dark=True,
                                 bias=True,
                                 det_noise=True,
                                 return_results=True)
            
            # Save noiseless slope image.
            hdul['SCI'].data = im_slope
            hdul.pop(2)
            hdul.writeto(file_out[:-9]+'nlslope.fits', output_verify='fix', overwrite=True)
        
        else:
            if (sets[i]['dithers'][2] == '5-POINT-BOX'):
                dithers = [(0., 0.), (15., 15.), (-15., 15.), (-15., -15.), (15., -15.)] # mas
            elif (sets[i]['dithers'][2] == '5-POINT-DIAMOND'):
                dithers = [(0., 0.), (0., 20.), (0., -20.), (20., 0.), (-20., 0.)] # mas
            elif (sets[i]['dithers'][2] == '9-POINT-CIRCLE'):
                dithers = [(0., 0.), (0., 20.), (-15., 15.), (-20., 0.), (-15., -15.), (0., -20.), (15., -15.), (20., 0.), (15., 15.)] # mas
            elif (sets[i]['dithers'][2] == '3-POINT-BAR'):
                dithers = [(0., 0.), (0., 15.), (0., -15.)] # mas
            elif (sets[i]['dithers'][2] == '5-POINT-BAR'):
                dithers = [(0., 0.), (0., 20.), (0., 10.), (0., -10.), (0., -20.)] # mas
            else:
                raise UserWarning('Does not support dither pattern '+sets[i]['dithers'][2]+' for reference')
            for k in range(len(dithers)):
                
                # Generate noiseless slope image.
                im_slope = obs.gen_slope_image(PA=PA1,
                                               exclude_disk=True,
                                               exclude_planets=True,
                                               exclude_noise=True,
                                               zfact=2.5,
                                               do_ref=True,
                                               do_roll2=False,
                                               im_star=None,
                                               wfe_drift0=wfe_drift0,
                                               wfe_ref_drift=wfe_ref_drift,
                                               wfe_roll_drift=wfe_roll_drift,
                                               dither=dithers[k])
                det = obs.nrc_ref.Detectors[0]
                im_slope = nrc_utils.sci_to_det(im_slope, det.detid)
                
                # Convert slope to ramp (the ramp will be saved automatically).
                file_out = odir+'obs_%03.0f_filt_' % sets[i]['nums'][2]+sets[i]['filts'][j]+'_dpos_%03.0f' % k+'_ramp.fits'
                hdul = slope_to_ramp(det,
                                     im_slope=im_slope,
                                     out_ADU=True,
                                     file_out=file_out,
                                     filter=obs.filter,
                                     pupil=obs.pupil,
                                     obs_time=obs_time,
                                     targ_name=name_sci,
                                     DMS=True,
                                     dark=True,
                                     bias=True,
                                     det_noise=True,
                                     return_results=True)
                
                # Save noiseless slope image.
                hdul['SCI'].data = im_slope
                hdul.pop(2)
                hdul.writeto(file_out[:-9]+'nlslope.fits', output_verify='fix', overwrite=True)
    
print('DONE')
