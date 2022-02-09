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

from astropy.time import Time
from copy import deepcopy
from matplotlib.patches import Circle
from scipy.ndimage import shift

import webbpsf

import util


# =============================================================================
# PARAMETERS
# =============================================================================

# Read config.
config = util.config()

# NIRCam pixel scale.
nc_temp = webbpsf.NIRCam()
pixscale_SW = nc_temp._pixelscale_short
pixscale_LW = nc_temp._pixelscale_long
del nc_temp

# Append whereistheplanet, WebbPSF_ext, and pyNRC paths.
sys.path.append(config.paths['whereistheplanet_dir'])
import whereistheplanet
sys.path.append(config.paths['webbpsf_ext_dir'])
sys.path.append(config.paths['pynrc_dir'])
import pynrc
import pynrc.nrc_utils as nrc_utils
from pynrc.nb_funcs import plot_hdulist, plot_contrasts, plot_planet_patches, plot_contrasts_mjup
from pynrc.simul.ngNRC import slope_to_ramp

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

# Detection significance for contrast curves.
nsig = 5

# Set output directory and enable or disable plots.
odir = config.paths['wdir']+config.paths['pynrc_data_dir']
if (not os.path.exists(odir)):
    os.makedirs(odir)
make_plots = config.pip['make_plots']
if (make_plots == True):
    fdir = config.paths['wdir']+config.paths['pynrc_figs_dir']
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
# sp_sci.convert('Jy')
# sp_ref.convert('Jy')
# np.save(pmdir+'sci_wave.npy', sp_sci.wave/1e4) # microns
# np.save(pmdir+'sci_flux.npy', sp_sci.flux/1e4) # Jy
# np.save(pmdir+'ref_wave.npy', sp_ref.wave/1e4) # microns
# np.save(pmdir+'ref_flux.npy', sp_ref.flux/1e4) # Jy
# sp_sci.convert('flam')
# sp_ref.convert('flam')

# Plot source spectra.
if (make_plots == True):
    bp = pynrc.read_filter(config.obs['filter'][0][0][0], config.obs['pupil_pynrc'][0][0], config.obs['mask'][0][0])
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
nobs = len(config.obs['filter'])
ctr1 = 0
for i in range(nobs):
    print('Simulating sequence '+str(i+1))
    nfilts = len(config.obs['filter'][i][0])
    ctr2 = 0
    for j in range(nfilts):
        print('   Filter '+config.obs['filter'][i][0][j])
        
        # Initialize nrc_hci object.
        obs = pynrc.obs_hci(sp_sci,
                            sp_ref,
                            dist_sci,
                            filter=config.obs['filter'][i][0][j],
                            mask=config.obs['mask'][i][0],
                            pupil=config.obs['pupil_pynrc'][i][0],
                            wfe_ref_drift=config.obs['wfe'][i][0],
                            fov_pix=config.obs['fov_pix'][i][0],
                            oversample=config.obs['oversample'],
                            wind_mode=config.obs['wind_mode'][i][0],
                            xpix=config.obs['fov_pix'][i][0],
                            ypix=config.obs['fov_pix'][i][0],
                            verbose=False,
                            bar_offset=config.obs['baroff'][i][0][j])
        
        # Update readout mode.
        obs.update_detectors(read_mode=config.obs['readpatt'][i][0][j], ngroup=config.obs['ngroups'][i][0][j], nint=config.obs['nints'][i][0][j], verbose=False)
        obs.nrc_ref.update_detectors(read_mode=config.obs['readpatt'][i][2][j], ngroup=config.obs['ngroups'][i][2][j], nint=config.obs['nints'][i][2][j])
        
        # Add companions to observation class.
        obs.kill_planets()
        for k, loc in enumerate(locs):
            temp = np.load(pmdir+tags[k]+'_'+config.obs['filter'][i][0][j]+'.npy')[0]
            obs.add_planet(mass=config.cmp[k]['mass'], entropy=config.cmp[k]['sent'], age=age_sci, xy=loc, runits='arcsec', renorm_args=(temp, 'vegamag', obs.bandpass))
        
        if (make_plots == True):
            
            # Generate image.
            im_planets, cons, psfs = obs.gen_planets_image(PA_offset=config.obs['pa'][i][0], return_cons_and_psfs=True)
            np.save(fdir+'seq_%03.0f_filt_' % i+config.obs['filter'][i][0][j]+'_cons.npy', cons)
            np.save(fdir+'seq_%03.0f_filt_' % i+config.obs['filter'][i][0][j]+'_psfs_%+04.0fdeg.npy' % config.obs['pa'][i][0], psfs)
            _, _, psfs = obs.gen_planets_image(PA_offset=config.obs['pa'][i][1], return_cons_and_psfs=True)
            np.save(fdir+'seq_%03.0f_filt_' % i+config.obs['filter'][i][0][j]+'_psfs_%+04.0fdeg.npy' % config.obs['pa'][i][1], psfs)
            np.save(fdir+'seq_%03.0f_filt_' % i+config.obs['filter'][i][0][j]+'_starmag.npy', np.array(obs.star_flux(fluxunit='vegamag')))
            
            # Plot image.
            f, ax = plt.subplots(1, 1, figsize=(6.4, 6.4))
            xasec = obs.det_info['xpix']*obs.pix_scale # arcsec
            yasec = obs.det_info['ypix']*obs.pix_scale # arcsec
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
                xc, yc = nrc_utils.xy_rot(xc, yc, config.obs['pa'][i][0])
                xc += xc_off
                circle = Circle((-xc, yc), radius=xylim/15., lw=1., edgecolor='red', facecolor='none')
                ax.add_artist(circle)
            ax.set_xlim(xlim-xc_off)
            ax.set_ylim(ylim)
            ax.set_xlabel('Arcsec')
            ax.set_ylabel('Arcsec')
            ax.set_title('{} planets - {} {}'.format(name_sci, obs.filter, obs.mask))
            ax.tick_params(axis='both', color='white', which='both')
            for k in ax.spines.keys():
                ax.spines[k].set_color('white')
            nrc_utils.plotAxes(ax, width=1., headwidth=5., alength=0.15, angle=config.obs['pa'][i][0], position=(0.75, 0.75), label1='E', label2='N', dir1=[-1, 0], dir2=[0, 1])
            plt.tight_layout()
            plt.savefig(fdir+'seq_%03.0f_filt_' % i+config.obs['filter'][i][0][j]+'_planets.pdf')
            plt.close()
            
            # Generate images.
            hdul_dict = {}
            for k, wfe_drift in enumerate(np.array(config.obs['wfe'][i])[[0, 2, 1]]):
                obs.wfe_ref_drift = wfe_drift
                hdul = obs.gen_roll_image(PA1=config.obs['pa'][i][0], PA2=config.obs['pa'][i][1])
                hdul_dict[wfe_drift] = hdul
            
            # Plot images.
            f, ax = plt.subplots(1, len(config.obs['wfe'][i]), figsize=(4.8*len(config.obs['wfe'][i]), 4.8))
            if (len(config.obs['wfe'][i]) == 1):
                ax = [ax]
            xasec = obs.det_info['xpix']*obs.pix_scale # arcsec
            yasec = obs.det_info['ypix']*obs.pix_scale # arcsec
            extent = [xasec/2., -xasec/2., -yasec/2., yasec/2.] # arcsec
            xylim = 2.5 # arcsec
            xlim = ylim = np.array([-1, 1])*xylim
            for k, wfe_drift in enumerate(config.obs['wfe'][i]):
                hdul = hdul_dict[wfe_drift]
                plot_hdulist(hdul, xr=xlim, yr=ylim, ax=ax[k], vmin=0., vmax=0.75*np.nanmax(hdul[0].data))
                for loc in locs:
                    xc, yc = deepcopy(loc)
                    circle = Circle((xc, yc), radius=xylim/15., lw=1., edgecolor='red', facecolor='none')
                    ax[k].add_artist(circle)
                ax[k].set_xlim(-xlim)
                ax[k].set_ylim(ylim)
                ax[k].set_xlabel('Arcsec')
                ax[k].set_ylabel('Arcsec')
                ax[k].set_title('$\Delta$WFE = {:.0f} nm'.format(wfe_drift))
                ax[k].tick_params(axis='both', color='white', which='both')
                for l in ax[k].spines.keys():
                    ax[k].spines[l].set_color('white')
                nrc_utils.plotAxes(ax[k], width=1., headwidth=5., alength=0.15, position=(0.95, 0.75), label1='E', label2='N')
            plt.suptitle('{} planets - {} {}'.format(name_sci, obs.filter, obs.mask))
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
            plt.savefig(fdir+'seq_%03.0f_filt_' % i+config.obs['filter'][i][0][j]+'_roll.pdf')
            plt.close()
            
            # Calculate contrast curves.
            curves = []
            for k, wfe_drift in enumerate(np.array(config.obs['wfe'][i])[[0, 2, 1]]):
                obs.wfe_ref_drift = wfe_drift
                curves += [obs.calc_contrast(roll_angle=np.abs(config.obs['pa'][i][0]-config.obs['pa'][i][1]), nsig=nsig)]
            
            # Plot contrast curves.
            f, ax = plt.subplots(1, 2, figsize=(6.4*2, 4.8*1))
            xr = [0., 5.] # arcsec
            yr = [24., 8.] # mag
            plot_contrasts(curves, nsig, np.array(config.obs['wfe'][i])[[0, 2, 1]], obs=obs, xr=xr, yr=yr, ax=ax[0], return_axes=False)
            seps = [np.sqrt(x**2+y**2) for x, y in locs]
            mags = []
            for k in range(len(tags)):
                mags += [np.load(pmdir+tags[k]+'_'+config.obs['filter'][i][0][j]+'.npy')[0]]
            ax[0].plot(seps, mags, marker='o', ls='None', label='Companions ({})'.format(obs.filter), color='black', zorder=10)
            plot_planet_patches(ax[0], obs, age=age_sci, entropy=config.cmp[0]['sent'], av_vals=None)
            ax[0].legend(ncol=2, fontsize=10)
            plot_contrasts_mjup(curves, nsig, np.array(config.obs['wfe'][i])[[0, 2, 1]], obs=obs, age=age_sci, ax=ax[1], twin_ax=False, xr=xr, yr=None, linder_models=False)
            ax[1].set_yscale('log')
            ax[1].set_ylim([0.05, 100.]) # Jupiter masses
            ax[1].legend(loc='upper right', title='COND ({:.0f} Myr)'.format(age_sci), fontsize=10)
            plt.suptitle('{} planets - {} {}'.format(name_sci, obs.filter, obs.mask))
            plt.tight_layout()
            plt.subplots_adjust(top=0.80)
            plt.savefig(fdir+'seq_%03.0f_filt_' % i+config.obs['filter'][i][0][j]+'_contrast.pdf')
            plt.close()
        
        # Activate wavefront drifts.
        obs.wfe_drift = True
        obs.nrc_ref.wfe_drift = True
        
        # Roll 1.
        if (config.obs['patttype'][i][0] == 'NONE'):
            
            # Generate noiseless slope image.
            im_slope = obs.gen_slope_image(PA=config.obs['pa'][i][0],
                                           exclude_disk=True,
                                           exclude_planets=False,
                                           exclude_noise=True,
                                           zfact=2.5,
                                           do_ref=False,
                                           do_roll2=False,
                                           im_star=None,
                                           wfe_drift0=config.obs['wfe'][i][0],
                                           wfe_ref_drift=config.obs['wfe'][i][2],
                                           wfe_roll_drift=config.obs['wfe'][i][1])
            det = obs.Detectors[0]
            im_slope = nrc_utils.sci_to_det(im_slope, det.detid)
            
            # Shift slope image to science reference pixel position.
            if (config.obs['wind_mode'][i][0] == 'WINDOW'):
                dely, delx = config.obs['crpix'][i][0][j]
                if (config.obs['baroff'][i][0][j] is None):
                    # sh = (delx-1.-config.obs['fov_pix'][i][0]/2., dely-1.-config.obs['fov_pix'][i][0]/2.)
                    sh = (delx-config.obs['fov_pix'][i][0]/2., dely-0.5-config.obs['fov_pix'][i][0]/2.)
                else:
                    if ('LONG' in config.obs['detector'][i][0]):
                        # sh = (delx-1.-config.obs['fov_pix'][i][0]/2., dely-1.-config.obs['fov_pix'][i][0]/2.+config.obs['baroff'][i][0][j]/pixscale_LW)
                        sh = (delx-config.obs['fov_pix'][i][0]/2., dely-0.5-config.obs['fov_pix'][i][0]/2.+config.obs['baroff'][i][0][j]/pixscale_LW)
                    else:
                        # sh = (delx-1.-config.obs['fov_pix'][i][0]/2., dely-1.-config.obs['fov_pix'][i][0]/2.+config.obs['baroff'][i][0][j]/pixscale_SW)
                        sh = (delx-config.obs['fov_pix'][i][0]/2., dely-0.5-config.obs['fov_pix'][i][0]/2.+config.obs['baroff'][i][0][j]/pixscale_SW)
                im_slope = shift(im_slope, sh, order=1, mode='constant', cval=0.)
            
            # Convert slope to ramp (the ramp will be saved automatically).
            file_out = odir+'obs_%03.0f_filt_' % config.obs['num'][i][0]+config.obs['filter'][i][0][j]+'_ramp.fits'
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
            raise UserWarning('Does not support dither pattern '+config.obs['patttype'][i][0]+' for roll 1')
        
        # Roll 2.
        if (config.obs['patttype'][i][1] == 'NONE'):
            
            # Generate noiseless slope image.
            im_slope = obs.gen_slope_image(PA=config.obs['pa'][i][1],
                                           exclude_disk=True,
                                           exclude_planets=False,
                                           exclude_noise=True,
                                           zfact=2.5,
                                           do_ref=False,
                                           do_roll2=True,
                                           im_star=None,
                                           wfe_drift0=config.obs['wfe'][i][0],
                                           wfe_ref_drift=config.obs['wfe'][i][2],
                                           wfe_roll_drift=config.obs['wfe'][i][1])
            det = obs.Detectors[0]
            im_slope = nrc_utils.sci_to_det(im_slope, det.detid)
            
            # Shift slope image to science reference pixel position.
            if (config.obs['wind_mode'][i][1] == 'WINDOW'):
                dely, delx = config.obs['crpix'][i][1][j]
                if (config.obs['baroff'][i][1][j] is None):
                    # sh = (delx-1.-config.obs['fov_pix'][i][1]/2., dely-1.-config.obs['fov_pix'][i][1]/2.)
                    sh = (delx-config.obs['fov_pix'][i][1]/2., dely-0.5-config.obs['fov_pix'][i][1]/2.)
                else:
                    if ('LONG' in config.obs['detector'][i][1]):
                        # sh = (delx-1.-config.obs['fov_pix'][i][1]/2., dely-1.-config.obs['fov_pix'][i][1]/2.+config.obs['baroff'][i][1][j]/pixscale_LW)
                        sh = (delx-config.obs['fov_pix'][i][1]/2., dely-0.5-config.obs['fov_pix'][i][1]/2.+config.obs['baroff'][i][1][j]/pixscale_LW)
                    else:
                        # sh = (delx-1.-config.obs['fov_pix'][i][1]/2., dely-1.-config.obs['fov_pix'][i][1]/2.+config.obs['baroff'][i][1][j]/pixscale_SW)
                        sh = (delx-config.obs['fov_pix'][i][1]/2., dely-0.5-config.obs['fov_pix'][i][1]/2.+config.obs['baroff'][i][1][j]/pixscale_SW)
                im_slope = shift(im_slope, sh, order=1, mode='constant', cval=0.)
            
            # Convert slope to ramp (the ramp will be saved automatically).
            file_out = odir+'obs_%03.0f_filt_' % config.obs['num'][i][1]+config.obs['filter'][i][0][j]+'_ramp.fits'
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
            raise UserWarning('Does not support dither pattern '+config.obs['patttype'][i][1]+' for roll 2')
        
        # Reference.
        if (config.obs['patttype'][i][2] == 'NONE'):
            
            # Generate noiseless slope image.
            im_slope = obs.gen_slope_image(PA=config.obs['pa'][i][2],
                                           exclude_disk=True,
                                           exclude_planets=True,
                                           exclude_noise=True,
                                           zfact=2.5,
                                           do_ref=True,
                                           do_roll2=False,
                                           im_star=None,
                                           wfe_drift0=config.obs['wfe'][i][0],
                                           wfe_ref_drift=config.obs['wfe'][i][2],
                                           wfe_roll_drift=config.obs['wfe'][i][1])
            det = obs.Detectors[0]
            im_slope = nrc_utils.sci_to_det(im_slope, det.detid)
            
            # Shift slope image to science reference pixel position.
            if (config.obs['wind_mode'][i][2] == 'WINDOW'):
                dely, delx = config.obs['crpix'][i][2][j]
                if (config.obs['baroff'][i][2][j] is None):
                    # sh = (delx-1.-config.obs['fov_pix'][i][2]/2., dely-1.-config.obs['fov_pix'][i][2]/2.)
                    sh = (delx-config.obs['fov_pix'][i][2]/2., dely-0.5-config.obs['fov_pix'][i][2]/2.)
                else:
                    if ('LONG' in config.obs['detector'][i][2]):
                        # sh = (delx-1.-config.obs['fov_pix'][i][2]/2., dely-1.-config.obs['fov_pix'][i][2]/2.+config.obs['baroff'][i][2][j]/pixscale_LW)
                        sh = (delx-config.obs['fov_pix'][i][2]/2., dely-0.5-config.obs['fov_pix'][i][2]/2.+config.obs['baroff'][i][2][j]/pixscale_LW)
                    else:
                        # sh = (delx-1.-config.obs['fov_pix'][i][2]/2., dely-1.-config.obs['fov_pix'][i][2]/2.+config.obs['baroff'][i][2][j]/pixscale_SW)
                        sh = (delx-config.obs['fov_pix'][i][2]/2., dely-0.5-config.obs['fov_pix'][i][2]/2.+config.obs['baroff'][i][2][j]/pixscale_SW)
                im_slope = shift(im_slope, sh, order=1, mode='constant', cval=0.)
            
            # Convert slope to ramp (the ramp will be saved automatically).
            file_out = odir+'obs_%03.0f_filt_' % config.obs['num'][i][2]+config.obs['filter'][i][0][j]+'_ramp.fits'
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
            if (config.obs['patttype'][i][2] == '5-POINT-BOX'):
                dithers = [(0., 0.), (15., 15.), (-15., 15.), (-15., -15.), (15., -15.)] # mas
            elif (config.obs['patttype'][i][2] == '5-POINT-DIAMOND'):
                dithers = [(0., 0.), (0., 20.), (0., -20.), (20., 0.), (-20., 0.)] # mas
            elif (config.obs['patttype'][i][2] == '9-POINT-CIRCLE'):
                dithers = [(0., 0.), (0., 20.), (-15., 15.), (-20., 0.), (-15., -15.), (0., -20.), (15., -15.), (20., 0.), (15., 15.)] # mas
            elif (config.obs['patttype'][i][2] == '3-POINT-BAR'):
                dithers = [(0., 0.), (0., 15.), (0., -15.)] # mas
            elif (config.obs['patttype'][i][2] == '5-POINT-BAR'):
                dithers = [(0., 0.), (0., 20.), (0., 10.), (0., -10.), (0., -20.)] # mas
            else:
                raise UserWarning('Does not support dither pattern '+config.obs['patttype'][i][2]+' for reference')
            for k in range(len(dithers)):
                
                # Generate noiseless slope image.
                im_slope = obs.gen_slope_image(PA=config.obs['pa'][i][2],
                                               exclude_disk=True,
                                               exclude_planets=True,
                                               exclude_noise=True,
                                               zfact=2.5,
                                               do_ref=True,
                                               do_roll2=False,
                                               im_star=None,
                                               wfe_drift0=config.obs['wfe'][i][0],
                                               wfe_ref_drift=config.obs['wfe'][i][2],
                                               wfe_roll_drift=config.obs['wfe'][i][1],
                                               dither=dithers[k])
                det = obs.nrc_ref.Detectors[0]
                im_slope = nrc_utils.sci_to_det(im_slope, det.detid)
                
                # Shift slope image to science reference pixel position.
                if (config.obs['wind_mode'][i][2] == 'WINDOW'):
                    dely, delx = config.obs['crpix'][i][2][j]
                    if (config.obs['baroff'][i][2][j] is None):
                        # sh = (delx-1.-config.obs['fov_pix'][i][2]/2., dely-1.-config.obs['fov_pix'][i][2]/2.)
                        sh = (delx-config.obs['fov_pix'][i][2]/2., dely-0.5-config.obs['fov_pix'][i][2]/2.)
                    else:
                        if ('LONG' in config.obs['detector'][i][2]):
                            # sh = (delx-1.-config.obs['fov_pix'][i][2]/2., dely-1.-config.obs['fov_pix'][i][2]/2.+config.obs['baroff'][i][2][j]/pixscale_LW)
                            sh = (delx-config.obs['fov_pix'][i][2]/2., dely-0.5-config.obs['fov_pix'][i][2]/2.+config.obs['baroff'][i][2][j]/pixscale_LW)
                        else:
                            # sh = (delx-1.-config.obs['fov_pix'][i][2]/2., dely-1.-config.obs['fov_pix'][i][2]/2.+config.obs['baroff'][i][2][j]/pixscale_SW)
                            sh = (delx-config.obs['fov_pix'][i][2]/2., dely-0.5-config.obs['fov_pix'][i][2]/2.+config.obs['baroff'][i][2][j]/pixscale_SW)
                    im_slope = shift(im_slope, sh, order=1, mode='constant', cval=0.)
                
                # Convert slope to ramp (the ramp will be saved automatically).
                file_out = odir+'obs_%03.0f_filt_' % config.obs['num'][i][2]+config.obs['filter'][i][0][j]+'_dpos_%03.0f' % k+'_ramp.fits'
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
