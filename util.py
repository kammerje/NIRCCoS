from __future__ import division

import matplotlib
matplotlib.rcParams.update({'font.size': 14})


# =============================================================================
# IMPORTS
# =============================================================================

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import yaml

from xml.dom import minidom


# =============================================================================
# MAIN
# =============================================================================

# Bar mask offsets for module A in arcsec from WebbPSF.
offset_swb = {'F182M': -1.856,
              'F187N': -1.571,
              'F210M': -0.071,
              'F212N': 0.143,
              'F200W': 0.232,
              'narrow': -8.00}
offset_lwb = {'F250M': 6.846,
              'F300M': 5.249,
              'F277W': 5.078,
              'F335M': 4.075,
              'F360M': 3.195,
              'F356W': 2.455,
              'F410M': 1.663,
              'F430M': 1.043,
              'F460M': -0.098,
              'F480M': -0.619,
              'F444W': -0.768,
              'narrow': 8.0}

# Keyword dictionary for pyNRC.
pupil_pynrc = {'MASK210R': 'CIRCLYOT',
               'MASK335R': 'CIRCLYOT',
               'MASK430R': 'CIRCLYOT',
               'MASKSWB': 'WEDGELYOT',
               'MASKLWB': 'WEDGELYOT'}
fov_pix_pynrc = {'SUB320': 320,
                 'SUB640': 640,
                 'FULL': 2048}
wind_mode_pynrc = {'SUB320': 'WINDOW',
                   'SUB640': 'WINDOW',
                   'FULL': 'FULL'}

# Keyword dictionary for MIRAGE and JWST Pipeline.
detector_stsci = {'MASK210R_SUB640': 'NRCA2',
                  'MASK335R_SUB320': 'NRCALONG',
                  'MASK430R_SUB320': 'NRCALONG',
                  'MASK430R_FULL': 'NRCALONG',
                  'MASKSWB_SUB640': 'NRCA4',
                  'MASKLWB_SUB320': 'NRCALONG'}
module_stsci = {'MASK210R_SUB640': 'A',
                'MASK335R_SUB320': 'A',
                'MASK430R_SUB320': 'A',
                'MASK430R_FULL': 'A',
                'MASKSWB_SUB640': 'A',
                'MASKLWB_SUB320': 'A'}
pupil_stsci = {'MASK210R_SUB640': 'MASKRND',
               'MASK335R_SUB320': 'MASKRND',
               'MASK430R_SUB320': 'MASKRND',
               'MASK430R_FULL': 'MASKRND',
               'MASKSWB_SUB640': 'MASKBAR',
               'MASKLWB_SUB320': 'MASKBAR'}
coronmsk_stsci = {'MASK210R_SUB640': 'MASKA210R',
                  'MASK335R_SUB320': 'MASKA335R',
                  'MASK430R_SUB320': 'MASKA430R',
                  'MASK430R_FULL': 'MASKA430R',
                  'MASKSWB_SUB640': 'MASKASWB',
                  'MASKLWB_SUB320': 'MASKALWB'}
subarray_stsci = {'MASK210R_SUB640': 'SUB640A210R',
                  'MASK335R_SUB320': 'SUB320A335R',
                  'MASK430R_SUB320': 'SUB320A430R',
                  'MASK430R_FULL': 'FULL',
                  'MASKSWB_SUB640': 'SUB640ASWB',
                  'MASKLWB_SUB320': 'SUB320ALWB'}
apername_stsci = {'MASK210R_SUB640': 'NRCA2_MASK210R',
                  'MASK335R_SUB320': 'NRCA5_MASK335R',
                  'MASK430R_SUB320': 'NRCA5_MASK430R',
                  'MASK430R_FULL': 'NRCA5_FULL_MASK430R',
                  'MASKSWB_SUB640_F182M': 'NRCA4_MASKSWB_F182M',
                  'MASKSWB_SUB640_F187N': 'NRCA4_MASKSWB_F187N',
                  'MASKSWB_SUB640_F210M': 'NRCA4_MASKSWB_F210M',
                  'MASKSWB_SUB640_F212N': 'NRCA4_MASKSWB_F212N',
                  'MASKSWB_SUB640_F200W': 'NRCA4_MASKSWB_F200W',
                  'MASKSWB_SUB640_NARROW': 'NRCA4_MASKSWB_NARROW',
                  'MASKLWB_SUB320_F250M': 'NRCA5_MASKLWB_F250M',
                  'MASKLWB_SUB320_F300M': 'NRCA5_MASKLWB_F300M',
                  'MASKLWB_SUB320_F277W': 'NRCA5_MASKLWB_F277W',
                  'MASKLWB_SUB320_F335M': 'NRCA5_MASKLWB_F335M',
                  'MASKLWB_SUB320_F360M': 'NRCA5_MASKLWB_F360M',
                  'MASKLWB_SUB320_F356W': 'NRCA5_MASKLWB_F356W',
                  'MASKLWB_SUB320_F410M': 'NRCA5_MASKLWB_F410M',
                  'MASKLWB_SUB320_F430M': 'NRCA5_MASKLWB_F430M',
                  'MASKLWB_SUB320_F460M': 'NRCA5_MASKLWB_F460M',
                  'MASKLWB_SUB320_F480M': 'NRCA5_MASKLWB_F480M',
                  'MASKLWB_SUB320_F444W': 'NRCA5_MASKLWB_F444W',
                  'MASKLWB_SUB320_NARROW': 'NRCA5_MASKLWB_NARROW'}
crpix_stsci = {'MASK210R_SUB640': (321.0, 336.0),
               'MASK335R_SUB320': (160.0, 172.5),
               'MASK430R_SUB320': (160.0, 172.5),
               'MASK430R_FULL': (973.5, 1673.5),
               'MASKSWB_SUB640_F182M': (263.6, 340.5),
               'MASKSWB_SUB640_F187N': (270.1, 340.5),
               'MASKSWB_SUB640_F210M': (319.4, 340.5),
               'MASKSWB_SUB640_F212N': (325.2, 340.5),
               'MASKSWB_SUB640_F200W': (326.9, 340.5),
               'MASKSWB_SUB640_NARROW': (57.5, 340.5),
               'MASKLWB_SUB320_F250M': (265.2, 175.5),
               'MASKLWB_SUB320_F300M': (240.8, 175.5),
               'MASKLWB_SUB320_F277W': (238.8, 175.5),
               'MASKLWB_SUB320_F335M': (222.1, 175.5),
               'MASKLWB_SUB320_F360M': (209.0, 175.5),
               'MASKLWB_SUB320_F356W': (197.3, 175.5),
               'MASKLWB_SUB320_F410M': (186.0, 175.5),
               'MASKLWB_SUB320_F430M': (176.0, 175.5),
               'MASKLWB_SUB320_F460M': (158.5, 175.5),
               'MASKLWB_SUB320_F480M': (146.3, 175.5),
               'MASKLWB_SUB320_F444W': (148.4, 175.5),
               'MASKLWB_SUB320_NARROW': (293.0, 175.5)}

class config():
    
    def __init__(self,
                 path_config='config.yaml'):
        
        config = yaml.load(open(path_config), Loader=yaml.BaseLoader)
        
        self.paths = {}
        keys = ['wdir', 'pmdir', 'pynrc_data_dir', 'pynrc_figs_dir', 'mirage_refs_dir', 'mirage_cats_dir', 'mirage_logs_dir', 'mirage_data_dir', 'jwst_s1s2_data_dir', 'jwst_s3_data_dir', 'species_dir', 'whereistheplanet_dir', 'webbpsf_ext_dir', 'pynrc_dir']
        for i in range(len(keys)):
            try:
                self.paths[keys[i]] = str(config['paths'][keys[i]])
            except:
                raise UserWarning('Config file needs paths:'+keys[i]+' entry')
        keys = ['xml_path', 'pointing_path']
        for i in range(len(keys)):
            try:
                self.paths[keys[i]] = str(config['apt'][keys[i]])
            except:
                raise UserWarning('Config file needs apt:'+keys[i]+' entry')
        
        self.obs = {}
        keys = ['date', 'oversample']
        for i in range(len(keys)):
            try:
                self.obs[keys[i]] = str(config['observation'][keys[i]])
            except:
                raise UserWarning('Config file needs observation:'+keys[i]+' entry')
        self.obs['ind'] = []
        try:
            nseq = len(config['observation']['sequences'])
        except:
            raise UserWarning('Config file needs observation:sequences entry')
        for key in config['observation']['sequences'].keys():
            nobs = len(str(config['observation']['sequences'][key]).split(','))
            if (nobs != 3):
                raise UserWarning('Config file needs three comma-separated numbers per observation:sequences entry')
            else:
                self.obs['ind'] += [str(config['observation']['sequences'][key]).split(',')]
        self.obs['tag'] = [['roll1', 'roll2', 'ref']]*len(self.obs['ind'])
        pa = []
        keys = ['pa_roll1', 'pa_roll2', 'pa_ref']
        keys_old = ['pa1', 'pa2', 'pa1']
        for i in range(len(keys)):
            try:
                pa += [float(config['observation'][keys[i]])]
            except:
                try:
                    pa += [float(config['observation'][keys_old[i]])]
                except:
                    raise UserWarning('Config file needs observation:'+keys[i]+' entry')
        self.obs['pa'] = [pa]*len(self.obs['ind'])
        wfe = []
        keys = ['wfe_drift_roll1', 'wfe_drift_roll2', 'wfe_drift_ref']
        keys_old = ['wfe_drift0', 'wfe_roll_drift', 'wfe_ref_drift']
        for i in range(len(keys)):
            try:
                wfe += [float(config['observation'][keys[i]])]
            except:
                try:
                    wfe += [float(config['observation'][keys_old[i]])]
                except:
                    raise UserWarning('Config file needs observation:'+keys[i]+' entry')
        self.obs['wfe'] = [wfe]*len(self.obs['ind'])
        
        self.src = []
        sci = {}
        keys = ['name', 'dist', 'age', 'sptype', 'teff', 'feh', 'logg', 'kmag']
        for i in range(len(keys)):
            try:
                if (keys[i] in ['name', 'sptype']):
                    sci[keys[i]] = str(config['sources']['sci'][keys[i]])
                else:
                    sci[keys[i]] = float(config['sources']['sci'][keys[i]])
            except:
                raise UserWarning('Config file needs sources:sci:'+keys[i]+' entry')
        self.src += [sci]
        ref = {}
        keys = ['name', 'sptype', 'teff', 'feh', 'logg', 'kmag']
        for i in range(len(keys)):
            try:
                if (keys[i] in ['name', 'sptype']):
                    ref[keys[i]] = str(config['sources']['ref'][keys[i]])
                else:
                    ref[keys[i]] = float(config['sources']['ref'][keys[i]])
            except:
                raise UserWarning('Config file needs sources:ref:'+keys[i]+' entry')
        self.src += [ref]
        
        self.cmp = []
        try:
            ncmp = len(config['companions'])
        except:
            raise UserWarning('Config file needs companions entry')
        keys = ['name_witp', 'name_spec', 'ra_off', 'de_off', 'mass', 'sent']
        for key in config['companions'].keys():
            temp = {}
            for i in range(len(keys)):
                try:
                    if (keys[i] in ['name_witp', 'name_spec']):
                        temp[keys[i]] = str(config['companions'][key][keys[i]])
                    elif (keys[i] in ['ra_off', 'de_off']):
                        try:
                            temp[keys[i]] = float(config['companions'][key][keys[i]])
                        except:
                            temp[keys[i]] = None
                    else:
                        temp[keys[i]] = float(config['companions'][key][keys[i]])
                except:
                    raise UserWarning('Config file needs companions:'+key+':'+keys[i]+' entry')
            self.cmp += [temp]
        
        self.pip = {}
        try:
            self.pip['model_spec'] = str(config['pipeline']['model_spec'])
        except:
            raise UserWarning('Config file needs pipeline:model_spec entry')
        try:
            self.pip['teff_range'] = [float(val) for val in str(config['pipeline']['teff_range']).split(',')]
        except:
            raise UserWarning('Config file needs pipeline:teff_range entry')
        try:
            self.pip['make_plots'] = str(config['pipeline']['make_plots']) == 'True'
        except:
            raise UserWarning('Config file needs pipeline:make_plots entry')
        
        self.read_xml()
        
        self.obs['baroff'] = []
        for i in range(len(self.obs['filter'])):
            baroff = []
            for j in range(len(self.obs['filter'][i])):
                temp = []
                for k in range(len(self.obs['filter'][i][j])):
                    if ('SWB' in self.obs['mask'][i][j]):
                        if ('NARROW' in self.obs['sreq'][i][j]):
                            temp += [-offset_swb['narrow']]
                        else:
                            temp += [-offset_swb[self.obs['filter'][i][j][k]]]
                    elif ('LWB' in self.obs['mask'][i][j]):
                        if ('NARROW' in self.obs['sreq'][i][j]):
                            temp += [-offset_lwb['narrow']]
                        else:
                            temp += [-offset_lwb[self.obs['filter'][i][j][k]]]
                    else:
                        temp += [None]
                baroff += [temp]
            self.obs['baroff'] += [baroff]
        
        self.match_pynrc()
        self.match_stsci()
        
        return None
    
    def read_xml(self):
        
        xml_file = minidom.parse(self.paths['wdir']+self.paths['xml_path'])
        obs_all = xml_file.getElementsByTagName('Observation')
        
        self.obs['num'] = []
        self.obs['mask'] = []
        self.obs['conf'] = []
        self.obs['subsize'] = []
        self.obs['patttype'] = []
        self.obs['filter'] = []
        self.obs['readpatt'] = []
        self.obs['ngroups'] = []
        self.obs['nints'] = []
        self.obs['sreq'] = []
        for i in range(len(self.obs['ind'])):
            print('Reading sequence %.0f from xml file' % (i+1))
            num = []
            mask = []
            conf = []
            subsize = []
            patttype = []
            filter = []
            readpatt = []
            ngroups = []
            nints = []
            sreq = []
            for j in range(len(self.obs['ind'][i])):
                print('   Reading observation '+self.obs['ind'][i][j]+' from xml file ('+self.obs['tag'][i][j]+')')
                try:
                    obs = obs_all[int(self.obs['ind'][i][j])-1]
                except:
                    raise UserWarning('%.0f is an invalid observation number' % (int(self.obs['id'][i][j])-1))
                num += [int(obs.getElementsByTagName('Number')[0].childNodes[0].data)]
                print('   APT observation number '+str(num[-1]))
                mask += [obs.getElementsByTagName('ncc:CoronMask')[0].childNodes[0].data]
                conf += [obs.getElementsByTagName('ncc:OptionalConfirmationImage')[0].childNodes[0].data == 'true']
                subsize += [obs.getElementsByTagName('ncc:Subarray')[0].childNodes[0].data]
                patttype += [obs.getElementsByTagName('ncc:DitherPattern')[0].childNodes[0].data]
                temp = [item.childNodes[0].data for item in obs.getElementsByTagName('ncc:Filter')]
                filter += [temp]
                temp = [item.childNodes[0].data for item in obs.getElementsByTagName('ncc:ReadoutPattern')]
                readpatt += [temp]
                temp = [item.childNodes[0].data for item in obs.getElementsByTagName('ncc:Groups')]
                ngroups += [temp]
                temp = [item.childNodes[0].data for item in obs.getElementsByTagName('ncc:Integrations')]
                nints += [temp]
                try:
                    sreq += [obs.getElementsByTagName('FiducialPointOverride')[0].childNodes[0].data]
                except:
                    sreq += [None]
            self.obs['num'] += [num]
            self.obs['mask'] += [mask]
            self.obs['conf'] += [conf]
            self.obs['subsize'] += [subsize]
            self.obs['patttype'] += [patttype]
            self.obs['filter'] += [filter]
            self.obs['readpatt'] += [readpatt]
            self.obs['ngroups'] += [ngroups]
            self.obs['nints'] += [nints]
            self.obs['sreq'] += [sreq]
        
        src_all = xml_file.getElementsByTagName('Target')
        for i in range(len(src_all)):
            src = src_all[i]
            name = src.getElementsByTagName('TargetArchiveName')[0].childNodes[0].data
            for j in range(len(self.src)):
                if (name == self.src[j]['name']):
                    break
            if (j >= len(self.src)):
                raise UserWarning('Source names in config file need to match target archive names in APT file')
            self.src[j]['icrs'] = str(src.getElementsByTagName('EquatorialCoordinates')[0].getAttribute('Value'))
        
        return None
    
    def match_pynrc(self):
        
        self.obs['pupil_pynrc'] = []
        self.obs['fov_pix'] = []
        self.obs['wind_mode'] = []
        for i in range(len(self.obs['mask'])):
            pupil = []
            fov_pix = []
            wind_mode = []
            for j in range(len(self.obs['mask'][i])):
                pupil += [pupil_pynrc[self.obs['mask'][i][j]]]
                fov_pix += [fov_pix_pynrc[self.obs['subsize'][i][j]]]
                wind_mode += [wind_mode_pynrc[self.obs['subsize'][i][j]]]
            self.obs['pupil_pynrc'] += [pupil]
            self.obs['fov_pix'] += [fov_pix]
            self.obs['wind_mode'] += [wind_mode]
        
        return None
    
    def match_stsci(self):
        
        self.obs['detector'] = []
        self.obs['module'] = []
        self.obs['pupil'] = []
        self.obs['coronmsk'] = []
        self.obs['subarray'] = []
        self.obs['apername'] = []
        self.obs['crpix'] = []
        for i in range(len(self.obs['filter'])):
            detector = []
            module = []
            pupil = []
            coronmsk = []
            subarray = []
            apername = []
            crpix = []
            for j in range(len(self.obs['filter'][i])):
                tag = self.obs['mask'][i][j]+'_'+self.obs['subsize'][i][j]
                detector += [detector_stsci[tag]]
                module += [module_stsci[tag]]
                pupil += [pupil_stsci[tag]]
                coronmsk += [coronmsk_stsci[tag]]
                subarray += [subarray_stsci[tag]]
                apername_temp = []
                crpix_temp = []
                for k in range(len(self.obs['filter'][i][j])):    
                    if (self.obs['mask'][i][j] in ['MASKSWB', 'MASKLWB']):
                        if ('NARROW' in self.obs['sreq'][i][j]):
                            tag = self.obs['mask'][i][j]+'_'+self.obs['subsize'][i][j]+'_NARROW'
                        else:
                            tag = self.obs['mask'][i][j]+'_'+self.obs['subsize'][i][j]+'_'+self.obs['filter'][i][j][k]
                    apername_temp += [apername_stsci[tag]]
                    crpix_temp += [crpix_stsci[tag]]
                apername += [apername_temp]
                crpix += [crpix_temp]
            self.obs['detector'] += [detector]
            self.obs['module'] += [module]
            self.obs['pupil'] += [pupil]
            self.obs['coronmsk'] += [coronmsk]
            self.obs['subarray'] += [subarray]
            self.obs['apername'] += [apername]
            self.obs['crpix'] += [crpix]
        
        return None
