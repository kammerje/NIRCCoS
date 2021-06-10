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

from xml.dom import minidom


# =============================================================================
# MAIN
# =============================================================================

def read_from_xml(xml_path,
                  seq):
    
    print('Reading sequence '+str(seq))
    
    xml_file = minidom.parse(xml_path)
    
    obss = xml_file.getElementsByTagName('Observation')
    masks = []
    astro = []
    subsizes = []
    dithers = []
    filts = []
    read_modes = []
    ngroups = []
    nints = []
    for i in range(len(seq)):
        obs = obss[int(seq[i])-1]
        num = obs.getElementsByTagName('Number')[0].childNodes[0].data
        
        print('   Observation '+num)
        
        masks += [obs.getElementsByTagName('ncc:CoronMask')[0].childNodes[0].data]
        astro += [obs.getElementsByTagName('ncc:OptionalConfirmationImage')[0].childNodes[0].data]
        subsizes += [obs.getElementsByTagName('ncc:Subarray')[0].childNodes[0].data]
        dithers += [obs.getElementsByTagName('ncc:DitherPattern')[0].childNodes[0].data]
        temp = [item.childNodes[0].data for item in obs.getElementsByTagName('ncc:Filter')]
        filts += [temp]
        temp = [item.childNodes[0].data for item in obs.getElementsByTagName('ncc:ReadoutPattern')]
        read_modes += [temp]
        temp = [item.childNodes[0].data for item in obs.getElementsByTagName('ncc:Groups')]
        ngroups += [temp]
        temp = [item.childNodes[0].data for item in obs.getElementsByTagName('ncc:Integrations')]
        nints += [temp]
    if (len(np.unique(masks)) != 1):
        raise UserWarning('Sequence '+str(seq)+' contains observations with multiple masks')
    if (len(np.unique(subsizes)) != 1):
        raise UserWarning('Sequence '+str(seq)+' contains observations with multiple subarrays')
    if ((len(np.unique(dithers[:2])) != 1) or (np.unique(dithers[:2])[0] != 'NONE')):
        raise UserWarning('Sequence '+str(seq)+' contains sci observations with a dither pattern that is not NONE and thus not supported yet')
    if (not all([set(filts[0]) == set(filts[i]) for i in range(1, len(filts))])):
        raise UserWarning('Sequence '+str(seq)+' contains observations with different sets of filters')
    if (not all([set(read_modes[0]) == set(read_modes[i]) for i in range(1, len(read_modes)-1)])):
        raise UserWarning('Sequence '+str(seq)+' contains observations with different sets of readout modes')
    if (not all([set(ngroups[0]) == set(ngroups[i]) for i in range(1, len(ngroups)-1)])):
        raise UserWarning('Sequence '+str(seq)+' contains observations with different numbers of groups')
    if (not all([set(nints[0]) == set(nints[i]) for i in range(1, len(nints)-1)])):
        raise UserWarning('Sequence '+str(seq)+' contains observations with different numbers of integrations')
    
    sets = {}
    sets['filts'] = filts[0]
    sets['masks'] = [masks[0]]*len(filts[0])
    if (masks[0] in ['MASK210R', 'MASK335R', 'MASK430R']):
        sets['pups'] = ['CIRCLYOT']*len(filts[0])
        sets['boff'] = [None]*len(filts[0])
    elif (masks[0] in ['MASKSWB', 'MASKLWB']):
        sets['pups'] = ['WEDGELYOT']*len(filts[0])
        sets['boff'] = [float(config['observation']['bar_offset'])]*len(filts[0])
    else:
        raise UserWarning(masks[0]+' is an unknown mask')
    if (subsizes[0] == 'SUB320'):
        sets['wind'] = ['WINDOW']*len(filts[0])
        sets['subs'] = [320]*len(filts[0])
        sets['fovp'] = [320]*len(filts[0])
    elif (subsizes[0] == 'FULL'):
        sets['wind'] = ['FULL']*len(filts[0])
        sets['subs'] = [2048]*len(filts[0])
        sets['fovp'] = [2048]*len(filts[0])
    else:
        raise UserWarning(masks[0]+' is an unknown subarray')
    sets['nums'] = [int(seq[0]), int(seq[1]), int(seq[2])]
    sets['read_sci'] = read_modes[0]
    sets['ngrp_sci'] = ngroups[0]
    sets['nint_sci'] = nints[0]
    sets['read_ref'] = read_modes[2]
    sets['ngrp_ref'] = ngroups[2]
    sets['nint_ref'] = nints[2]
    sets['astro'] = astro
    sets['dithers'] = dithers
    
    return sets
