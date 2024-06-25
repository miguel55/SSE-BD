# -*- coding: utf-8 -*-
"""
Created on Mon May 10 10:17:09 2021

@author: mmolina
"""

import os
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
cfg = __C

#########################
#  Experiment options   #
#########################

__C.arch="lstm"
__C.result_dir = os.path.abspath(os.path.join("results"))
__C.train_dir = os.path.abspath(os.path.join("results","train"))
__C.data_dir = os.path.abspath(os.path.join("data"))
__C.group_ids=['WT','antiPlt','FGR-KO','FGR-INH']

# Experiment parameters
__C.lr=2.5e-6
__C.batch_size=40
__C.beta=8e-1
__C.beta_traj=1e-1
__C.pnum_traj=11
__C.pnum=30
__C.outputdims=[2,4,8,16,32,64,128,256]
__C.index_insert=4
__C.n_features=21
__C.train_groups=[1,2] # groups_for_training
__C.test_groups=[3,4] # groups_for_test
__C.K_selected=6
__C.PTH=3
__C.SEED = 0

#########################
#  Data visualization   #
#########################
__C.REMAP=True # To obtain the same results of the paper with new versions of Python
__C.TITLE_SIZE=24
__C.POINT_SIZE=60
__C.SUBSAMPLE_PERC=100
__C.groups_rep=[1,2]
__C.ALPHA=1
__C.alg='tsne' # 'tsne' or 'umap'
if (__C.alg=='tsne'):
    # param1: perplexity and param2: exaggeration
    __C.param1=np.linspace(1,197,50)
    __C.param2=np.linspace(1,57,29)
else:
    # param1: number of neighbors and param2: minimum distance
    __C.param1=np.linspace(3,199,50)
    __C.param2=np.linspace(0.1,0.9,9)

__C.categories_orig = ['volume','surface area','height','max-length','height-length ratio','sphericity','prol ellipticity',
              'obl ellipticity','princ axis length (1)','princ axis length (2)','princ axis length (3)',
              'extent', 'solidity', 'eq diameter', 'X-orientation','Y-orientation','Z-orientation','polar radius',
              'polar angle','center distance to vessel','min distance to vessel','volume']

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        if type(b[k]) is not type(v):
            raise ValueError(('Type mismatch ({} vs. {}) '
                              'for config key: {}').format(type(b[k]),
                                                           type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
