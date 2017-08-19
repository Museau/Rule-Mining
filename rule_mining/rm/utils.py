# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Author:
# - Margaux Luck
#   <margaux.luck@institut-hypercube.org>, <margaux.luck@gmail.com>
# Language: python2.7


import numpy as np


def key_with_max_val(d):
    '''
    From:
    http://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
    Find the key with the highest value in a dictionnary.

    Parameters:
    - d, dict. The values are float.

    Output:
    - key corresponding to the largest value
    '''
    v = list(d.values())
    k = list(d.keys())
    return k[v.index(np.max(v))]
