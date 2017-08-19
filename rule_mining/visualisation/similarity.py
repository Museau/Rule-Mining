# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Author:
# - Margaux Luck
#   <margaux.luck@institut-hypercube.org>, <margaux.luck@gmail.com>
# Language: python2.7


import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import groupby, chain
from operator import itemgetter


def get_union_of_rules_by_var_modality(rules):
    '''
    Get the union of the rules per feature modalities.

    Parameters:
    - rules, np.array. Array of rules.

    Output:
    - dict_union_rules_inter, dictionary of the rules per features modalities.
    '''

    dict_union_rules_inter = defaultdict(list)

    for g in groupby(rules, key=itemgetter(0, 3)):
        for r in g[1]:
            for idx, f in enumerate(g[0][0]):
                if r[7][idx] == 'c':
                    r_ = range(r[1][idx][0], r[1][idx][1]+1)
                elif r[7][idx] == 'd':
                    r_ = r[1][idx]
                dict_union_rules_inter[(f, g[0][1])].append(r_)

    for k, v in dict_union_rules_inter.items():
        dict_union_rules_inter[k] = list(set(chain(*v)))

    return dict_union_rules_inter


def jaccard_index_local_union_var_commun_matrix(rules, n_model):
    '''
    Matrix representing the Jaccard index for local var in commun between
    models 2 by 2. The Jaccard index is the non strict version that consider
    overlap between the rules if they are in the same feature.
    Only the superior part of the matrix is completed.

    Parameters:
    - rules, list of np.array of rules. shape = len(n_model)
    - n_model, nomber of models to compare

    Output:
    - similarity, pd.dataFrame. Shape (n_model, n_model)
      Matrix representing the Jaccard index for global var in commun between
      models 2 by 2.
     - list_similarity, list. len=n_model. list of the Jaccard index for global
      var in commun between models 2 by 2.
    '''

    list_similarity = []
    a = np.zeros(shape=(n_model, n_model))

    for i in range(0, n_model):

        for j in range(i, n_model):

            # dict union rules inter i
            d1 = get_union_of_rules_by_var_modality(rules[i])
            # dict union rules inter j
            d2 = get_union_of_rules_by_var_modality(rules[j])

            inter = {}

            for k, v in d1.iteritems():
                set_d1_d2 = list(set(d1[k]) & set(d2[k]))
                if k in d2.keys() and set_d1_d2:
                    inter[k] = sorted(set_d1_d2)

            inter_size = {k: len(v) for k, v in inter.iteritems()}
            inter_size = sum(inter_size.itervalues())

            union = defaultdict(list)

            # you can list as many input dicts as you want here
            for dico in (d1, d2):
                for k, v in dico.iteritems():
                    union[k].append(v)

            union = {k: sorted(list(set(sum(v, []))))
                     for k, v in union.iteritems()}
            union_size = {k: len(v) for k, v in union.iteritems()}
            union_size = sum(union_size.itervalues())

            jaccard_index = float(inter_size)/float(union_size)
            a[i, j] = jaccard_index
            list_similarity.append(jaccard_index)

    similarity = pd.DataFrame(
        a, index=range(0, n_model), columns=range(0, n_model))

    return similarity, list_similarity
