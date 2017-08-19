# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Author:
# - Margaux Luck
#   <margaux.luck@institut-hypercube.org>, <margaux.luck@gmail.com>
# Language: python2.7


import numpy as np
from rule_mining.visualisation.similarity import (
    jaccard_index_local_union_var_commun_matrix)

from itertools import product


def main(file_name, experiment):
    '''
    Print the model complexity and stability for the non L1 penalized models.

    Parameters:
    - file_name, string. The name of the dataset.
    - experiment, string. The name of the experiment.
    '''

    print 'File name: ', file_name
    print 'Experiment: ', experiment

    path = experiment
    path_rules = 'results/' + file_name + '/' + path + '/final_rules/'

    # Load the rules datasets
    if path.startswith('test_ORM1D'):
        rules = np.load(path_rules + 'final_rules_ORM1D.npz')

    elif path.startswith('test_decision_tree'):
        rules = np.load(path_rules + 'final_rules_decision_tree.npz')

    elif path.startswith('test_association_rules'):
        rules = np.load(path_rules + 'final_rules_association_rules.npz')

    rules = rules['arr_0'].item()

    keys = ['t'+str(i) for i in xrange(0, 5)]

    # Models complexity
    n_rules = []
    for k in keys:
        n_rules.append(rules[k].shape[0])

    print 'Model complexity'
    print 'Median: ', np.median(n_rules)

    var_ = []
    rules_ = []

    for k in keys:
        var = rules[k][:, 0]
        var = list(var)
        var = sum(var, [])
        var_.append(var)
        rules_.append(rules[k])

    n_model = len(var_)

    similarity_local_, \
        list_similarity_local_ = jaccard_index_local_union_var_commun_matrix(
            rules_, n_model)
    print 'Jaccard Idx loc lev with union rules, models compared two by two: '
    print 'Mean: ', np.mean(list_similarity_local_)
    print 'Std: ', np.std(list_similarity_local_)


if __name__ == '__main__':
    # Dataset with continuous features
    file_names = ['wdbc', 'wine', 'iris']
    experiments = [
        'test_ORM1D_L2LR_deltaTTT',
        'test_decision_tree_L2LR',
        'test_association_rules_L2LR',
    ]

    for exp in list(product(*[file_names, experiments])):
        file_name = exp[0]
        experiment = exp[1]
        main(file_name=file_name, experiment=experiment)

    file_names = ['balance_scale']
    experiments = [
        'test_ORM1D_L2LR',
        'test_decision_tree_L2LR_du',
        'test_association_rules_L2LR',
    ]

    for exp in list(product(*[file_names, experiments])):
        file_name = exp[0]
        experiment = exp[1]
        main(file_name=file_name, experiment=experiment)

    file_names = [
        'heart_disease', 'synthetic', 'synthetic_noisy']
    experiments = [
        'test_ORM1D_L2LR_deltaTTT_d',
        'test_decision_tree_L2LR_du',
        'test_association_rules_L2LR',
    ]

    for exp in list(product(*[file_names, experiments])):
        file_name = exp[0]
        experiment = exp[1]
        main(file_name=file_name, experiment=experiment)
