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
    Print the model complexity and stability for the L1 penalized models.

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

    # Load the scores, params and coefs
    path_scores = 'results/' + file_name + '/' + path + '/scores/'

    if path.startswith('test_ORM1D'):
        scores = np.load(path_scores + 'scores_ORM1D.npz')

    elif path.startswith('test_decision_tree'):
        scores = np.load(path_scores + 'scores_decision_tree.npz')

    elif path.startswith('test_association_rules'):
        scores = np.load(path_scores + 'scores_association_rules.npz')

    scores = scores['arr_0'].item()

    print scores.keys()
    coefs = scores['coef_']
    idx_non_zero_coef = []

    for i in range(0, len(coefs)):
        coef = coefs[i]
        coef = np.absolute(coef)
        coef = coef/np.sum(coef, axis=1).reshape(coef.shape[0], 1)*100
        coef = np.mean(coef, axis=0)
        non_neg_coef = np.where(coef != 0.)[0]
        idx_non_zero_coef.append(non_neg_coef)
        # Sort coef:
        sort_coef = np.absolute(np.sort(-coef))
        c = 0
        e = 0
        for j in sort_coef:
            e += j
            if e < 80:
                c += 1
        c += 1

    rules = rules['arr_0'].item()

    keys = ['t'+str(i) for i in xrange(0, 5)]

    # Rules used in the models (regarding the coef.)
    for i in range(0, len(coefs)):
        idx = idx_non_zero_coef[i]
        rules['t'+str(i)] = rules['t'+str(i)][idx]

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
    file_names = ['wdbc', 'wine', 'iris']
    experiments = [
        'test_ORM1D_L1LR_deltaTTT',
        'test_decision_tree_L1LR',
        'test_association_rules_L1LR',
    ]

    for exp in list(product(*[file_names, experiments])):
        file_name = exp[0]
        experiment = exp[1]
        main(file_name=file_name, experiment=experiment)

    file_names = ['balance_scale']
    experiments = [
        'test_ORM1D_L1LR',
        'test_decision_tree_L1LR_du',
        'test_association_rules_L1LR',
    ]

    for exp in list(product(*[file_names, experiments])):
        file_name = exp[0]
        experiment = exp[1]
        main(file_name=file_name, experiment=experiment)

    file_names = [
        'heart_disease', 'synthetic', 'synthetic_noisy']
    experiments = [
        'test_ORM1D_L1LR_deltaTTT_d',
        'test_decision_tree_L1LR_du',
        'test_association_rules_L1LR',
    ]

    for exp in list(product(*[file_names, experiments])):
        file_name = exp[0]
        experiment = exp[1]
        main(file_name=file_name, experiment=experiment)
