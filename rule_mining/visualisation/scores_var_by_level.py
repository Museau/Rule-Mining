# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Author:
# - Margaux Luck
#   <margaux.luck@institut-hypercube.org>, <margaux.luck@gmail.com>
# Language: python2.7


import numpy as np


def main(file_name):
    '''
    Print the variance of the mean F1 scores accross the different level of
    comparison.

    - file_name, string. The name of the dataset.
    '''

    print 'File name: ', file_name

    def return_scores(file_name, experiment):
        # Load the scores, params and coefs
        path_scores = 'results/' + file_name + '/' + experiment + '/scores/'

        if experiment.startswith('test_ORM1D'):
            scores = np.load(path_scores + 'scores_ORM1D.npz')

        elif experiment.startswith('test_decision_tree'):
            scores = np.load(path_scores + 'scores_decision_tree.npz')

        elif experiment.startswith('test_association_rules'):
            scores = np.load(path_scores + 'scores_association_rules.npz')

        elif experiment in ['random_forest', 'gradient_boosted_tree']:
            path_scores = 'results/' + file_name + '/' + experiment + '/'
            scores = np.load(path_scores + 'scores_params.npz')

        else:
            scores = np.load(path_scores + 'scores_None.npz')

        scores = scores['arr_0'].item()

        if experiment in ['random_forest', 'gradient_boosted_tree']:
            return scores['score']

        else:
            return scores['score_']

    level1_models = [
        'test_L2LR', 'test_L1LR', 'test_SVMlinear', 'test_SVMrbf']

    if file_name == 'balance_scale':
        level1_models = [i + '_du' for i in level1_models]

    if file_name in [
            'heart_disease',
            'synthetic', 'synthetic_noisy']:
        level1_models = [i + '_ddu' for i in level1_models]

    dict_var_level1 = {}

    for experiment in level1_models:
        dict_var_level1[experiment] = np.mean(
            return_scores(file_name, experiment))*100.

    print 'var mean score perf level1: ', np.var(dict_var_level1.values())

    level2_models = ['random_forest', 'gradient_boosted_tree']

    dict_var_level2 = {}

    for experiment in level2_models:
        dict_var_level2[experiment] = np.mean(
            return_scores(file_name, experiment))*100.

    print 'var mean score perf level2: ', np.var(dict_var_level2.values())

    level3_models_dt = [
        'test_decision_tree_L2LR',
        'test_decision_tree_L1LR',
        'test_decision_tree_SVMlinear',
        'test_decision_tree_SVMrbf']

    level3_models_ar = [
        'test_association_rules_L2LR',
        'test_association_rules_L1LR',
        'test_association_rules_SVMlinear',
        'test_association_rules_SVMrbf']

    if file_name in [
            'balance_scale', 'heart_disease',
            'synthetic', 'synthetic_noisy']:

        level3_models_dt = [
            'test_decision_tree_L2LR_du',
            'test_decision_tree_L1LR_du',
            'test_decision_tree_SVMlinear_du',
            'test_decision_tree_SVMrbf_du']

        level3_models_ar = [
            'test_association_rules_L2LR',
            'test_association_rules_L1LR',
            'test_association_rules_SVMlinear']

    dict_var_level3_dt = {}

    for experiment in level3_models_dt:
        dict_var_level3_dt[experiment] = np.mean(
            return_scores(file_name, experiment))*100.

    print 'var mean score level3 dt: ', np.var(dict_var_level3_dt.values())

    dict_var_level3_ar = {}

    for experiment in level3_models_ar:
        dict_var_level3_ar[experiment] = np.mean(
            return_scores(file_name, experiment))*100.

    print 'var mean score level3 ar: ', np.var(dict_var_level3_ar.values())

    local_models = [
        'test_ORM1D_L2LR_deltaTTT',
        'test_ORM1D_L1LR_deltaTTT',
        'test_ORM1D_SVMlinear_deltaTTT',
        'test_ORM1D_SVMrbf_deltaTTT']

    if file_name in [
            'heart_disease',
            'synthetic', 'synthetic_noisy']:
        local_models = [i + '_d' for i in local_models]

    elif file_name == 'balance_scale':
        local_models = [
            'test_ORM1D_L2LR', 'test_ORM1D_L1LR',
            'test_ORM1D_SVMlinear', 'test_ORM1D_SVMrbf']

    dict_var_local = {}
    for experiment in local_models:
        dict_var_local[experiment] = np.mean(
            return_scores(file_name, experiment))*100.

    print 'var mean score local: ', np.var(dict_var_local.values())


if __name__ == '__main__':
    # Dataset with continuous features
    file_names = [
        'wdbc', 'wine', 'iris', 'balance_scale', 'heart_disease',
        'synthetic', 'synthetic_noisy']

    for file in file_names:
        main(file)
