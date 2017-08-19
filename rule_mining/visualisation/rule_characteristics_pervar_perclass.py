# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Author:
# - Margaux Luck
#   <margaux.luck@institut-hypercube.org>, <margaux.luck@gmail.com>
# Language: python2.7


import numpy as np


def main(file_name):
    '''
    Print the mean z-score of the rules per features and per classes.

    Parameters:
    - file_name, string. The name of the dataset.
    '''

    print 'file_name: ', file_name

    path = 'test_ORM1D_L2LR_deltaTTT'

    if file_name == 'balance_scale':
        path = 'test_ORM1D_L2LR'

    elif file_name in [
            'heart_disease',
            'synthetic', 'synthetic_noisy']:
        path = 'test_ORM1D_L2LR_deltaTTT_d'

    path_rules = 'results/' + file_name + '/' + path + '/final_rules/'

    # Load the rules datasets
    rules = np.load(path_rules + 'final_rules_ORM1D.npz')

    rules = rules['arr_0'].item()

    keys = ['t'+str(i) for i in xrange(0, 5)]

    # Models complexity
    rule_set = np.array([])
    rule_set = rule_set.reshape(0, 8)

    for k in keys:
        rule_set = np.append(rule_set, rules[k], axis=0)

    print '# rules: ', rule_set.shape[0]

    if file_name == 'balance_scale':
        path_col_names = 'data'

    else:
        path_col_names = 'data_discretized'

    features = np.load(
        'results/' + file_name + '/rules/' + path_col_names + '/col_names.npy')

    for c in np.unique(rule_set[:, 3]):
        print 'Class: ', c
        idx = np.where(rule_set[:, 3] == c)[0]
        rule_set_c = rule_set[idx]

        for f in np.unique(rule_set_c[:, 0]):
            f = f[0]
            print 'feature: ', features[f]
            l = sum(rule_set_c[:, 0], [])
            l = np.array(l)
            idx = np.where(l == f)[0]
            rule_set_c_f = rule_set_c[idx]
            print 'z-score'
            print 'mean: ', np.mean(rule_set_c_f[:, 6], axis=0)


if __name__ == '__main__':
    # Dataset with continuous features
    file_names = [
        'wdbc', 'wine', 'iris', 'balance_scale',
        'heart_disease', 'synthetic', 'synthetic_noisy']

    for file_name in file_names:
        main(file_name=file_name)
