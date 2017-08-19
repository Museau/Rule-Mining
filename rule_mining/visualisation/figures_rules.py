# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Author:
# - Margaux Luck
#   <margaux.luck@institut-hypercube.org>, <margaux.luck@gmail.com>
# Language: python2.7


import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from itertools import groupby, chain
from operator import itemgetter
import matplotlib.pyplot as plt


def get_union_of_rules_by_var_modality(rules):
    '''
    Prepare the array for representing the rules in a map.

    Parameters:
    - rules, np.array. The rules.
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
        dict_union_rules_inter[k] = Counter(list(chain(*v)))

    return dict_union_rules_inter


def plot_rules(rules, labels, save):
    '''
    Plot the rule, save the figure.

    Parameters:
    - rules, np.array. The rules.
    - labels, list. The labels.
    - save, string. Save name.
    '''

    fig, ax = plt.subplots()
    im = ax.pcolor(rules, cmap=plt.cm.Blues, alpha=0.8)

    # Format
    fig = plt.gcf()

    # turn off the frame
    ax.set_frame_on(False)
    # Ensure heatmap cells are square.
    ax.set_aspect('equal')

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(rules.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(rules.shape[1]) + 0.5, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # Set the labels
    ax.set_xticklabels(np.arange(1, rules.shape[1]+1), minor=False)
    ax.set_yticklabels(labels, minor=False)

    # rotate the
    plt.xticks(rotation=90)

    ax.grid(False)

    # Turn off all the ticks
    ax = plt.gca()

    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    fig.set_tight_layout(True)
    fig.colorbar(im)

    fig.savefig(save + '_rule_map.pdf')
    plt.close()


def main(file_name):
    '''
    Plot the rules and print some statistics for a dataset.

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

    dict_union_rules_inter = get_union_of_rules_by_var_modality(rule_set)

    if file_name == 'balance_scale':
        path_col_names = 'data'

    else:
        path_col_names = 'data_discretized'

    features = np.load(
        'results/' + file_name + '/rules/' + path_col_names + '/col_names.npy')
    n_features = len(features)

    classes = np.unique(rule_set[:, 3])

    rule_map = {}

    for cl in classes:
        rule_map[cl] = np.zeros((n_features, 10))

    for k, v in dict_union_rules_inter.items():
        f = k[0]
        c = k[1]

        for k_, v_ in v.items():
            rule_map[c][f, k_] = v_

    path_rule_map = 'results/' + file_name + '/rules/'

    for k, v in rule_map.items():
        save = path_rule_map + 'class' + str(int(k))

        if file_name == 'wdbc':
            ordered_features = [
                'mean concave points', 'worst perimeter',
                'worst radius', 'worst area',
                'worst concave points', 'mean concavity',
                'worst concavity', 'mean perimeter',
                'mean radius', 'mean area',
                'se area', 'mean compactness',
                'worst compactness', 'se concavity',
                'se radius', 'se perimeter',
                'se concave points', 'worst texture',
                'mean texture', 'mean smoothness',
                'se compactness', 'worst smoothness',
                'mean symmetry', 'worst symmetry',
                'worst fractal dimension', 'se fractal dimension',
                'se symmetry', 'mean fractal dimension',
                'se texture', 'se smoothness']
            ordered_f = [
                'cell1_concave_points', 'cell3_perimeter',
                'cell3_radius', 'cell3_area',
                'cell3_concave_points', 'cell1_concavity',
                'cell3_concavity', 'cell1_perimeter',
                'cell1_radius', 'cell1_area',
                'cell2_area', 'cell1_compactness',
                'cell3_compactness', 'cell2_concavity',
                'cell2_radius', 'cell2_perimeter',
                'cell2_concave_points', 'cell3_texture',
                'cell1_texture', 'cell1_smoothness',
                'cell2_compactness', 'cell3_smoothness',
                'cell1_symmetry', 'cell3_symmetry',
                'cell3_fractal_dimension', 'cell2_fractal_dimension',
                'cell2_symmetry', 'cell1_fractal_dimension',
                'cell2_texture', 'cell2_smoothness']
            r = pd.DataFrame(v, index=features)
            r = r.ix[ordered_f]
            v = r.as_matrix()
            features_final = ordered_features

        elif file_name in ['wine', 'iris']:
            features_final = features

        elif file_name == 'balance_scale':
            ordered_features = [
                'right distance', 'right weight',
                'left distance', 'left weight']
            ordered_f = [
                'right_distance', 'right_weight',
                'left_distance', 'left_weight']
            r = pd.DataFrame(v, index=features)
            r = r.ix[ordered_f]
            v = r.as_matrix()
            features_final = ordered_features

        elif file_name == 'heart_disease':
            ordered_features = [
                'thal', 'cp', 'ca',
                'oldpeak', 'thalach', 'exang',
                'slope', 'age', 'sex',
                'chol', 'trestbps',
                'restecg']
            r = pd.DataFrame(v, index=features)
            r = r.ix[ordered_features]
            v = r.as_matrix()
            features_final = ordered_features

        elif file_name in [
                'synthetic', 'synthetic_noisy']:
            ordered_features = ['x1', 'x2', 'x3', 'x4']
            r = pd.DataFrame(v, index=features)
            r = r.ix[ordered_features]
            v = r.as_matrix()
            features_final = ordered_features

        plot_rules(v, features_final, save)

    # [[col_names] => 0
    # [inter] => 1
    # rule_size => 2
    # rule_interes_mod => 3
    # rule_mod_sizes => 4
    # rule_purities => 5
    # rule_z_scores => 6
    # rule_type] => 7

    for c in np.unique(rule_set[:, 3]):
        print 'Class: ', c
        idx = np.where(rule_set[:, 3] == c)[0]
        rule_set_c = rule_set[idx]
        print '# rules: ', rule_set_c.shape[0]
        print 'z-score, size, size mod., purity'
        print 'mean: ', np.mean(rule_set_c[:, [6, 2, 4, 5]], axis=0)
        print 'min: ', np.min(rule_set_c[:, [6, 2, 4, 5]], axis=0)
        print 'max: ', np.max(rule_set_c[:, [6, 2, 4, 5]], axis=0)


def main2():
    '''
    Plot the true rules for the synthetic dataset.
    '''
    ordered_features = ['x1', 'x2', 'x3', 'x4']
    r_class0 = np.array([
        [1, 1, 1, 1, 1.25, 2, 1.82, 1, 1, 1],
        [1, 1, 1.21, 2, 2, 1.25, 1, 1, 1, 1],
        [2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 2, 0, 0, 0, 0, 0, 0, 0]])
    r_class1 = np.array([
        [1, 1, 1, 1, 0.75, 0, 0.28, 1, 1, 1],
        [1, 1, 0.21, 0, 0, 0.75, 1, 1, 1, 1],
        [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 2, 1, 0, 0, 0, 0, 0, 0, 0]])
    r_class2 = np.array([
        [0, 0, 0, 0, 0, 0, 0.28, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
    path_rule_map = 'results/' + 'synthetic' + '/rules/'
    save_class0 = path_rule_map + 'class0_true'
    save_class1 = path_rule_map + 'class1_true'
    save_class2 = path_rule_map + 'class2_true'
    plot_rules(r_class0, ordered_features, save_class0)
    plot_rules(r_class1, ordered_features, save_class1)
    plot_rules(r_class2, ordered_features, save_class2)


if __name__ == '__main__':
    # Dataset with continuous features
    file_names = [
        'wdbc', 'wine', 'iris', 'balance_scale',
        'heart_disease',
        'synthetic', 'synthetic_noisy']

    for file_name in file_names:
        main(file_name=file_name)

    main2()
