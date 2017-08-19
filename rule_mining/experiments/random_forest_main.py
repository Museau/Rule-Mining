# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Author:
# - Margaux Luck
#   <margaux.luck@institut-hypercube.org>, <margaux.luck@gmail.com>
# Language: pyhton2.7


import os
import pandas as pd
import numpy as np
import argparse

from collections import defaultdict, Counter

from sklearn.model_selection import StratifiedShuffleSplit

from rule_mining.models.random_forest import modelization
from rule_mining.rm.datasets import load_data


def main(file_name, prepro_type, nb_bin, binning_way):
    '''
    Random Forest model.

    Parameters:
    - file_name, string. The name of the datasets wanted. Must be in
      ['wdbc', 'wine', 'iris', 'balance_scale', 'heart_disease', 'synthetic',
       'synthetic_noisy'].
    - prepro_type, string. The type of pre-processing wanted for the dataset.
      Must be in ['None', 'discretized'] if file_name in ['wdbc', 'wine',
      'iris'] ; in ['None', 'dummy'] if file_name equal 'balance_scale' and
      in ['None', 'dummy', 'discretized', 'discretized_dummy'] of file_name in
      ['heart_disease', 'synthetic', 'synthetic_noisy'].
    - nb_bin, int. Number of bin to used for the quantile based discretization.
      If nb_bin = -1, the binning_way must be specify else it will be ignored.
      Only used if prepro_type in ['discretized', 'discretized_dummy'].
    - binning_way, string. Specify the formula to used for determined the
      number of bin from variable distribution for latter non supervised
      discretization. Must be in ['Scott', 'Freedman-Diaconis',
      'Brooks-Carruthers', 'Huntsberger', 'Struge', 'Rice', 'Square-root'].
      Used only if nb_bin = -1.

    Output:
    - print the mean +/- std of the weighted F1 score
    - print the occurence of the parameters used in the final models for
      the 5 splits
    '''

    if prepro_type == 'None':
        prepro_type = None

    # Load data
    if prepro_type in ['discretized', 'discretized_dummy']:
        data, var_type, df_bins_corres = load_data(
            file_name, prepro_type, nb_bin, binning_way)

    else:
        data, var_type = load_data(
            file_name, prepro_type, nb_bin, binning_way)

    col_name = list(set(data.columns.tolist())-set(['target']))
    y = data['target']
    x = data[col_name]

    n_splits = 5
    test_size = 0.3

    # Grid search cv
    random_state = 0

    cv = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=random_state)

    # Random Forest parameters
    rf_params = {
        'n_estimators': range(100, 600, 100),
        'random_state': [0],
        'class_weight': ['balanced']}

    feature_importances_, y_proba, score, best_param = modelization(
        x, y, cv, rf_params, gscv=True)

    # Save the parameters and scores
    scores_params = {
        'feature_importances_': feature_importances_, 'y_proba_': y_proba,
        'score': score, 'best_param': best_param}

    path_rf = 'results/' + file_name + '/random_forest/'

    if not os.path.exists(
            os.path.dirname(path_rf)):
        os.makedirs(path_rf)

    np.savez(
        path_rf + 'scores_params',
        scores_params)

    # Print mean +/- weighted F1 score
    print 'Model: Random Forest'
    print 'Mean score:  {:.1f}'.format(np.mean(score)*100)
    print 'Std score: {:.1f}'.format(np.std(score)*100)

    d = defaultdict(list)

    for i in best_param:
        for k, v in i.items():
            d[k].append(v)

    df = pd.DataFrame.from_dict(d)

    df['param_list'] = df.apply(
        lambda row: ' '.join([str(val) for val in row]), axis=1)

    # Print the occurence of the parameters used in the final models for the
    # 5 splits
    print 'Parameters occurences: ', dict(Counter(df['param_list']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('--file_name',
                        type=str,
                        default='iris')
    parser.add_argument('--prepro_type',
                        type=str,
                        default='discretized')
    parser.add_argument('--nb_bin',
                        type=int,
                        default=10)
    parser.add_argument('--binning_way',
                        type=str,
                        default='Struge')

    args = parser.parse_args()

    main(args.file_name, args.prepro_type, args.nb_bin, args.binning_way)
