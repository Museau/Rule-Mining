# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Author:
# - Margaux Luck
#   <margaux.luck@institut-hypercube.org>, <margaux.luck@gmail.com>
# Language: python2.7

import os
import json
import argparse
import numpy as np

from collections import Mapping, Iterable, Counter

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from rule_mining.rm.datasets import load_data
from rule_mining.rm.RMxD_classification import BuildModel


def main(
        path_test, file_name, prepro_type, nb_bin, binning_way,
        orm_type,
        # orm_params_init
        mod_size_threshold_init,
        size_threshold_init,
        purity_threshold_init,
        z_score_threshold_init,
        # orm_params
        mod_size_threshold,
        size_threshold,
        purity_threshold,
        z_score_threshold,
        local_feature_type,
        local_feature_params,
        classifier_type,
        classifier_params,
        # Missing value completion
        fill_nan):
    '''
    Function that buid either a local model or is corresponding global model.

    Parameters:
    - path_test, string. The name of the experiment. For saving in a specific
      folder.
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
      nb_bin = 10 by default.
    - binning_way, string. Specify the formula to used for determined the
      number of bin from variable distribution for latter non supervised
      discretization. Must be in ['Scott', 'Freedman-Diaconis',
      'Brooks-Carruthers', 'Huntsberger', 'Struge', 'Rice', 'Square-root'].
      Used only if nb_bin = -1. 'Struge' by default.
    - orm_type, string. Could be 'ORM1D' for the generation if 1D rules or
      'ORM1D&2D' for the generation of 1D and 2D rules.
    - mod_size_threshold_init, dict of float. The modality size threshold for
      each one of the modalities at the initialization of the algorithm.
    - size_threshold_init, dict of float. The size threshold for each one of
      the modalities at the initialization of the algorithm.
    - purity_threshold_init, dict of float. The purity threshold for each one
      of the modalities at the initialization of the algorithm.
    - z_score_threshold_init, float. The z-score threshold for each one of the
      modalities at the initialization of the algorithm.
    - mod_size_threshold, dict of list. The list of modality size threshold  to
      test in the grid search cv for each one of the modalities.
    - size_threshold, dict of list. The list of size threshold  to test in the
      grid search cv for each one of the modalities.
    - purity_threshold, dict of list. The list of purity threshold  to
      test in the grid search cv for each one of the modalities.
    - z_score_threshold, dict of list. The list of z-score threshold  to
      test in the grid search cv for each one of the modalities.
    - local_feature_type, str. The method used for the transformation of the
      rules into local features.
      Could be: "binerization" or "distance_rule".
      If "binerization", individuals in the rules received 1 and the others 0
      in the local matrix.
      If "distance_rule", the individuals received specific values according to
      their distance to the rule or to the center of the rule.
    - local_feature_params, dict. Parameters specific to the method used for
      the transformation  of the rules into local features.
    - classifier_type, str. The classifier to use.
      Could be: *Logistic Regression -> 'LR'
                *Support Vector Machine -> 'SVM'
    - classifier_params, dict. Dict containing the classifier parameters.
    Specific to each classier. See sklearn for the possible parameters.
    The function used are:
    *sklearn.linear_model.LogisticRegression() for 'LR'
    *sklearn.svm.SVC() for 'SVM'
    - fill_nan, boolean. Default, True. Can be change in False if you want to
      try without the completion of missing values. Don't work with all the
      methods. When set to False the continuous features are scaled by removing
      the mean and scaling to unit variance.

    Output:
    - print the mean +/- std of the weighted F1 score
    - print the occurence of the parameters used in the final models for
      the 5 splits
    '''

    print "Path: ", path_test

    if orm_type not in ['association_rules', 'decision_tree', 'None']:
        # Set RMxD parameters
        orm_params_init = {
            'mod_size_threshold': mod_size_threshold_init,
            'size_threshold': size_threshold_init,
            'purity_threshold': purity_threshold_init,
            'z_score_threshold': z_score_threshold_init,
        }

        orm_params = {
            'mod_size_threshold': mod_size_threshold,
            'size_threshold': size_threshold,
            'purity_threshold': purity_threshold,
            'z_score_threshold': z_score_threshold,
        }

    elif orm_type == 'association_rules':
        # Set association rules parameters
        # "supp": Support is an indication of how frequently the itemset
        # appears in the database.
        if file_name == 'iris':
            orm_params_init = {"supp": -5, "conf": 0., "thresh": 100}
            orm_params = {"supp": [-5], "conf": [0.], "thresh": [100]}

        elif file_name == 'wine':
            orm_params_init = {"supp": -5, "conf": 0., "thresh": 100}
            orm_params = {"supp": [-5], "conf": [0.], "thresh": [100]}

        elif file_name == 'wdbc':
            orm_params_init = {"supp": -21, "conf": 0., "thresh": 100}
            orm_params = {"supp": [-21], "conf": [0.], "thresh": [100]}

        elif file_name == 'balance_scale':
            orm_params_init = {"supp": -5, "conf": 0., "thresh": 100}
            orm_params = {"supp": [-5], "conf": [0.], "thresh": [100]}

        elif file_name == 'heart_disease':
            orm_params_init = {"supp": -15, "conf": 0., "thresh": 100}
            orm_params = {"supp": [-15], "conf": [0.], "thresh": [100]}

        elif file_name in [
                'synthetic', 'synthetic_noisy']:
            orm_params_init = {"supp": -4, "conf": 0., "thresh": 100}
            orm_params = {"supp": [-4], "conf": [0.], "thresh": [100]}

        else:
            orm_params_init = {"conf": 30, "thresh": 120}
            orm_params = {"conf": [30], "thresh": [120]}

    elif orm_type == 'decision_tree':
        # Set decision tree parameters
        if file_name == 'iris':
            orm_params_init = {
                "min_samples_leaf": 5, "random_state": 10}
            orm_params = {
                "min_samples_leaf": [5], "random_state": [10]}

        elif file_name == 'wine':
            orm_params_init = {
                "min_samples_leaf": 5, "random_state": 10}
            orm_params = {
                "min_samples_leaf": [5], "random_state": [10]}

        elif file_name == 'wdbc':
            orm_params_init = {
                "min_samples_leaf": 21, "random_state": 10}
            orm_params = {
                "min_samples_leaf": [21], "random_state": [10]}

        elif file_name == 'balance_scale':
            orm_params_init = {
                "min_samples_leaf": 5, "random_state": 10}
            orm_params = {
                "min_samples_leaf": [5], "random_state": [10]}

        elif file_name == 'heart_disease':
            orm_params_init = {
                "min_samples_leaf": 15, "random_state": 10}
            orm_params = {
                "min_samples_leaf": [15], "random_state": [10]}

        elif file_name in [
                'synthetic', 'synthetic_noisy']:
            orm_params_init = {
                "min_samples_leaf": 4, "random_state": 10}
            orm_params = {
                "min_samples_leaf": [4], "random_state": [10]}

        else:
            orm_params_init = {"min_samples_leaf": 10, "random_state": 10}
            orm_params = {"min_samples_leaf": [10], "random_state": [10]}

    else:
        orm_params_init = {}
        orm_params = {}

    def convert(data):
        '''
        Get the right encoding for data.
        '''
        if isinstance(data, basestring):
            return str(data)
        elif isinstance(data, Mapping):
            return dict(map(convert, data.iteritems()))
        elif isinstance(data, Iterable):
            return type(data)(map(convert, data))
        else:
            return data

    # Get the right encoding
    orm_params_init = convert(orm_params_init)
    orm_params = convert(orm_params)
    local_feature_params = convert(local_feature_params)
    classifier_params = convert(classifier_params)

    if prepro_type == 'None':
        prepro_type = None

    df_bins_corres = None

    if prepro_type in ['discretized', 'discretized_dummy']:
        # discretization or discretization + one-hot encoding
        data, var_type_, df_bins_corres = load_data(
            file_name, prepro_type, nb_bin, binning_way, fill_nan=fill_nan)

    else:
        # No discretization and/or one-hot encoding
        data, var_type_ = load_data(
            file_name, prepro_type, nb_bin, binning_way, fill_nan=fill_nan)

    print 'Feature types: ', var_type_

    x = data[list(set(data.columns.tolist()) - set(['target']))]
    x_col = x.columns.tolist()
    var_type = []

    for i in x_col:
        var_type.append(var_type_[i])

    y = data['target']

    x = x.as_matrix()
    y = y.as_matrix()

    print 'Input shape: ', x.shape
    print 'Nb of samples by class: ', Counter(y)

    n_splits = 5
    test_size = 0.3
    random_state = 0

    cv1 = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=random_state)

    cv2 = StratifiedKFold(n_splits=5, random_state=2)

    path = 'results/' + file_name

    model = BuildModel(x, y, var_type, cv1, cv2)

    minimization_param = 2

    if orm_type != 'None':

        if prepro_type:
            path_rules_dataset = path + '/rules/data_' + prepro_type + '/'

        else:
            path_rules_dataset = path + '/rules/data/'

        if not os.path.exists(
                os.path.dirname(path_rules_dataset)):
            # Create a folder for save the results
            os.makedirs(path_rules_dataset)

        if orm_type == 'ORM1D':
            # Save the columns name of the input matrix
            np.save(path_rules_dataset + 'col_names', x_col)

        if prepro_type in ['discretized', 'discretized_dummy']:
            # Save the correspondance between the original values of the
            # features and the bins obtains bu discretization
            df_bins_corres.to_csv(
                path_rules_dataset + 'df_bins_corres.csv', index=False)

        id_rules = str(nb_bin) + '_' + binning_way + '_' + \
            orm_type

        if os.path.isfile(path_rules_dataset + 'rules_' + id_rules + '.npz'):
            # Load the rules if they already exist
            model.rules = np.load(
                path_rules_dataset +
                'rules_' + id_rules + '.npz')['arr_0'].item()

        else:
            # Compute the rules
            model.get_rules(
                path_rules_dataset, id_rules,
                orm_type, orm_params_init)

    score_ = model.plug_model(
            path, path_test,
            orm_type, orm_params,
            minimization_param,
            local_feature_type, local_feature_params,
            classifier_type, classifier_params)

    # Print mean +/- weighted F1 score
    print 'Mean score:  {:.1f}'.format(np.mean(score_)*100)
    print 'Std score: {:.1f}'.format(np.std(score_)*100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('--path_test',
                        type=str,
                        default='')
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
                        default='Scott')
    parser.add_argument('--orm_type',
                        type=str,
                        default='ORM1D')
    parser.add_argument('--mod_size_threshold_init',
                        type=json.loads,
                        default='{"0": 0, "1": 0, "2": 0, "3": 0}')
    parser.add_argument('--size_threshold_init',
                        type=json.loads,
                        default='{"0": 0 , "1": 0, "2": 0, "3": 0}')
    parser.add_argument('--purity_threshold_init',
                        type=json.loads,
                        default='{"0": 0, "1": 0, "2": 0, "3": 0}')
    parser.add_argument('--z_score_threshold_init',
                        type=json.loads,
                        default='{"0": 1.96, "1": 1.96, "2": 1.96, "3": 1.96}')
    parser.add_argument('--mod_size_threshold',
                        type=json.loads,
                        default='{"0": [0], "1": [0], "2": [0], "3": [0]}')
    parser.add_argument('--size_threshold',
                        type=json.loads,
                        default='{"0": [0], "1": [0], "2": [0], "3": [0]}')
    parser.add_argument('--purity_threshold',
                        type=json.loads,
                        default='{"0": [0], "1": [0], "2": [0], "3": [0]}')
    parser.add_argument('--z_score_threshold',
                        type=json.loads,
                        default='{"0": [1.96], "1": [1.96], "2": [1.96], "3": \
                            [1.96]}')
    parser.add_argument('--local_feature_type',
                        type=str,
                        default='binerization')
    parser.add_argument('--local_feature_params',
                        type=json.loads,
                        default='{}')
    parser.add_argument('--classifier_type',
                        type=str,
                        default='LR')
    parser.add_argument('--classifier_params',
                        type=json.loads,
                        default='{"penalty": ["l2"], \
                        "C": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], \
                        "class_weight": ["balanced"], "random_state": [0], \
                        "multi_class": ["ovr"]}')
    parser.add_argument('--fill_nan',
                        type=bool,
                        default=True)

    args = parser.parse_args()

    main(args.path_test, args.file_name,
         args.prepro_type, args.nb_bin, args.binning_way,
         args.orm_type,
         args.mod_size_threshold_init, args.size_threshold_init,
         args.purity_threshold_init, args.z_score_threshold_init,
         args.mod_size_threshold, args.size_threshold,
         args.purity_threshold, args.z_score_threshold,
         args.local_feature_type, args.local_feature_params,
         args.classifier_type, args.classifier_params,
         args.fill_nan)
