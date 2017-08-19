# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Author:
# - Margaux Luck
#   <margaux.luck@institut-hypercube.org>, <margaux.luck@gmail.com>
# - Cecilia Damon
#   <cecilia.damon@institut-hypercube.org>, <cecilia.damon@gmail.com>
# Language: python2.7


import os
import pandas as pd
import numpy as np

from pkg_resources import resource_filename

from rule_mining.rm.datasets.data_preprocess import (
    preprocess_categorical_dataset,
    preprocess_continuous_dataset,
    preprocess_categorical_continuous_dataset)
from rule_mining.rm.datasets.synthetic_dataset import DecisionTreeSamples


'''
The datasets used for the benchmark are download and preprocess
automatically in the Datasets folder by running this python script.

The link for seeing the description of the datasets are given in the following
lines:
http://archive.ics.uci.edu/ml/datasets/Mushroom

*Balance scale dataset
http://archive.ics.uci.edu/ml/datasets/Balance+Scale

*WDBC dataset
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

*Iris dataset
http://archive.ics.uci.edu/ml/datasets/Iris

*Wine dataset
http://archive.ics.uci.edu/ml/datasets/Wine

*Heart desease dataset
http://archive.ics.uci.edu/ml/datasets/Heart+Disease

'''


def load_data(
        file_name, prepro_type, nb_bin=10, binning_way='Struge',
        fill_nan=True):
    '''
    Datasets loader.

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
      nb_bin = 10 by default.
    - binning_way, string. Specify the formula to used for determined the
      number of bin from variable distribution for latter non supervised
      discretization. Must be in ['Scott', 'Freedman-Diaconis',
      'Brooks-Carruthers', 'Huntsberger', 'Struge', 'Rice', 'Square-root'].
      Used only if nb_bin = -1. 'Struge' by default.
    - fill_nan, boolean. Default, True. Can be change in False if you want to
      try without the completion of missing values. Don't work with all the
      methods. When set to False the continuous features are scaled by removing
      the mean and scaling to unit variance.

    Output:
    - data, pandas.DataFrame, shape = [n_samples, n_features + 1]
      Dataset where n_samples is the number of samples and n_features is the
      number of features. The column 'target' correspond to the target.
    - var_type, dict. The keys are the features' names and the values are the
      type of the feature. 'c' means continuous feature and 'd' means discrete
      feature.
    - df_bins_corres, pandas.DataFrame. Return only if prepro_type in
      ['discretized', 'discretized_dummy']. This dataframe gives the
      correspondances between ths bins obtained after discretization and the
      orignial values of the continuous features.
    '''

    path = resource_filename('rule_mining', 'rm/datasets/' + file_name)

    del_col = None
    var_type = {}

    if file_name == 'balance_scale':
        header = [
            'target', 'left_weight', 'left_distance', 'right_weight',
            'right_distance']

        if not os.path.exists(path + '.csv'):
            os.system(
                'wget -O ' + path + '.csv http://archive.ics.uci.edu/ml' +
                '/machine-learning-databases/balance-scale/balance-scale.data')

        data = preprocess_categorical_dataset(
            path, prepro_type, header, fill_nan=fill_nan)

        for colnames in data.columns.tolist():
            var_type[colnames] = 'd'

    elif file_name in ['wdbc', 'wine', 'iris']:

        if file_name == 'wdbc':
            header = ['id', 'target']
            del_col = ['id']

            l_var = [
                'radius', 'texture', 'perimeter', 'area', 'smoothness',
                'compactness', 'concavity', 'concave_points', 'symmetry',
                'fractal_dimension']

            for n_cell in range(1, 4):
                header += ['cell' + str(n_cell) + '_' + i for i in l_var]

            if not os.path.exists(path + '.csv'):
                os.system(
                    'wget -O ' + path + '.csv https://archive.ics.uci.edu/' +
                    'ml/machine-learning-databases/breast-cancer-wisconsin/' +
                    'wdbc.data')

        elif file_name == 'wine':
            header = [
                'target', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash',
                'magnesium', 'total_phenols', 'flavanoids',
                'nonflavanoid_phenols', 'proanthocyanins', 'hue',
                'color_intensity', 'proline',
                'OD280_div_OD315_of_diluted_wines']

            if not os.path.exists(path + '.csv'):
                os.system(
                    'wget -O ' + path + '.csv http://archive.ics.uci.edu/ml/' +
                    'machine-learning-databases/wine/wine.data')

        elif file_name == 'iris':
            header = [
                'sepal_length', 'sepal_width', 'petal_length', 'petal_width',
                'target']

            if not os.path.exists(path + '.csv'):
                os.system(
                    'wget -O ' + path + '.csv http://archive.ics.uci.edu/ml/' +
                    'machine-learning-databases/iris/iris.data')

        if prepro_type == 'discretized':
            data, df_bins_corres = preprocess_continuous_dataset(
                path, prepro_type, header, del_col=del_col,
                nb_bin=nb_bin, binning_way=binning_way, fill_nan=fill_nan)
        else:
            data = preprocess_continuous_dataset(
                path, prepro_type, header, del_col=del_col,
                nb_bin=nb_bin, binning_way=binning_way, fill_nan=fill_nan)

        for colnames in data.columns.tolist():
            var_type[colnames] = 'c'

    elif file_name in ['heart_disease', 'synthetic', 'synthetic_noisy']:

        if file_name == 'heart_disease':
            header = [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

            cat_col = [
                'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal',
                'target']

            if not os.path.exists(path + '.csv'):
                os.system(
                    'wget -O ' + path + '.csv http://archive.ics.uci.edu/ml' +
                    '/machine-learning-databases/heart-disease/' +
                    'processed.cleveland.data')

        elif file_name in ['synthetic', 'synthetic_noisy']:
            n = 500
            header = ['x1', 'x2', 'x3', 'x4', 'target']

            cat_col = ['x3', 'x4', 'target']
            if file_name == 'synthetic':
                dt = DecisionTreeSamples(n)

            elif file_name == 'synthetic_noisy':
                dt = DecisionTreeSamples(
                    n,
                    e={'r01': 12, 'r02': 12, 'r03': 12, 'r04': 12,
                       'r11': 12, 'r12': 12, 'r13': 12, 'r21': 12})

            df = pd.DataFrame(np.concatenate(
                (dt.X, dt.y.reshape(dt.y.shape[0], 1)), 1), dtype='float64')
            df.to_csv(
                path + '.csv', sep=',', index=False, header=False)

        if prepro_type in ['discretized', 'discretized_dummy']:
            data, df_bins_corres = preprocess_categorical_continuous_dataset(
                path, prepro_type, header, cat_col,
                nb_bin=nb_bin, binning_way=binning_way, fill_nan=fill_nan)
        else:
            data = preprocess_categorical_continuous_dataset(
                path, prepro_type, header, cat_col,
                nb_bin=nb_bin, binning_way=binning_way, fill_nan=fill_nan)

        if file_name == 'heart_disease':
            data.loc[data['target'] != 0, 'target'] = 1

        for colnames in data.columns.tolist():
            if colnames in cat_col:
                var_type[colnames] = 'd'
            else:
                var_type[colnames] = 'c'

    if prepro_type in ['discretized', 'discretized_dummy']:
        return data, var_type, df_bins_corres

    else:
        return data, var_type
