# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Author:
# - Margaux Luck
#   <margaux.luck@institut-hypercube.org>, <margaux.luck@gmail.com>
# Language: python2.7


import pandas as pd
import numpy as np

from math import ceil, pow, sqrt
from scipy.stats.mstats import mquantiles
from sklearn.preprocessing import Imputer, StandardScaler
from collections import Counter


def missing_values(i):
    '''
    Replace i if a string contening "?" with or without empty space arount into
    None.
    '''
    if pd.notnull(i):
        if str(i).strip() == "?":
            i = None
    return i


def categories_to_int(X, columns):
    '''
    Transform categorical features encoded with string to categorical features
    encoded with int.

    Parameters:
    - X, pandas.DataFrame, shape = [n_samples, , n_features + 1]
      Dataset where n_samples is the number of samples and n_features is the
      number of features. The column 'target' correspond to the target.
    - columns, list. List of columns to encode.

    Output:
    - X_new, pandas.DataFrame, shape = [n_samples, , n_features + 1]
      Dataset  with the new encodage where n_samples is is the number of
      samples and n_features is the number of features. The column 'target'
      correspond to the target.
    '''
    X_new = X.copy()

    for col in columns:
        # Get the unique categories
        ori = list(np.sort(np.unique(X_new[col].dropna())))
        print 'Feature name: ', col
        print 'Original values: ', ori
        # Asign a int to each unique categories
        new = [i for i in range(len(ori))]
        print 'New values: ', new
        X_new[col] = X_new[col].replace(ori, new)

    return X_new


def bin_number(X, binning_way):
    '''
    Determine the number of bin from feature distribution for latter non
    supervised discretization.

    Parameters:
    - X, pandas.Series. The feature to discretized.
    - binning_way, string. Specify the formula to used for determined the
      number of bin from feature distribution for latter non supervised
      discretization. Must be in ['Scott', 'Freedman-Diaconis',
      'Brooks-Carruthers', 'Huntsberger', 'Struge', 'Rice', 'Square-root'].

    Output:
    - int, the number of bin to used for latter non supervised discretization.
    '''

    n = len(X)

    if binning_way == 'Scott':
        nb_bin = ceil((max(X)-min(X))/(3.5*np.std(X)*pow(n, (-1.0/3))))

    elif binning_way == 'Freedman-Diaconis':
        iqr = np.subtract(*np.percentile(X, [75, 25]))
        nb_bin = ceil((max(X)-min(X))/(2*iqr*pow(n, (-1.0/3))))

    elif binning_way == 'Brooks-Carruthers':
        nb_bin = ceil(5 * np.log10(n))

    elif binning_way == 'Huntsberger':
        nb_bin = ceil(1 + 3.332 * np.log10(n))

    elif binning_way == 'Struge':
        nb_bin = ceil(1+np.log2(n))

    elif binning_way == 'Rice':
        nb_bin = ceil(2*pow(n, (1/3)))

    elif binning_way == 'Square-root':
        nb_bin = ceil(sqrt(n))

    return int(nb_bin)


def discretization(X, columns, nb_bin, binning_way='Struge'):
    '''
    Do the discretization of the continuous features.

    Parameters:
    - X, pandas.DataFrame, shape = [n_samples, , n_features + 1]
      Dataset where n_samples is the number of samples and n_features is the
      number of features. The column 'target' correspond to the target.
    - columns, list. List of columns to discretized.
    - nb_bin, int. Number of bin to used for the quantile based discretization.
      If nb_bin = -1, the binning_way must be specify else it will be ignored.
      Only used if prepro_type in ['discretized', 'discretized_dummy'].
      nb_bin = 10 by default.
    - binning_way, string. Specify the formula to used for determined the
      number of bin from variable distribution for latter non supervised
      discretization. Must be in ['Scott', 'Freedman-Diaconis',
      'Brooks-Carruthers', 'Huntsberger', 'Struge', 'Rice', 'Square-root'].
      Used only if nb_bin = -1. 'Struge' by default.

    Output:
    - X_discretized, pandas.DataFrame, shape = [n_samples, , n_features + 1]
      Discretized dataset where n_samples is the number of samples and
      n_features is the number of features. The column 'target' correspond to
      the target.
    - df_bins_corres, pandas.DataFrame. Return only if prepro_type in
      ['discretized', 'discretized_dummy']. This dataframe gives the
      correspondances between ths bins obtained after discretization and the
      orignial values of the continuous features.
    '''

    nb_bin_ = nb_bin
    X_discretized = X.copy()

    # Create a table for save the correspondances between bins and bin_edges
    ar_bins_corres = np.zeros((X.shape[0], len(columns)), dtype='object')

    for idx, col in enumerate(columns):
        print 'Feature name: ', col

        if nb_bin == -1:
            # Calculate the number of bins according to the feature
            # distribution
            nb_bin_ = bin_number(X_discretized[col].dropna(), binning_way)
            print '# bin: ', nb_bin_

        prob = [q/float(nb_bin_) for q in range(1, nb_bin_)]
        bin_edges = mquantiles(X_discretized[col].dropna(), prob=prob)
        # The bin edges are computed as describe above:
        # < bin_edges[0], [bin_edges[0], bin_edges[1][ , ... , >= bin_edges[-1]
        X_discretized[col] = np.digitize(X_discretized[col], bin_edges)
        print 'Bin edges: ', bin_edges
        d = Counter(X_discretized[col])
        print 'List of tuples (Bin, Number of samples): ', sorted(d.items())

        # Records the correspondances
        l = [np.min(X[col])]
        l += list(bin_edges)
        l.append(np.max(X[col]))
        l_ = []
        for j in range(len(l)-1):
            l_.append('['+str(round(l[j], 3))+', '+str(round(l[j+1], 3))+'[')
        ar_bins_corres[:len(l_), idx] = l_

    # Creat a pandas.DataFrame with the correspondances between the Original
    # values and the bins (i.e., save the bin edges for each features)
    df_bins_corres = pd.DataFrame(ar_bins_corres, columns=columns)
    df_bins_corres = df_bins_corres[(df_bins_corres.T != 0).any()]
    df_bins_corres = df_bins_corres.reset_index()
    df_bins_corres = df_bins_corres.rename(columns={'index': 'Bins'})

    return X_discretized, df_bins_corres


def categories_to_dummies(X, columns):
    '''
    Transform categorical features into dummies features (i.e., one-hot
    encoding).

    Parameters:
    - X, pandas.DataFrame, shape = [n_samples, , n_features + 1]
      Dataset where n_samples is the number of samples and n_features is the
      number of features. The column 'target' correspond to the target.
    - columns, list. List of columns to Transform.

    Output:
    - X_dummy, pandas.DataFrame, shape = [n_samples, , n_features + 1]
      Transformed dataset where n_samples is the number of samples and
      n_features is the number of features (i.e., after the one-hot encoding).
      The column 'target' correspond to the target. The features are renamed as
      fellow: OriginalFeatureName_Category.
    '''
    X_dummy = X.copy()
    for col in columns:
        if len(np.unique(X_dummy[col].dropna())) > 2:
            dummies = pd.get_dummies(
                X_dummy[col].dropna()).rename(
                    columns=lambda x: col + '_' + str(x))
            X_dummy = pd.concat(
                [X_dummy, dummies], axis=1)
            X_dummy = X_dummy.drop([col], axis=1)
    return X_dummy


def preprocess_categorical_dataset(
        file_path, prepro_type, header,
        del_col=None, del_row=None, fill_nan=True):
    '''
    Function for pre-procesed the dataset with only categorical features.

    Parameters:
    - file_path, string. Path for load the dataset.
    - prepro_type, string. The type of pre-processing wanted for the dataset.
      Must be in ['None', 'dummy'].
    - header, list. The header of the dataset.
    - del_col, list or None. The columns to delete. Default, None.
    - del_row, list or None. The rows to delete. Default, None.
    - fill_nan, boolean. Default, True. Can be change in False if you want to
      try without the completion of missing values. Don't work with all the
      methods.

    Output:
    - data (i.e., data_preprocess or data_preprocess_dummy in
      function of the parameters choose)
      pandas.DataFrame, shape = [n_samples, n_features + 1]
      Dataset where n_samples is the number of samples and n_features is the
      number of features. The column 'target' correspond to the target.
    '''
    data = pd.read_csv(
        file_path + '.csv',
        delimiter=',',
        header=None,
        names=header)

    if del_col:
        # Delete columns not needed
        header = list(set(header)-set(del_col))
        data = data[header]

    if del_row:
        # Delete rows not needed
        data.drop(data.index[del_row], inplace=True)

    colnames = data.columns.tolist()

    # Process missing values
    for col in colnames:
        data[col] = map(lambda x: missing_values(x), data[col].copy())

    # Transform catgories to int
    data_preprocess = categories_to_int(
        data, header)

    # Make sure missing values are np.nan
    data_preprocess.fillna(value=np.nan, inplace=True)

    if fill_nan:
        # Fill missing values if any
        imputer = Imputer(strategy='most_frequent', copy=False)
        data_preprocess = imputer.fit_transform(data_preprocess)
        data_preprocess = pd.DataFrame(data_preprocess, columns=colnames)

    if prepro_type is None:
        # No one-hot encoding
        return data_preprocess

    elif prepro_type == 'dummy':
        # One-hot encoding for discrete features
        data_preprocess_dummy = categories_to_dummies(
            data_preprocess, list(set(header) - set(['target'])))
        return data_preprocess_dummy


def preprocess_continuous_dataset(
        file_path, prepro_type, header, del_col=None, del_row=None,
        nb_bin=10, binning_way='Struge', fill_nan=True):
    '''
    Function for pre-procesed the dataset with only continuous features.

    Parameters:
    - file_path, string. Path for load the dataset.
    - prepro_type, string. The type of pre-processing wanted for the dataset.
      Must be in ['None', 'discretized'].
    - header, list. The header of the dataset.
    - del_col, list or None. The columns to delete. Default, None.
    - del_row, list or None. The rows to delete. Default, None.
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
    - data (i.e., data_preprocess or data_preprocess_discretized in
      function of the parameters choose)
      pandas.DataFrame, shape = [n_samples, n_features + 1]
      Dataset where n_samples is the number of samples and n_features is the
      number of features. The column 'target' correspond to the target.
    - df_bins_corres, pandas.DataFrame. Return only if prepro_type is
      'discretized'. This dataframe gives the correspondances between ths bins
      obtained after discretization and the orignial values of the continuous
      features.
    '''

    # Load the dataset
    data = pd.read_csv(
        file_path + '.csv',
        delimiter=',',
        header=None,
        names=header)

    if del_col is not None:
        # Delete the columns not needed
        header = list(set(header)-set(del_col))
        data = data[header]

    if del_row:
        # Delete the rows not needed
        data.drop(data.index[del_row], inplace=True)

    # Transform catgories of the target to int
    data_preprocess = categories_to_int(data, ['target'])
    colnames = data_preprocess.columns.tolist()

    # Make sure missing values are np.nan
    data_preprocess.fillna(value=np.nan, inplace=True)

    if fill_nan:
        # Fill missing values if any
        imputer = Imputer(strategy='median', copy=False)
        data_preprocess = imputer.fit_transform(data_preprocess)
        data_preprocess = pd.DataFrame(data_preprocess, columns=colnames)

    if prepro_type is None:
        # No discretization but apply normalization by removing the mean and
        # scaling to unit variance
        scaler = StandardScaler()
        X = data_preprocess[list(set(colnames)-set(['target']))]
        X_scale = scaler.fit_transform(X)
        X_scale = pd.DataFrame(
            X_scale, columns=list(set(colnames)-set(['target'])))
        y = data_preprocess['target']
        data_preprocess = pd.concat([y, X_scale], axis=1)
        return data_preprocess

    elif prepro_type == 'discretized':
        # Apply discretization
        data_preprocess_discretized, df_bins_corres = discretization(
            data_preprocess, list(set(colnames)-set(['target'])),
            nb_bin=nb_bin, binning_way=binning_way)
        return data_preprocess_discretized, df_bins_corres


def preprocess_categorical_continuous_dataset(
            file_path, prepro_type, header, cat_col,
            del_col=None, del_row=None,
            nb_bin=10, binning_way='Struge', fill_nan=True):
    '''
    Function for pre-procesed the dataset with categorical and continuous
    features.

    Parameters:
    - file_path, string. Path for load the dataset.
    - prepro_type, string. The type of pre-processing wanted for the dataset.
      Must be in ['None', 'dummy', 'discretized', 'discretized_dummy'].
    - header, list. The header of the dataset.
    - cat_col, list. The list of categorical columns.
    - del_col, list or None. The columns to delete. Default, None.
    - del_row, list or None. The rows to delete. Default, None.
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
    - data (i.e., data_preprocess or data_preprocess_dummy or
      data_preprocess_discretized or data_preprocess_discretized_dummy in
      function of the parameters choose)
      pandas.DataFrame, shape = [n_samples, n_features + 1]
      Dataset where n_samples is the number of samples and n_features is the
      number of features. The column 'target' correspond to the target.
    - df_bins_corres, pandas.DataFrame. Return only if prepro_type in
      ['discretized', 'discretized_dummy']. This dataframe gives the
      correspondances between ths bins obtained after discretization and the
      orignial values of the continuous features.
    '''

    # Load the dataset
    data = pd.read_csv(
        file_path + '.csv',
        delimiter=',',
        header=None,
        names=header,
        low_memory=False)

    if del_col:
        # Delete the columns not needed
        header = list(set(header)-set(del_col))
        data = data[header]

    if del_row:
        # Delete the rows not needed
        data.drop(data.index[del_row], inplace=True)

    colnames = data.columns.tolist()

    for col in colnames:
        # Make sure all missing values are uder the right form
        data[col] = map(
            lambda x: missing_values(x), data[col].copy())

    # Make sure missing values are np.nan
    data.fillna(value=np.nan, inplace=True)

    # Transform catgories to int
    data_preprocess = categories_to_int(data, cat_col)

    for col in header:
        data_preprocess[col] = pd.to_numeric(data_preprocess[col])

    if fill_nan:
        # Fill missing values if any
        imputer = Imputer(strategy='median', copy=True)
        imputer_cat = Imputer(strategy='most_frequent', copy=True)
        cont_col = list(set(colnames) - set(cat_col))
        # Fill nan values in continuous features
        data_cont = data_preprocess[cont_col]
        colnames_data_cont = data_cont.columns.tolist()
        data_cont = imputer.fit_transform(data_cont)
        data_cont = pd.DataFrame(data_cont, columns=colnames_data_cont)
        # Fill nan values in categorical features
        data_cat = data_preprocess[cat_col]
        colnames_data_cat = data_cat.columns.tolist()
        data_cat = imputer_cat.fit_transform(data_cat)
        data_cat = pd.DataFrame(data_cat, columns=colnames_data_cat)

        data_preprocess = pd.concat([data_cont, data_cat], axis=1)

    if prepro_type is None:
        # No discretization but apply normalization by removing the mean and
        # scaling to unit variance
        scaler = StandardScaler()
        X = data_preprocess[list(set(colnames)-set(['target']))]
        X_scale = scaler.fit_transform(X)
        X_scale = pd.DataFrame(
            X_scale, columns=list(set(colnames)-set(['target'])))
        y = data_preprocess['target']
        data_preprocess = pd.concat([y, X_scale], axis=1)
        return data_preprocess

    elif prepro_type == 'dummy':
        # One-hot encoding on the discrete features
        data_preprocess_dummy = categories_to_dummies(
            data_preprocess, list(set(cat_col)-set(['target'])))
        header = data_preprocess_dummy.columns.tolist()
        # No discretization but apply normalization by removing the mean and
        # scaling to unit variance
        scaler = StandardScaler()
        colnames = data_preprocess_dummy.columns.tolist()
        X = data_preprocess_dummy[list(set(colnames)-set(['target']))]
        X_scale = scaler.fit_transform(X)
        X_scale = pd.DataFrame(
            X_scale, columns=list(set(colnames)-set(['target'])))
        y = data_preprocess_dummy['target']
        data_preprocess = pd.concat([y, X_scale], axis=1)
        return data_preprocess_dummy

    else:
        # Discretization of the continuous features
        data_preprocess_discretized, df_bins_corres = discretization(
            data_preprocess,
            list(set(colnames) - set(cat_col)),
            nb_bin=nb_bin, binning_way=binning_way)

        if prepro_type == 'discretized':
            return data_preprocess_discretized, df_bins_corres

        elif prepro_type == 'discretized_dummy':
            # One-hot encoding on discrete features
            data_preprocess_discretized_dummy = categories_to_dummies(
                data_preprocess_discretized,
                list(set(cat_col)-set(['target'])))
            return data_preprocess_discretized_dummy, df_bins_corres
