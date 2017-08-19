# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Author:
# - Margaux Luck
#   <margaux.luck@institut-hypercube.org>, <margaux.luck@gmail.com>
# Language: pyhton2.7


import numpy as np
import itertools

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.utils import compute_class_weight, indexable

from joblib import Parallel, delayed


def modelization(x, y, cv, rf_param, gscv):
    '''
    Random forest model using sklearn.

    Parameters:
    - x, pandas.DataFrame, shape = [n_samples, n_features]
      Training vector, where n_samples is the number of samples and
      n_features is the number of features.
    - y, pandas.Series, shape = [n_samples]
      Target vector relative to X.
    - cv, int, cross-validation generator.
    - rf_param, dict or dict of list.  The dict correspond to Random Forest
      parameters from sklearn. If the dict is a dict of list. The lists contain
      the different parameters to test for the grid search cv.
    - gscv, boolean. True if a grid search cv must be executed and False
      either. Note: must be True if rf_param is a dict of list, else must be
      False.

    Output:
    - feature_importances_, list (shape=[n_model]) of arrays,
      of shape = [n_features]. n_model is the number of models generated in
      the cross validation generator and n_features is the number of fatures.
      The feature importances (the higher, the more important the feature).
    - y_proba, list of list. Each list correspond to the probability estimates
      for one model.
    - score, array, shape = [n_model]
      n_model is the number of models generated in the cross_validation
      generator.
      Weighted average F1 score.
    - best_param. list of the best parameters if gscv parameter is set to True.
      shape = len(n_model). Else, the list is empty.
    '''

    y_proba = []
    score = []
    feature_importances_ = []
    best_param = []

    n_model = 0

    cv.get_n_splits(x, y)

    for train_index, test_index in cv.split(x, y):

        y_train, y_test = y.iloc[list(train_index)], y.iloc[list(test_index)]
        x_train, x_test = x.iloc[list(train_index)], x.iloc[list(test_index)]

        if gscv is True:
            # Do a grid search cv
            parameters = grid_search_cv(x_train, y_train, rf_param)

            best_param.append(parameters)

            rf = RandomForestClassifier(**parameters)

        else:
            rf = RandomForestClassifier(**rf_param)

        model = rf.fit(x_train, y_train)

        # Importance of the features in the model
        feature_importances_.append(model.feature_importances_)

        # Prediction
        y_pred = model.predict(x_test)

        # Prediction in term of probabilities
        y_proba.append(model.predict_proba(x_test))

        # Weighted F1 Score
        kw = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_test), y=y_test)
        ks = np.array([kw[int(yi)] for yi in y_test])
        score.append(f1_score(
            y_test, y_pred, pos_label=None,
            average='weighted', sample_weight=ks))

        n_model += 1

    score = np.array(score)

    return feature_importances_, y_proba, score, best_param


def modelization_gscv(x, y, cv, rf_param):
    '''
    Random forest model using sklearn.
    Fonction used for the grid search cv parallel computing.

    Parameters:
    - x, pandas.DataFrame, shape = [n_samples, n_features]
      Training vector, where n_samples is the number of samples and
      n_features is the number of features.
    - y, pandas.Series, shape = [n_samples]
      Target vector relative to X.
    - cv, int, cross-validation generator.
    - rf_param, dict.  The dict correspond to Random Forest
      parameters from sklearn.

    Output:
    - score, array, shape = [n_model]
      n_model is the number of models generated in the cross validation
      generator.
      Weighted average F1 score.
    '''

    feature_importances_, y_proba, score, best_param = modelization(
            x, y, cv, rf_param, False)

    return score


def grid_search_cv(x, y, rf_param):
    """
    Grid search cv function.

    Parameters:
    - x, pandas.DataFrame, shape = [n_samples, n_features]
      Training vector, where n_samples is the number of samples and
      n_features is the number of features.
    - y, pandas.Series, shape = [n_samples]
      Target vector relative to X.
    - rf_param, dict of list.  The dict correspond to Random Forest parameters
      from sklearn. The lists contain the different parameters to test for the
      grid search cv.

    Output:
    - parameters, dict. The dict correspond to Random Forest parameters choosen
      during the grid search cv.
    """
    # Construct all possible combination of parameters
    product = [i for i in apply(itertools.product, rf_param.values())]
    params = [dict(zip(rf_param.keys(), p)) for p in product]

    x, y = indexable(x, y)

    cv = StratifiedKFold(n_splits=5, random_state=2)

    score = Parallel(n_jobs=-2)(delayed(modelization_gscv)(
        x, y, cv, param) for param in params)
    score = np.array(score)

    mean_score = np.mean(score, axis=1).tolist()
    idx = mean_score.index(max(mean_score))
    parameters = params[idx]

    return parameters
