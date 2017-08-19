# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Author:
# - Margaux Luck
#   <margaux.luck@institut-hypercube.org>, <margaux.luck@gmail.com>
# Language: python2.7


import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils import compute_class_weight
from sklearn.metrics import f1_score


class Classification():
    '''
    Select a classifier  among this list:
    [Logistic Regression, Support Vecto Machine].
    Permit his fit, predict and calcul of the F1 score.
    '''
    def __init__(self, classifier_type='LR', classifier_params={}):
        '''
        Initialize a classifier and his parameters.

        Parameters:
        - classifier_type, str. The classifier to use.
          Could be: *Logistic Regression -> 'LR'
                    *Support Vector Machine -> 'SVM'
        - classifier_params, dict. Dict containing the classifier parameters.
        Specific to each classier. See sklearn for the possible parameters.
        The function used are:
        *sklearn.linear_model.LogisticRegression() for 'LR'
        *sklearn.svm.SVC() for 'SVM'

        Output:
        Initialize a classifier
        '''
        self.classifier_type = classifier_type
        self.classifier_params = classifier_params

        if self.classifier_type == 'LR':
            self.model = LogisticRegression(**self.classifier_params)

        elif self.classifier_type == 'SVM':
            self.model = SVC(**self.classifier_params)

        else:
            raise ValueError("classifier_type must be in ['LR', 'SVM']")

        self.fited_model = None

    def fit_model(self, x_train, y_train):
        '''
        Initialze a fited model using the initialized classifier.

        Parameters:
        - x_train, np.array. shape = [n_samples, n_features]
          Training vector, where n_samples is the number of samples and
          n_features is the number of features.
        - y_train, np.array. shape = [n_samples]
          Target vector relative to x_train.

        Output:
        Initialize a fited model.
        '''
        self.y_train = y_train

        self.fited_model = self.model.fit(x_train, y_train)

        return self.fited_model

    def get_score(self, x_test, y_test):
        '''
        Calcul y predict and the weighted F1 score from a fited model.

        Parameters:
        - x_test, np.array. shape = [n_samples, n_features]
          Test vector, where n_samples is the number of samples and
          n_features is the number of features.
        - y_test, np.array. shape = [n_samples]
          Target vector relative to x_test.

        Output:
        - y_proba, np.array. shape = [n_samples]
          Prediction probability vector relative to x_test.
        - y_pred, np.array. shape = [n_samples]
          Prediction vector relative to x_test.
        - y_test, np.array. shape = [n_samples]
          Target vector relative to x_test.
        - score, float. Weighted F1 score.
        - coef, list or np.array. The weights of the models.
        '''
        if self.fited_model is None:
            raise ValueError("self.fited_model must be defined before use")

        # Prediction
        y_pred = self.fited_model.predict(x_test)
        y_proba = self.fited_model.predict_proba(x_test)

        if self.classifier_type == 'LR':
            coef = self.fited_model.coef_

        elif self.classifier_type == 'SVM':
            if self.classifier_params['kernel'] == 'linear':
                coef = self.fited_model.coef_
            elif self.classifier_params['kernel'] == 'rbf':
                coef = [
                    self.fited_model.dual_coef_,
                    self.fited_model.support_vectors_]
            else:
                raise ValueError(
                    "self.classifier_params['kernel'] " +
                    "must be in ['linear', 'rbf']")

        # Weighted F1 Score
        kw = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_test),
                y=y_test)
        ks = np.array([kw[int(yi)] for yi in y_test])
        score = f1_score(
                y_test, y_pred, pos_label=None,
                average='weighted', sample_weight=ks)

        return y_proba, y_pred, y_test, score, coef
