# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Author:
# - Margaux Luck
#   <margaux.luck@institut-hypercube.org>, <margaux.luck@gmail.com>
# - Cecilia Damon
#   <cecilia.damon@institut-hypercube.org>, <cecilia.damon@gmail.com>
# Language: python2.7

import os
import numpy as np
import itertools

from collections import OrderedDict

from rule_mining.rm.RMxD import ORM1D, ORM2D
from rule_mining.rm.association_rules import get_association_rules
from rule_mining.rm.RMDT import get_decision_tree_rules
from rule_mining.rm.minimization_utils import minimization_by_dimension
from rule_mining.rm.local_var_utils import get_local_features
from rule_mining.rm.classification import Classification


class BuildModel(object):

    def __init__(self, x, y, var_type, cv1, cv2):
        '''
        Initialization of the class BuildModel.

        Parameters:
        - x, pandas.DataFrame, shape = [n_samples, n_features]
          Training vector, where n_samples is the number of samples and
          n_features is the number of features.
        - y, pandas.Series, shape = [n_samples]
          Target vector relative to X.
        - var_type, list. List of the types of the features of x. 'c' is for
          continuous and 'd' for discrete.
        - cv1, int. Cross-validation generator to used for the train/test
          split.
        - cv2, int. Cross-validation generator to used for the
          sub-train/sub-test split.

        Output:
        - initialization of the class BuildModel.
        '''

        self.x = x
        self.y = y
        self.var_type = var_type
        self.cv1 = cv1
        self.cv2 = cv2
        self.rules = {}
        self.rules_minimized = {}

    def get_rules(
            self, path, id_rules, orm_type, orm_params_init):
        '''
        Get the matrices of rules.

        Parameters:
        - path, string. The path the save the figures and rules.
        - id_rules, string. Extension to give to the rules file name.
        - orm_type, string. Could be 'ORM1D' for the generation if 1D rules or
          'ORM1D&2D' for the generation of 1D and 2D rules.
        - orm_params_init, dict. The parameters to used for the rule
          generation.

        Output:
        - Save the rules under .npz format.

        '''
        self.path = path
        self.id_rules = id_rules
        self.orm_type = orm_type
        self.orm_params_init = orm_params_init

        self.cv1.get_n_splits(self.x, self.y)

        if not os.path.exists(
                os.path.dirname(self.path + '/figures/')):
            os.makedirs(self.path + '/figures/')

        n_split_cv1 = 0

        for train_index, test_index in self.cv1.split(self.x, self.y):

            x_train = self.x[train_index]
            y_train = self.y[train_index]

            def rule_mining(path, x, y, var_type, orm_type, orm_params):
                '''
                Get the matrix of rules.

                Parameters:
                - path, string. The path the save the figures and rules.
                - x, pandas.DataFrame, shape = [n_samples, n_features]
                  Training vector, where n_samples is the number of samples and
                  n_features is the number of features.
                - y, pandas.Series, shape = [n_samples]
                  Target vector relative to X.
                - var_type, list. List of the types of the features of x. 'c'
                  is for continuous and 'd' for discrete.
                - orm_type, string. Could be 'ORM1D' for the generation if 1D
                  rules or 'ORM1D&2D' for the generation of 1D and 2D rules.
                - orm_params_init, dict. The parameters to used for the rule
                  generation.

                Output:
                - return the relevant rules under the form of a np.array.
                  shape (n_rules, n_rule_caracteristics) where n_rules is the
                  number of relevant rules and n_rule_caracteristics the number
                  of caracteristics of the rules(equal to 8).
                '''

                if orm_type in ['ORM1D', 'ORM1&2D']:
                    orm1d = ORM1D(x, np.ravel(y), var_type)
                    orm1d.get_rule_candidates()
                    rules_orm1d = orm1d.select_rules(**orm_params)

                if orm_type in ['ORM2D', 'ORM1&2D']:
                    orm2d = ORM2D(x, np.ravel(y), var_type)
                    orm2d.get_rule_candidates()
                    rules_orm2d = orm2d.select_rules(**orm_params)

                if orm_type == 'ORM1D':
                    return rules_orm1d

                elif orm_type == 'ORM2D':
                    return rules_orm2d

                elif orm_type == 'ORM1&2D':
                    return np.concatenate((rules_orm1d, rules_orm2d))

                elif orm_type == 'association_rules':
                    return get_association_rules(x, y, orm_params)

                elif orm_type == 'decision_tree':
                    return get_decision_tree_rules(
                        path, x, y, orm_params)

            path_dt_rules = self.path + 'figures/' + 't' + str(n_split_cv1)

            rules_orm = rule_mining(
                path_dt_rules,
                x_train, y_train, self.var_type,
                self.orm_type, self.orm_params_init)

            print 'n_plit_cv1: ', n_split_cv1
            print 'n_rules: ', rules_orm.shape[0]

            self.rules['t' + str(n_split_cv1)] = rules_orm

            self.cv2.get_n_splits(x_train, y_train)

            n_split_cv2 = 0

            for sub_train_index, sub_test_index in \
                    self.cv2.split(x_train, y_train):

                x_sub_train = x_train[sub_train_index]
                y_sub_train = y_train[sub_train_index]

                path_dt_rules = self.path + '/figures/' + 't' + \
                    str(n_split_cv1) + '_' + str(n_split_cv2)

                rules_orm = rule_mining(
                    path_dt_rules,
                    x_sub_train, y_sub_train, self.var_type,
                    self.orm_type, self.orm_params_init)

                print 'n_spit_cv1, n_split_cv2:', n_split_cv1, n_split_cv2
                print 'n_rules:', rules_orm.shape[0]

                self.rules['t' + str(n_split_cv1) +
                           '_' + str(n_split_cv2)] = rules_orm

                n_split_cv2 += 1

            n_split_cv1 += 1

        np.savez(self.path + 'rules_' + self.id_rules, self.rules)

    def plug_model(
            self,
            path, path_test,
            orm_type, orm_params,
            minimization_param,
            local_feature_type, local_feature_params,
            classifier_type, classifier_params):

        '''
        Do a predictive model.

        Parameters:
        - path, string.
        - path_test, string.
        - orm_type, string. Could be 'ORM1D' for the generation if 1D rules or
          'ORM1D&2D' for the generation of 1D and 2D rules.
        - orm_params, dict. The parameters to used for the rule mining
          algorithm.
        - local_feature_type, str. The method used for the transformation of
          the rules into local features.
          Could be: "binerization" or "distance_rule".
          If "binerization", individuals in the rules received 1 and the others
          0 in the local matrix.
          If "distance_rule", the individuals received specific values
          according to their distance to the rule or to the center of the rule.
        - local_feature_params, dict. Parameters specific to the method used
          for the transformation  of the rules into local features.
        - classifier_type, str. The classifier to use.
          Could be: *Logistic Regression -> 'LR'
                    *Support Vector Machine -> 'SVM'
        - classifier_params, dict. Dict containing the classifier parameters.
        Specific to each classier. See sklearn for the possible parameters.
        The function used are:
        *sklearn.linear_model.LogisticRegression() for 'LR'
        *sklearn.svm.SVC() for 'SVM'

        Output:
        - score_, array, shape = [n_model]
          n_model is the number of models generated in the cross_validation
          generator.
          Weighted average F1 score.
        '''

        orm_params = OrderedDict(orm_params)
        classifier_params = OrderedDict(classifier_params)

        parameters_ = []
        y_proba_ = []
        y_pred_ = []
        y_true_ = []
        score_ = []
        coef_ = []

        if orm_type not in ['association_rules', 'decision_tree', 'None']:
            classes = np.sort(np.unique(self.y))

            z_score_ = [
                orm_params['z_score_threshold'][str(int(i))] for i in classes]
            z_score_ = list(itertools.product(*z_score_))
            size_threshold_ = [
                orm_params['size_threshold'][str(int(i))] for i in classes]
            size_threshold_ = list(itertools.product(*size_threshold_))
            mod_size_threshold_ = [
                orm_params['mod_size_threshold'][str(int(i))] for i in classes]
            mod_size_threshold_ = list(itertools.product(*mod_size_threshold_))
            purity_threshold_ = [
                orm_params['purity_threshold'][str(int(i))] for i in classes]
            purity_threshold_ = list(itertools.product(*purity_threshold_))

            l = [
                z_score_, size_threshold_,
                mod_size_threshold_, purity_threshold_]
            product = list(itertools.product(*l))

            params_orm = [
                dict(zip(orm_params.keys(), p)) for p in product]

        else:
            product = [
                i for i in apply(itertools.product, orm_params.values())]
            params_orm = [
                dict(zip(orm_params.keys(), p)) for p in product]

        product = [
            i for i in apply(itertools.product, classifier_params.values())]
        params_classif = [
            dict(zip(classifier_params.keys(), p)) for p in product]

        params = list(itertools.product(params_orm, params_classif))

        self.cv1.get_n_splits(self.x, self.y)

        n_split_cv1 = 0

        for train_index, test_index in self.cv1.split(self.x, self.y):

            x_train = self.x[train_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            self.cv2.get_n_splits(x_train, y_train)

            score_gscv = []

            for param in params:

                if orm_type not in [
                        'association_rules', 'decision_tree', 'None']:
                    if isinstance(param[0]['z_score_threshold'], (tuple)):
                        for k, v in param[0].iteritems():
                            param[0][k] = {
                                str(int(k)): v for k, v in dict(
                                    enumerate(v)).items()}

                score_param = []

                n_split_cv2 = 0

                for sub_train_index, sub_test_index in \
                        self.cv2.split(x_train, y_train):

                    y_sub_train, y_sub_test = y_train[sub_train_index], \
                        y_train[sub_test_index]

                    if orm_type != 'None':
                        rules_orm = self.rules['t' + str(n_split_cv1) +
                                               '_' + str(n_split_cv2)]

                        if orm_type not in [
                                'association_rules', 'decision_tree']:
                            # Filter the rules
                            for c in classes:
                                c_ = str(int(c))
                                f_r = rules_orm[
                                    (rules_orm[:, 3] == c) &
                                    (rules_orm[:, 2] >= param[0][
                                        'size_threshold'][c_]) &
                                    (rules_orm[:, 4] >= param[0][
                                        'mod_size_threshold'][c_]) &
                                    (rules_orm[:, 5] >= param[0][
                                        'purity_threshold'][c_]) &
                                    (rules_orm[:, 6] >= param[0][
                                        'z_score_threshold'][c_])]
                                if c == 0:
                                    filtered_rules = f_r
                                else:
                                    filtered_rules = np.concatenate(
                                        (filtered_rules, f_r), axis=0)

                            # Rule minimization
                            minimized_rules = minimization_by_dimension(
                                filtered_rules, minimization_param)

                        else:
                            minimized_rules = rules_orm

                        self.rules_minimized[
                            't' + str(n_split_cv1) +
                            '_' + str(n_split_cv2)] = minimized_rules

                        # Local features
                        local_x_train = get_local_features(
                                minimized_rules, x_train,
                                local_feature_type, local_feature_params)

                    else:
                        local_x_train = x_train

                    # Prediction
                    local_x_sub_train, local_x_sub_test = \
                        local_x_train[sub_train_index], \
                        local_x_train[sub_test_index]

                    classification = Classification(
                        classifier_type=classifier_type,
                        classifier_params=param[1])
                    classification.fit_model(
                        local_x_sub_train, np.ravel(y_sub_train))
                    y_proba, y_pred, y_true, score, coef = \
                        classification.get_score(
                            local_x_sub_test, np.ravel(y_sub_test))

                    score_param.append(score)

                    n_split_cv2 += 1

                score_gscv.append(score_param)

            score_gscv = np.array(score_gscv)

            # Select best combinations of parameters
            mean_score = np.mean(score_gscv, axis=1).tolist()
            idx = mean_score.index(max(mean_score))
            parameters = params[idx]

            if orm_type != 'None':
                rules_orm = self.rules['t' + str(n_split_cv1)]

                if orm_type not in ['association_rules', 'decision_tree']:
                    # Filter the rules
                    for c in classes:
                        c_ = str(int(c))
                        f_r = rules_orm[
                            (rules_orm[:, 3] == c) &
                            (rules_orm[:, 2] >= param[0][
                                'size_threshold'][c_]) &
                            (rules_orm[:, 4] >= param[0][
                                'mod_size_threshold'][c_]) &
                            (rules_orm[:, 5] >= param[0][
                                'purity_threshold'][c_]) &
                            (rules_orm[:, 6] >= param[0][
                                'z_score_threshold'][c_])]
                        if c == 0:
                            filtered_rules = f_r
                        else:
                            filtered_rules = np.concatenate(
                                (filtered_rules, f_r), axis=0)

                    # Rule minimization
                    minimized_rules = minimization_by_dimension(
                        filtered_rules, minimization_param)

                else:
                    minimized_rules = rules_orm

                self.rules_minimized['t' + str(n_split_cv1)] = minimized_rules

                # Local features
                local_x = get_local_features(
                    minimized_rules, self.x,
                    local_feature_type, local_feature_params)

            else:
                local_x = self.x

            # Prediction
            local_x_train, local_x_test = local_x[train_index], \
                local_x[test_index]

            classification = Classification(
                classifier_type=classifier_type,
                classifier_params=parameters[1])
            classification.fit_model(local_x_train, np.ravel(y_train))
            y_proba, y_pred, y_true, score, coef = classification.get_score(
                local_x_test, np.ravel(y_test))

            parameters_.append(parameters)
            y_proba_.append(y_proba)
            y_pred_.append(y_pred)
            y_true_.append(y_true)
            score_.append(score)
            coef_.append(coef)

            print 'n_plit_cv1: ', n_split_cv1
            print 'Best parameter: ', parameters

            if orm_type != 'None':
                print 'n_rules: ', minimized_rules.shape[0]

            n_split_cv1 += 1

        if self.rules_minimized:
            if not os.path.exists(
                    os.path.dirname(path + '/' + path_test + '/final_rules/')):
                os.makedirs(path + '/' + path_test + '/final_rules/')

            np.savez(
                path + '/' + path_test + '/final_rules/final_rules_' +
                orm_type, self.rules_minimized)

        score_ = np.array(score_)

        scores_param_cv = {
            'parameters_': parameters_, 'y_proba_': y_proba_,
            'y_pred_': y_pred_, 'y_true_': y_true_, 'score_': score_,
            'coef_': coef_}

        if not os.path.exists(
                os.path.dirname(path + '/' + path_test + '/scores/')):
            os.makedirs(path + '/' + path_test + '/scores/')

        np.savez(
            path + '/' + path_test + '/scores/scores_' +
            orm_type, scores_param_cv)

        return score_
