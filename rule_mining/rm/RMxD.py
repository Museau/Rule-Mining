# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Author:
# - Margaux Luck
#   <margaux.luck@institut-hypercube.org>, <margaux.luck@gmail.com>
# Language: python2.7


import numpy as np

from collections import Counter
from utils import key_with_max_val
from itertools import combinations

from rule_mining.rm.RMxD_utils import (
    get_intervals_orm_1d_c,
    get_intervals_orm_1d_d,
    get_intervals_orm_2d_cc,
    get_intervals_orm_2d_dd,
    get_intervals_orm_2d_cd)


class ORM_Rule():
    def __init__(self, base_purities, idx_var, inter, rule, rule_type):
        '''
        Initialize the caracteristics of a rule.

        Parameters:
        - basePurities, dict. The purity of the training vector regarding the
          target. k: class modality and v: the purity for the corresponding
          class modality.
        - idx_var, int or list. index of the variable(s)
        - inter, list or list of list. intervale(s)
          (i.e., the condition(s) on the feature(s))
        - rule, list of tuple. The tuple are under the form (tgt, (vf1, vf2))
          where tgt correspond to the target value within the rule and vf1, vf2
          correspond to the sample's values (int) for the features within
          the rule.
        - rule_type, str or list. The rule type
          (i.e., if the rule comes from interger ordered feature(s) ('c') or
          categorical feature(s) ('d') or the both ('c','d')).

        Output:
        Initialize the caracteristics of a rule.
        '''
        self.idx_var = idx_var
        self.inter = inter
        self.base_purities = base_purities
        self.rule = rule
        self.rule_type = rule_type
        self.rule_mod_sizes = Counter(i[0] for i in self.rule)
        self.rule_size = float(sum(self.rule_mod_sizes.values()))
        self.rule_purities = {
                    k: v / self.rule_size
                    for k, v in self.rule_mod_sizes.iteritems()}

        def z_score(rule_size, rule_purity, base_purity):
            '''
            Compute the z-score of a rule regarding a particular modality.

            Parameters:
            - rule_size, int. (i.e., the number of suject in the rule)
            - rule_purity, float. Rule purity.
            - base_purity, float. Base purity.

            return z-score
            '''
            if np.sqrt(base_purity * (1-base_purity)) == 0:
                return 0
            else:
                return np.sqrt(rule_size) * (
                        rule_purity-base_purity)/np.sqrt(
                            base_purity * (1-base_purity))

        self.rule_z_scores = {k: z_score(
            self.rule_size,
            self.rule_purities[k],
            self.base_purities[k]) for k, v in self.rule_purities.iteritems()}
        self.rule_interest_mod = key_with_max_val(self.rule_z_scores)

    def is_relevant_rule(self, mod_size_threshold, size_threshold,
                         purity_threshold, z_score_threshold):
        '''
        Test if a rule is relevant according to rule quality measures
        threshold.

        Parameters:
        - mod_size_threshold, dict of float. The modality size threshold for
          each one of the modalities.
        - size_threshold, dict of float. The size threshold for each one of the
          modalities.
        - purity_threshold, dict of float. The purity threshold for each one of
          the modalities.
        - z_score_threshold, float. The z-score threshold for each one of the
          modalities.

        Output:
        return True if the rule is relevant (i.e., the rule quality measures
        of the rule pass the threshold), False either.
        '''

        rule_mod = str(int(self.rule_interest_mod))

        if self.rule_mod_sizes[self.rule_interest_mod] >= mod_size_threshold[
            rule_mod] and self.rule_size >= size_threshold[rule_mod] and \
                self.rule_purities[self.rule_interest_mod] >= purity_threshold[
                    rule_mod] and self.rule_z_scores[
                        self.rule_interest_mod] >= z_score_threshold[rule_mod]:
            return True

        else:
            return False

    def get_rule_metadata(self):
        '''
        Save the rules caracteristics.

        Output:
        return a np.array of shape = [1,8] containing the rules caracteristics
        in the fellowing order: 1) index of the variable(s), 2) intervals
        (i.e., the condition(s) on the feature(s)), 3) the size of the rule
        (i.e., the number of suject in the rule), 4) most representing
        modality, 5) the rule modality size
        (i.e., the number of subject regarding the majority class of the rule),
        6) the rule purity, 7) the rule z-score and 8) the rule type
        (i.e., if the rule comes from interger ordered feature(s) ('c') or
        categorical feature(s) ('d') or the both ('c','d')).
        '''

        if isinstance(self.idx_var, int):
            idx_var_ = [self.idx_var]
            inter_ = [self.inter]
            rule_type_ = [self.rule_type]

        else:
            idx_var_ = list(self.idx_var)
            inter_ = self.inter
            rule_type_ = self.rule_type

        return np.array([idx_var_,  # 0
                         inter_,  # 1
                         self.rule_size,  # 2
                         self.rule_interest_mod,  # 3
                         self.rule_mod_sizes[self.rule_interest_mod],  # 4
                         self.rule_purities[self.rule_interest_mod],  # 5
                         self.rule_z_scores[self.rule_interest_mod],  # 6
                         rule_type_])  # 7


class ORM(object):
    def __init__(self, x, y, var_type):
        '''
        Initialize the parameters of the ORM algorithm.

        Parameters:
        - x, np.array. shape = [n_samples, n_features]. Training matrix.
        - y, np.array. shape = [n_features, ]. Training vector relative to x.
        - var_type, list. shape = n_features. Metadata indicating for each
          feature if it is discrete categorical ('d') or
          interger ordered ('c'). E.g. ['c', 'd', 'c', 'd', 'd']
        '''
        self.x = x
        self.var_type = np.array(var_type)
        self.y = y
        self.x_bin = np.nanmax(x, axis=0)
        x_bin_C = self.x_bin[list(np.where(np.array(self.var_type) == 'c')[0])]
        x_bin_D = self.x_bin[list(np.where(np.array(self.var_type) == 'd')[0])]
        self.bins_C = np.unique(x_bin_C)
        self.bins_D = np.unique(x_bin_D)
        self.base_mod_sizes = Counter(self.y)
        self.base_size = float(len(self.y))
        self.base_purities = {
            k: v / self.base_size for k, v in self.base_mod_sizes.iteritems()}

    def get_rule_candidates(self, **kargs):
        raise NotImplementedError()

    def select_rules(self, **kargs):
        raise NotImplementedError()


class ORM1D(ORM, object):
    def __init__(self, x, y, var_type):
        '''
        Initialize the parameters of the ORM1D algorithm.

        Parameters:
        - x, np.array. shape = [n_samples, n_features]. Training matrix.
        - y, np.array. shape = [n_features, ]. Training vector relative to x.
        - var_type, list. shape = n_features. Metadata indicating for each
          feature if it is discrete categorical ('d') or
          interger ordered ('c'). E.g. ['c', 'd', 'c', 'd', 'd']
        '''
        super(ORM1D, self).__init__(x, y, var_type)
        self.bin_set_C = None
        self.bin_set_D = None

        classes = list(np.unique(y))
        self.mod_size_threshold = dict((str(el), 10.) for el in classes)
        self.size_threshold = dict((str(el), 10.) for el in classes)
        self.purity_threshold = dict((str(el), 0.) for el in classes)
        self.z_score_threshold = dict((str(el), 1.96) for el in classes)

    def get_rule_candidates(self):
        '''
        Construct all possible candidate rule combinations made from adjacent
        bins for all the integer ordered and categorical features.

        Output:
        - bin_set_C: dict, contains list of closed intervals of the form
          [min, max] corresponding to all possible rules for a range of values
          (e.g. , rules for range of values 0 to 9).
          k: max of a feature and v: list of closed intervals
        - bin_set_D: dict, contains list of tuples of the form
          [(1, ), (2, ), (3, ), (1, 2), (1, 3), ..., (1, 2, 3), ...]
          corresponding to all possible rules.
          k: max of a feature and v: list of closed intervals
        '''
        self.bin_set_C = {}
        self.bin_set_D = {}

        for i in self.bins_C:
            self.bin_set_C[i] = get_intervals_orm_1d_c(int(i))

        for i in self.bins_D:
            self.bin_set_D[i] = get_intervals_orm_1d_d(int(i))

        return self.bin_set_C, self.bin_set_D

    def select_rules(self,
                     mod_size_threshold=None,
                     size_threshold=None,
                     purity_threshold=None,
                     z_score_threshold=None):
        '''
        Iterate over the feature and the intervals and calcul the rule quality
        measure. Then return the rules matching the rule quality measures
        threshod.

        Parameters:
        - mod_size_threshold, dict of float. The modality size threshold for
          each one of the modalities.
        - size_threshold, dict of float. The size threshold for each one of the
          modalities.
        - purity_threshold, dict of float. The purity threshold for each one of
          the modalities.
        - z_score_threshold, float. The z-score threshold for each one of the
          modalities.

        Output:
        - return the relevant rules under the form of a np.array.
          shape (n_rules, n_rule_caracteristics) where n_rules is the number
          of relevant rules and n_rule_caracteristics the number of
          caracteristics of the rules (equal to 7).
        '''
        if not mod_size_threshold:
            mod_size_threshold = self.mod_size_threshold

        if not size_threshold:
            size_threshold = self.size_threshold

        if not purity_threshold:
            purity_threshold = self.purity_threshold

        if not z_score_threshold:
            z_score_threshold = self.z_score_threshold

        selection_param = {
                          "mod_size_threshold": mod_size_threshold,
                          "size_threshold": size_threshold,
                          "purity_threshold": purity_threshold,
                          "z_score_threshold": z_score_threshold
                          }
        rules = []

        # Iterating over the features
        for idx_var, val in enumerate(self.x_bin):
            tgt_var_rule = zip(self.y, self.x[:, idx_var])
            var_rule_type = self.var_type[idx_var]

            if var_rule_type == 'c':
                inter_set = self.bin_set_C[val]
            elif var_rule_type == 'd':
                inter_set = self.bin_set_D[val]

            for inter in inter_set:
                if var_rule_type == 'c':
                    rule = filter(lambda x: x[1] in range(
                                inter[0], inter[1]+1), tgt_var_rule)

                elif var_rule_type == 'd':
                    rule = filter(lambda x: x[1] in inter, tgt_var_rule)

                if rule:
                    rule = ORM_Rule(self.base_purities, idx_var,
                                    inter, rule, var_rule_type)
                    if rule.is_relevant_rule(**selection_param):
                        rules.append(rule.get_rule_metadata())

        rules = np.array(rules)
        return rules


class ORM2D(ORM, object):
    def __init__(self, x, y, var_type):
        '''
        Initialize the parameters of the ORM2D algorithm.

        Parameters:
        - x, np.array. shape = [n_samples, n_features]. Training matrix.
        - y, np.array. shape = [n_features, ]. Training vector relative to x.
        - var_type, list. shape = n_features. Metadata indicating for each
          feature if it is discrete categorical ('d') or
          interger ordered ('c'). E.g. ['c', 'd', 'c', 'd', 'd']
        '''
        super(ORM2D, self).__init__(x, y, var_type)
        self.rule_candidate = None

        classes = list(np.unique(y))
        self.mod_size_threshold = dict((str(el), 10.) for el in classes)
        self.size_threshold = dict((str(el), 10.) for el in classes)
        self.purity_threshold = dict((str(el), 0.) for el in classes)
        self.z_score_threshold = dict((str(el), 1.96) for el in classes)

    def get_rule_candidates(self):
        '''
        Construct all possible candidate rule combinations made from adjacent
        bins for all the "integer continuous" feature.

        Output:
        - dict: dict, contains list of closed intervals of the form [min, max]
          corresponding to all possible rules for each faeture.
          k: feature index and v: list of closed intervals
        '''
        self.rule_candidate = {}
        idx_x = range(self.x.shape[1])

        if self.x.shape[1] != 2:
            idx_x_2by2 = list(combinations(idx_x, 2))

        else:
            idx_x_2by2 = [tuple(idx_x)]

        for idx_var in idx_x_2by2:
            idx_var_type = self.var_type[list(idx_var)]
            idx_var = tuple([idx_var[i] for i in np.argsort(idx_var_type)])

            if idx_var_type[0] == 'c' and idx_var_type[1] == 'c':
                self.rule_candidate[idx_var] = \
                            get_intervals_orm_2d_cc(self.x, idx_var)

            elif idx_var_type[0] == 'd' and idx_var_type[1] == 'd':
                self.rule_candidate[idx_var] = \
                            get_intervals_orm_2d_dd(self.x, idx_var)

            elif np.sort(
                idx_var_type)[0] == 'c' and np.sort(
                    idx_var_type)[1] == 'd':
                self.rule_candidate[idx_var] = \
                            get_intervals_orm_2d_cd(self.x, idx_var)

        return self.rule_candidate

    def select_rules(self,
                     mod_size_threshold=None,
                     size_threshold=None,
                     purity_threshold=None,
                     z_score_threshold=None):
        '''
        Iterate over the feature and the intervals and calcul the rule quality
        measure. Then return the rules matching the rule quality measures
        threshod.

        Parameters:
        - mod_size_threshold, dict of float. The modality size threshold for
          each one of the modalities.
        - size_threshold, dict of float. The size threshold for each one of the
          modalities.
        - purity_threshold, dict of float. The purity threshold for each one of
          the modalities.
        - z_score_threshold, float. The z-score threshold for each one of the
          modalities.

        Output:
        - return the relevant rules under the form of a np.array.
          shape (n_rules, n_rule_caracteristics) where n_rules is the number of
          relevant rules and n_rule_caracteristics the number of
          caracteristics of the rules(equal to 8).
        '''
        if not mod_size_threshold:
            mod_size_threshold = self.mod_size_threshold

        if not size_threshold:
            size_threshold = self.size_threshold

        if not purity_threshold:
            purity_threshold = self.purity_threshold

        if not z_score_threshold:
            z_score_threshold = self.z_score_threshold

        selection_param = {
                          "mod_size_threshold": mod_size_threshold,
                          "size_threshold": size_threshold,
                          "purity_threshold": purity_threshold,
                          "z_score_threshold": z_score_threshold
                          }
        rules = []
        # Iterating over the features
        for idx_var, intervals in self.rule_candidate.iteritems():
            idx_var_type = self.var_type[list(idx_var)]
            tgt_var_rule = zip(
                self.y, zip(self.x[:, idx_var[0]], self.x[:, idx_var[1]]))
            # Iterating over the intervals
            for inter in intervals:

                rule = filter(lambda x: np.isfinite(x[1]).all(), tgt_var_rule)

                # Continuous case
                if idx_var_type[0] == 'c' and idx_var_type[1] == 'c':
                    for i in range(len(idx_var)):
                        rule = filter(lambda x: int(x[1][i]) in range(
                                int(inter[i][0]), int(inter[i][1])+1), rule)

                # Categorical case
                elif idx_var_type[0] == 'd' and idx_var_type[1] == 'd':
                    rule = filter(lambda x: x[1] in inter, rule)

                # Continuous and categorical case
                elif idx_var_type[0] == 'c' and idx_var_type[1] == 'd':
                    [(min, max), (0, 1, 2)]
                    rule = filter(lambda x: int(x[1][0]) in range(
                            int(inter[0][0]), int(inter[0][1])+1), rule)
                    rule = filter(lambda x: x[1][1] in inter[1], rule)

                if rule:
                    rule = ORM_Rule(self.base_purities, idx_var,
                                    inter, rule, list(idx_var_type))
                    if rule.is_relevant_rule(**selection_param):
                        rules.append(rule.get_rule_metadata())

        rules = np.array(rules)
        return rules
