# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Author:
# - Margaux Luck
#   <margaux.luck@institut-hypercube.org>, <margaux.luck@gmail.com>
# Language: python2.7


import numpy as np


def compare_rules_two_by_two_c(r1, r2, index_r2, step):
    '''
    Compare two rules generated from the same feature according to step 1 or
    step 2 procedure.
    - step1: delete rules that are included in larger rules with smaller
      z-score
    - step2: delete rules covering smaller rules with higher z-score

    Parameters:
    - r1, np.array. Rule 1. shape = [n_rule_caracteristics, ] where
      n_rules_caracteristic are the number of characreristics of rule 1.
    - r2, np.array. Rule 2. shape = [n_rule_caracteristics, ] where
      n_rules_caracteristic are the number of characreristics of rule 2.
    - index_r2, index of the rule 2 in the np.array containing the sub set
       of rules generated from the same feature(s) to minimize.
    - step, int. Step of the algorithm to perform.
      If 0: perform step1
      If 1: perform step2

    Output:
        If one of the two compared rules must be delete, return the index of
        the rule to delete else return None.
    '''
    test_inter = []
    for i in xrange(len(r1[0])):
        r1_low = r1[1][i][0]
        r1_up = r1[1][i][1]
        r2_low = r2[1][i][0]
        r2_up = r2[1][i][1]

        if step == 0 and r2_low >= r1_low and r2_up <= r1_up:
            test_inter.append(1)
        elif step == 1 and r2_low <= r1_low and r2_up >= r1_up:
            test_inter.append(1)
        else:
            test_inter.append(0)

    if sum(test_inter) == len(r1[0]):
        return index_r2


def compare_rules_two_by_two_d(r1, r2, index_r2, step):
    '''
    Compare two rules generated from the same feature according to step 1 or
    step 2 procedure.
    - step1: delete rules that are included in larger rules with smaller
      z-score
    - step2: delete rules covering smaller rules with higher z-score

    Parameters:
    - r1, np.array. Rule 1. shape = [n_rule_caracteristics, ] where
      n_rules_caracteristic are the number of characreristics of rule 1.
    - r2, np.array. Rule 2. shape = [n_rule_caracteristics, ] where
      n_rules_caracteristic are the number of characreristics of rule 2.
    - index_r2, index of the rule 2 in the np.array containing the sub set
       of rules generated from the same feature(s) to minimize.
    - step, int. Step of the algorithm to perform.
      If 0: perform step1
      If 1: perform step2

    Output:
        If one of the two compared rules must be delete, return the index of
        the rule to delete else return None.
    '''
    if len(r1[7]) == 1:
        if step == 0 and set(r2[1][0]).issubset(set(r1[1][0])):
            return index_r2

        elif step == 1 and set(r1[1][0]).issubset(set(r2[1][0])):
            return index_r2

    else:
        if step == 0 and set(r2[1]).issubset(set(r1[1])):
            return index_r2

        elif step == 1 and set(r1[1]).issubset(set(r2[1])):
            return index_r2


def compare_rules_two_by_two_c_d(r1, r2, index_r2, step):
    '''
    Compare two rules generated from the same feature according to step 1 or
    step 2 procedure.
    - step1: delete rules that are included in larger rules with smaller
      z-score
    - step2: delete rules covering smaller rules with higher z-score

    Parameters:
    - r1, np.array. Rule 1. shape = [n_rule_caracteristics, ] where
      n_rules_caracteristic are the number of characreristics of rule 1.
    - r2, np.array. Rule 2. shape = [n_rule_caracteristics, ] where
      n_rules_caracteristic are the number of characreristics of rule 2.
    - index_r2, index of the rule 2 in the np.array containing the sub set
       of rules generated from the same feature(s) to minimize.
    - step, int. Step of the algorithm to perform.
      If 0: perform step1
      If 1: perform step2

    Output:
        If one of the two compared rules must be delete, return the index of
        the rule to delete else return None.
    '''
    test_inter = []
    for i in xrange(len(r1[0])):

        if i == 0:
            r1_low = r1[1][i][0]
            r1_up = r1[1][i][1]
            r2_low = r2[1][i][0]
            r2_up = r2[1][i][1]

            if step == 0 and r2_low >= r1_low and r2_up <= r1_up:
                test_inter.append(1)

            elif step == 1 and r2_low <= r1_low and r2_up >= r1_up:
                test_inter.append(1)

            else:
                test_inter.append(0)

        elif i == 1:

            if step == 0 and set(r2[1][i]).issubset(set(r1[1][i])):
                test_inter.append(1)

            elif step == 1 and set(r1[1][i]).issubset(set(r2[1][i])):
                test_inter.append(1)

            else:
                test_inter.append(0)

    if sum(test_inter) == len(r1[0]):

        return index_r2


def compare_rules_by_var(rules_sub_set, step):
    '''
    Perform step 1 or step 2 of the rule mininimization algorithm on a subset
    of rules corresponding to rules generated from the same feature(s).
    - step1: delete rules that are included in larger rules with smaller
      z-score
    - step2: delete rules covering smaller rules with higher z-score

    Parameters:
    - rules_sub_set, np.array. Set of rule generated from the same feature(s).
      shape = [n_rules, n_rule_caracteristics] where n_rules is the number of
      initial rules generated from the same feature(s) and
      n_rule_caracteristics are the number of characreristics of a rule.
    - step, int. Step of the algorithm to perform.
      If 0: perform step1
      If 1: perform step2

    Output:
    - minimized_rules_sub_set, np.array.
      shape = [n_rules, n_rule_caracteristics] where n_rules is the number of
      rules generated from the same feature(s) after the minimization procedure
      and n_rule_caracteristics are the number of characreristics of a rule.
    '''
    to_delete = []

    for index_ri, ri in enumerate(rules_sub_set):
        for index_rj, rj in enumerate(rules_sub_set):

            if step == 0:
                if index_ri != index_rj and rj[6] < ri[6]:  # Compare z-scores
                    var_type = np.unique(ri[7])

                    if len(var_type) == 1 and var_type == 'c':
                        index_del = compare_rules_two_by_two_c(
                            ri, rj, index_rj, step)

                    elif len(var_type) == 1 and var_type == 'd':
                        index_del = compare_rules_two_by_two_d(
                            ri, rj, index_rj, step)

                    else:
                        index_del = compare_rules_two_by_two_c_d(
                            ri, rj, index_rj, step)

                    if index_del is not None:
                        to_delete.append(index_del)

            elif step == 1:
                if index_ri != index_rj and rj[6] <= ri[6]:  # Compare z-scores
                    var_type = np.unique(ri[7])

                    if len(var_type) == 1 and var_type == 'c':
                        index_del = compare_rules_two_by_two_c(
                            ri, rj, index_rj, step)

                    elif len(var_type) == 1 and var_type == 'd':
                        index_del = compare_rules_two_by_two_d(
                            ri, rj, index_rj, step)

                    else:
                        index_del = compare_rules_two_by_two_c_d(
                            ri, rj, index_rj, step)

                    if index_del is not None:
                        to_delete.append(index_del)

    to_delete = list(set(to_delete))

    if len(to_delete) != 0:
        minimized_rules_sub_set = np.delete(rules_sub_set, to_delete, axis=0)

    else:
        minimized_rules_sub_set = rules_sub_set

    return minimized_rules_sub_set


def minimization_1_2(rules, n_step):
    '''
    Perform step 1 (and step 2) of the rule mininimization algorithm on a
    set of rules.
    - step1: delete rules that are included in larger rules with smaller
      z-score
    - step2: delete rules covering smaller rules with higher z-score

    Parameters:
    - rules, np.array. shape = [n_rules, n_rule_caracteristics] where n_rules
      is the number of initial rules and n_rule_caracteristics are the number
      of characreristics of a rule.
    - n_step, int. The step(s) to perform. 1 if you want to perform only step1
      and 2 if you want to perform step 1 and 2 of the minimization procedure.

    Output:
    - minimized_rules, np.array. shape = [n_rules, n_rule_caracteristics] where
      n_rules is the number of rules after the minimization procedure and
      n_rule_caracteristics are the number of characreristics of a rule.
    '''

    for step in xrange(n_step):
        c = 0

        for i in np.unique(rules[:, 0]):

            c += 1
            rules_sub_set = rules[[idx for idx, r
                                  in enumerate(rules[:, 0]) if r == i]]

            # Minimisation discretized features
            var_type = rules_sub_set[0, 7][0][0]
            if var_type == 'd' and step == 1:
                if c == 1:
                    minimized_rules = rules_sub_set
                else:
                    minimized_rules = np.concatenate((minimized_rules,
                                                      rules_sub_set))
                continue

            if rules_sub_set.shape[0] > 1:
                minimized_rules_sub_set = compare_rules_by_var(rules_sub_set,
                                                               step)
            else:
                minimized_rules_sub_set = rules_sub_set

            if c == 1:
                minimized_rules = minimized_rules_sub_set

            else:
                minimized_rules = np.concatenate((minimized_rules,
                                                  minimized_rules_sub_set))

        rules = minimized_rules

    return minimized_rules


def minimization_by_modalities(rules, n_step):
    '''
    Perform step 1 (and step 2) of the rule mininimization algorithm on a
    set of rules taking into account the modalities of the rules.
    - step1: delete rules that are included in larger rules with smaller
      z-score
    - step2: delete rules covering smaller rules with higher z-score

    Parameters:
    - rules, np.array. shape = [n_rules, n_rule_caracteristics] where n_rules
      is the number of initial rules and n_rule_caracteristics are the number
      of characreristics of a rule.
    - n_step, int. The step(s) to perform. 1 if you want to perform only step1
      and 2 if you want to perform step 1 and 2 of the minimization procedure.

    Output:
    - minimized_rules, np.array. shape = [n_rules, n_rule_caracteristics] where
      n_rules is the number of rules after the minimization procedure and
      n_rule_caracteristics are the number of characreristics of a rule.
    '''
    c = 0

    for modality in np.unique(rules[:, 3]):
        c += 1
        rules_modality = rules[
            [idx for idx, r in enumerate(rules[:, 3]) if r == modality]]
        minimized_rules_modality = minimization_1_2(rules_modality, n_step)

        if c == 1:
            minimized_rules = minimized_rules_modality

        else:
            minimized_rules = np.concatenate((minimized_rules,
                                              minimized_rules_modality))

    return minimized_rules


def minimization_by_dimension(rules, n_step):
    '''
    Perform step 1 (and step 2) of the rule mininimization algorithm on a
    set of rules taken into acount the dimension of the rule.
    - step1: delete rules that are included in larger rules with smaller
      z-score
    - step2: delete rules covering smaller rules with higher z-score

    Parameters:
    - rules, np.array. shape = [n_rules, n_rule_caracteristics] where n_rules
      is the number of initial rules and n_rule_caracteristics are the number
      of characreristics of a rule.
    - n_step, int. The step(s) to perform. 1 if you want to perform only step1
      and 2 if you want to perform step 1 and 2 of the minimization procedure.

    Output:
    - minimized_rules, np.array. shape = [n_rules, n_rule_caracteristics] where
      n_rules is the number of rules after the minimization procedure and
      n_rule_caracteristics are the number of characreristics of a rule.
    '''

    c = 0
    dimension = [len(i) for i in rules[:, 0]]

    for dim in np.unique(dimension):
        c += 1
        rules_dim = rules[[idx for idx, r in enumerate(dimension) if r == dim]]
        minimized_rules_dim = minimization_by_modalities(rules_dim, n_step)

        if c == 1:
            minimized_rules = minimized_rules_dim

        else:
            minimized_rules = np.concatenate((minimized_rules,
                                              minimized_rules_dim))

    return minimized_rules
