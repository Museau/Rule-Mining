# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Author:
# - Margaux Luck
#   <margaux.luck@institut-hypercube.org>, <margaux.luck@gmail.com>
# Language: python2.7


import numpy as np
import math

from collections import Counter


def binerization_c(value, inter):
    '''
    Test if value(s) of an individual is(are) in the rule
    and return 1 if the value(s) is(are) in the rule.

    Parameters:
    - value, list.
      List under the form [vf1, vf2,...,vfm] corresponding to the sample's
      values (int) for all the features within the rule.
    - inter, list.
      List of list under the form [ri_low, ri_up].
      Each inter list correspond to one variable conditions.
        -ri_low, int. Lower limit of the rule/interval.
        -ri_up, int. Upper limit of the rule/interval.

    Output:
    - new_val, int. Return 1 if the sample's values
      are in the rule else return 0.
    '''
    new_val = 0
    c = 0
    for i in xrange(len(value)):
        if value[i] >= inter[i][0] and value[i] <= inter[i][1]:
            c += 1
    if c == len(value):
        new_val = 1
    return new_val


def binerization_d(value, inter):
    '''
    Test if value(s) of an individual is(are) in the rule
    and return 1 if the value(s) is(are) in the rule.

    Parameters:
    - value, list.
      List under the form [vf1, vf2,...,vfm] corresponding to the sample's
      values (int) for all the features within the rule.
    - inter, list.
      List of tuple under the form:
      for 1D rules: tuple of conditions -> e.g., (1, 2, 3)
      for 2D rules: tuple of conditions that are tuples in this case
      -> ((1,2), (2,3), (3,4))
      Each tuple correspond to a rule. And each inter tuples to the conditions
      on a(the) variable(s).


    Output:
    - new_val, int. Return 1 if the sample's values
      are in the rule else return 0.
    '''
    new_val = 0

    if len(value) == 1:
        inter = inter[0]

    if set(value).issubset(set(inter)):
        new_val = 1
    return new_val


def binerization_c_d(value, inter):
    '''
    Test if value(s) of an individual is(are) in the rule
    and return 1 if the value(s) is(are) in the rule.

    Parameters:
    - value, list.
      List under the form [vf1, vf2,...,vfm] corresponding to the sample's
      values (int) for all the features within the rule.
    - inter, list.
      List of tuple under the form:
        only applicable for 2D rules: [(min, max), (0, 1, 2)]
        the first tuple is the conditions on the continuous feature and the
        second tuple is the conditions on the discrte feature
      Each tuple correspond to a rule. And each inter tuples to the conditions
      on a(the) variable(s).


    Output:
    - new_val, int. Return 1 if the sample's values
      are in the rule else return 0.
    '''
    new_val = 0
    c = 0
    for i in xrange(len(value)):

        # Conditions on the continuous feature
        if i == 0:
            if value[i] >= inter[i][0] and value[i] <= inter[i][1]:
                c += 1

        # Conditions on the discrete feature
        elif i == 1:
            if value[i] in inter[i]:
                c += 1

    if c == len(value):
        new_val = 1
    return new_val


def distance_rule_fc(value, ri_low, ri_up, maxfc, minfc,
                     wi=False, centered=False):
    '''
    For one feature condition fci, test if a value of an individual
    is in the rule r and change this value into delta_xr_fci
    if the value is not in the rule and asign 1 if the value is in the rule
    or,
    if centered is True: calcul the distance delta_xrC_fci
    of a value of an individual to the center of the rule and
    change this value into delta_xr_fci. If the individual is exactely to
    the center of the rule, he received 1.

    delta_xr_fci = (wi*delta_xr_i)^2

    if centered is False:
        delta_xr_i = 1 - ((xi-ri_up)/(max(Xi)-min(Xi))
                     if xi>ri_up (ri_up is the upper limit)
        delta_xr_i = 1 - ((ri_low-xi)/(max(Xi)-min(Xi))
                     if xi<ri_low (ri_low is the lower limit)
        delta_xr_i = 1 otherwise

    if centered is True:
        r_center = ri_low + ((ri_up-ri_low)/2)
        where:
        *r_center is the center of the rule
        *ri_low is the lower limit of the rule
        *ri_up is the upper limit of the rules

        delta_xrC_i = 1 - ((xi-r_center)/(max(Xi)-min(Xi)) if xi>r_center
        delta_xrC_i = 1 - ((r_center-xi)/(max(Xi)-min(Xi)) if xi<r_center
        delta_xrC_i = 1 otherwise

    Parameters:
    - value, int. Original value of the individual. Value to test.
    - ri_low.
        *Case where the feature is continuous, int.
         Lower limit of the feature's condition of the rule.
        *Case where the feture is discrete, list of int.
         List of the feature's categories of the rule.
    - ri_up.
        *Case where the feature is continuous, int.
         Upper limit of the feature's condition of the rule.
        *Case where the feature is discrete, None.
    - maxfc, int. maximum value of the feature within the training set.
    - minfc, int. minimum value of the feature within the training set.
    - wi, bool. If True, compute wi. If False, wi = 1.
      wi is the weight associated to the feature i
      (e.g., feature frequency among the final set of rules
      (=#occurences/sum of the rule dimensions)).
    - centered, bool. If True, compute the distance to the center of the rule,
                      if False, compute the distance to the rule.

    Output:
    - value.
        *Case where the feature is continuous, float.
         Distance to the hyperrectangle or to the center of the hyperrectangle
         if the sample is outside the rule,
         1 otherwise.
    '''

    if centered:
        r_center = ri_low + ((ri_up-ri_low)/2.0)
        if value == r_center:
            delta_xr_i = 1.0
        elif value >= r_center:
            delta_xr_i = 1.0 - ((value-r_center)/float((maxfc-minfc)))
        else:
            delta_xr_i = 1.0 - ((r_center-value)/float((maxfc-minfc)))

    else:
        if value >= ri_low and value <= ri_up:
            delta_xr_i = 1.0
        elif value >= ri_up:
            delta_xr_i = 1.0 - ((value-ri_up)/float((maxfc-minfc)))
        else:
            delta_xr_i = 1.0 - ((ri_low-value)/float((maxfc-minfc)))

    if not wi:
        wi = 1.0

    return np.float32((wi*delta_xr_i)**2.0)


def distance_rule(val, inter, maxfc, minfc,
                  wi=False, wr=False, centered=False):
    '''
    Compute the distance of a sample to the rule or rule center:
    delta_xr if the sample is outside the rule or is not to the rule center
    and 1 otherwise.

    delta_xr = wr*sqrt(sum_i=1tom(wi*delta_xr_i))^2)

    with - wr : weight associated to the rule (e.g., z-score).
         - wi : weight associated to the feature i
               (e.g., feature frequency among the final set of rules
               (=#occurences/sum of the rule dimensions)).
         - delta_xr_i : distance of the sample to the rule or the rule center
           considering only the feature condition fci
           (cf distance_rule_fc function).

    Parameters:
    - val, list.
      List under the form [vf1, vf2,...,vfm] corresponding to the sample's
      values (int) for all the features within the rule.
    - inter, list.
      List of list under the form [ri_low, ri_up].
      Each inter list correspond to one variable conditions.
        -ri_low, int. Lower limit of the rule/interval.
        -ri_up, int. Upper limit of the rule/interval.
    - maxfc, list of int.
      list of maximum values of the features within the training set.
    - minfc, list of int.
      list of minimum values of the features within the training set.
    - wi, bool. If True, compute wi. If False, wi = 1.
      wi is the weight associated to the feature i
      (e.g., feature frequency among the final set of rules
      (=#occurences/sum of the rule dimensions)).
    - wr, bool. If True, compute wr. If False, wr = 1.
      wr if the weight associated to the rule (e.g., z-score).
    - centered, bool. If True, compute the distance to the center of the rule,
                      if False, compute the distance to the rule.

    Output:
        *Case where the feature is continuous, float.
         - Distance to the rule or to the center of the rule
           if the sample is outside the rule,
           1 otherwise.
    '''
    delta_xr_fci = 0

    if not wi:
        wi = [1]*len(val)

    for i in xrange(len(val)):
        if math.isnan(val[i]):
            val[i] = max(inter[i][0] - minfc[i], maxfc[i] - inter[i][1])
        delta_xr_fci += distance_rule_fc(
            val[i], inter[i][0], inter[i][1],
            maxfc[i], minfc[i], wi[i], centered)

    if not wr:
        wr = 1

    return np.float32(wr*np.sqrt(delta_xr_fci))


def distance_rule_c_d(
        val, inter, maxfc, minfc,
        wi=False, wr=False, centered=False):
    '''

    Parameters:
    - val, list.
      List under the form [vf1, vf2,...,vfm] corresponding to the sample's
      values (int) for all the features within the rule.
    - inter, list.
      List of list under the form [ri_low, ri_up].
      Each inter list correspond to one variable conditions.
        -ri_low, int. Lower limit of the rule/interval.
        -ri_up, int. Upper limit of the rule/interval.
    - maxfc, list of int.
      list of maximum values of the features within the training set.
    - minfc, list of int.
      list of minimum values of the features within the training set.
    - wi, bool. If True, compute wi. If False, wi = 1.
      wi is the weight associated to the feature i
      (e.g., feature frequency among the final set of rules
      (=#occurences/sum of the rule dimensions)).
    - wr, bool. If True, compute wr. If False, wr = 1.
      wr if the weight associated to the rule (e.g., z-score).
    - centered, bool. If True, compute the distance to the center of the rule,
                      if False, compute the distance to the rule.

    Output:
    - float, distance to the rule or to the center of the rule if the sample is
      outside the rule, 1 otherwise.
    '''

    delta_xr_fci = 0

    if not wi:
        wi = [1]*len(val)

    for i in xrange(len(val)):

        # Conditions on the continuous feature
        if i == 0:
            if math.isnan(val[i]):
                val[i] = max(inter[i][0] - minfc[i], maxfc[i] - inter[i][1])
            delta_xr_fci += distance_rule_fc(
                val[i], inter[i][0], inter[i][1],
                maxfc[i], minfc[i], wi[i], centered)

        # Conditions on the discrete feature
        elif i == 1:
            if val[i] in inter[i]:
                delta_xr_fci += np.float32(wi[i]**2.0)

    if not wr:
        wr = 1

    return np.float32(wr*np.sqrt(delta_xr_fci))


def get_local_features(rules, x, method_type, method_params):
    '''
    Transform a global matrix into a local matrix using rules. The local matrix
    is a type of representation of the rules.

    Parameters:
    - rules, np.array. shape = [n_rules, n_rules_caracteristic] where n_rules
      is the number of rules and n_rules_caracteristic are the number of
      characreristic of a rule.
    - x, np.array. shape = [n_sample, n_features] where n_sample is the number
      of samples and n_feature is the number of features.
    - method_type, str. The method used for the transformation.
      Could be: "binerization" or "distance_rule".
      If "binerization", individuals in the rules received 1 and the others 0
      in the local matrix.
      If "distance_rule", the individuals received specific values according to
      their distance to the rule or to the center of the rule.
    - method_params, dict. Parameters specific to the method used for the
      transformation.
      If "binerization": method_params = {}
      If "distance_rule": method_params = {"wi": True (compute wi) or False,
                                           "wr": True (compute wr) or False,
                                           "centered": True
                                           (compute the distance to the center
                                           of the rule) or False}

    Output:
    - local_data, np.array.
    '''
    local_data = []
    for row in rules:
        var = x[:, row[0]]
        inter = row[1]
        var_type = np.unique(row[7])

        if len(var_type) == 1 and var_type == 'c':
            if method_type == "distance_rule":
                maxfc = np.nanmax(var, axis=0)
                minfc = np.nanmin(var, axis=0)

                if method_params["wr"]:
                    method_params["wr"] = row[6]

                if method_params["wi"]:
                    rules_var = sum(rules[:, 0], [])
                    rules_dim = float(len(rules_var))
                    rules_var_freq = Counter(rules_var)
                    method_params["wi"] = [
                                          rules_var_freq[row[0][i]]/rules_dim
                                          for i in xrange(len(row[0]))]

                local_data.append(map(lambda x: distance_rule(
                        x, inter, maxfc, minfc, **method_params), var))

            elif method_type == "binerization":
                local_data.append(map(lambda x: binerization_c(
                            x, inter), var))

        elif len(var_type) == 1 and var_type == 'd':
            local_data.append(map(lambda x: binerization_d(
                x, inter), var))

        else:
            if method_type == 'distance_rule':
                maxfc = np.nanmax(var, axis=0)
                minfc = np.nanmin(var, axis=0)

                if method_params["wr"]:
                    method_params["wr"] = row[6]

                if method_params["wi"]:
                    rules_var = sum(rules[:, 0], [])
                    rules_dim = float(len(rules_var))
                    rules_var_freq = Counter(rules_var)
                    method_params["wi"] = [
                                          rules_var_freq[row[0][i]]/rules_dim
                                          for i in xrange(len(row[0]))]

                local_data.append(map(lambda x: distance_rule_c_d(
                        x, inter, maxfc, minfc, **method_params), var))

            elif method_type == 'binerization':
                local_data.append(map(lambda x: binerization_c_d(
                    x, inter), var))

    local_data = np.array(local_data).T
    return local_data
