# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Author:
# - Margaux Luck
#   <margaux.luck@institut-hypercube.org>, <margaux.luck@gmail.com>
# - Cecilia Damon
#   <cecilia.damon@institut-hypercube.org>, <cecilia.damon@gmail.com>
# Language: python2.7


import pandas as pd
import numpy as np
import fim


'''
    http://www.borgelt.net/doc/apriori/apriori.html
    report  values to report with a assoc. rule    (default: aC)
            a     absolute item set  support (number of transactions)
            s     relative item set  support as a fraction
            S     relative item set  support as a percentage
            b     absolute body set  support (number of transactions)
            x     relative body set  support as a fraction
            X     relative body set  support as a percentage
            h     absolute head item support (number of transactions)
            y     relative head item support as a fraction
            Y     relative head item support as a percentage
            c     rule confidence as a fraction
            C     rule confidence as a percentage
            l     lift value of a rule (confidence/prior)
            L     lift value of a rule as a percentage
            e     value of rule evaluation measure
            E     value of rule evaluation measure as a percentage
            Q     support of the empty set (total number of transactions)
            (     combine values in a tuple (must be first character)
            [     combine values in a list  (must be first character)
            #     pattern spectrum as a dictionary  (no patterns)
            =     pattern spectrum as a list        (no patterns)
            |     pattern spectrum as three columns (no patterns)

    supp    minimum support    of an assoc. rule   (default: 10)
            (positive: percentage, negative: absolute number)
    conf    minimum confidence of an assoc. rule   (default: 80%)
    zmin    mi
####### Eval ############
           x     none       no measure / zero (default)
            b     ldratio    binary logarithm of support quotient       (+)
            c     conf       rule confidence                            (+)
            d     confdiff   absolute confidence difference to prior    (+)
            l     lift       lift value (confidence divided by prior)   (+)
            a     liftdiff   absolute difference of lift value to 1     (+)
            q     liftquot   difference of lift quotient to 1           (+)
            v     cvct       conviction (inverse lift for negated head) (+)
            e     cvctdiff   absolute difference of conviction to 1     (+)
            r     cvctquot   difference of conviction quotient to 1     (+)
            k     cprob      conditional probability ratio              (+)
            j     import     importance (binary log. of prob. ratio)    (+)
            z     cert       certainty factor (relative conf. change)   (+)
            n     chi2       normalized chi^2 measure                   (+)
            p     chi2pval   p-value from (unnormalized) chi^2 measure  (-)
            y     yates      normalized chi^2 with Yates' correction    (+)
            t     yatespval  p-value from Yates-corrected chi^2 measure (-)
            i     info       information difference to prior            (+)
            g     infopval   p-value from G statistic/info. difference  (-)
            f     fetprob    Fisher's exact test (table probability)    (-)
            h     fetchi2    Fisher's exact test (chi^2 measure)        (-)
            m     fetinfo    Fisher's exact test (mutual information)   (-)
            s     fetsupp    Fisher's exact test (support)              (-)
            Measures marked with (+) must meet or exceed the threshold,
            measures marked with (-) must not exceed the threshold
            in order for the item set to be reported.
    thresh  threshold for evaluation measure       (default: 10%)
'''


def get_info_rules(rule):
    '''
    Compute the caracteristics of a rule.

    Parameters:
    - rule, tuple.
      Tuple is under the form:
      ('modality', (item1, item2), (rule modality size, rule_size,
        number of samples with the modality of interest in the whole,
        lift value of the rule, total number of samples in the whole dataset)
      E.g.: ('l-0', ('v1-2',), (35, 35, 50, 3.0, 150)) where 'l-0' is

    Output:
    - info, list. List of the caracteristics of the rule. Under the form:
          [[index of the column(s)]
          [values]
          rule size
          rule modality
          rule modality size
          rule purity
          rule z-score
          rule type]
    '''

    var = []
    val = []

    for var_val in rule[1]:
        var.append(int(var_val.split('-')[0][1:]))
        val.append([int(var_val.split('-')[1]), int(var_val.split('-')[1])])

    modality = int(rule[0].split('-')[1])
    rule_mod_size = rule[2][0]
    rule_size = rule[2][1]
    # Proportion of subject with the modality in the dataset
    base_purity = rule[2][2]/float(rule[2][4])
    # Proportion of subject with the modality in the rule
    rule_purity = rule_mod_size / float(rule_size)

    if np.sqrt(base_purity * (1.-base_purity)) == 0:
        zscore = 0
    else:
        zscore = np.sqrt(rule_size)*((rule_purity-base_purity)/np.sqrt(
            base_purity*(1.-base_purity)))

    info = [
        var, val, rule_size, modality,
        rule_mod_size, rule_purity, zscore, ['c']*len(var)]

    return info


def get_association_rules(
        x, y, asr_params={'supp': -10, 'conf': 0., 'thresh': 100},
        z_score_threshold=1.96):
    '''
    Get the matrix of rules generated using the association rules algorithm.

    Parameters:
    - x, pandas.DataFrame, shape = [n_samples, n_features]
      Training vector, where n_samples is the number of samples and
      n_features is the number of features.
    - y, pandas.Series, shape = [n_samples]
      Target vector relative to X.
    - asr_params, dict. Parameters to used for the association rules
      algorithm. See package pyfim for the parameters that can be used.
    - z_score_threshold, float. The threshold to use for the z-score parameter.

    Output:
    - return the relevant rules under the form of a np.array.
      shape (n_rules, n_rule_caracteristics) where n_rules is the number of
      relevant rules and n_rule_caracteristics the number of
      caracteristics of the rules(equal to 8).
    '''
    # Confidence is an indication of how often the rule has been found to be
    # true.
    # The confidence value of a rule is the proportion of the transactions that
    # contains X which also contains Y.
    # {conf (X-> Y)=supp(X U Y)/supp(X).

    data = pd.DataFrame(x, dtype=np.int32)
    data['target'] = y
    data['target'] = data['target'].astype(np.int32)
    data = data.applymap(str)
    data['target'] = data['target'].apply(lambda x: 'l' + '-' + x)

    for col in list(set(data.columns.tolist()) - set(['target'])):
        data[col] = data[col].apply(lambda x: 'v' + str(col) + '-' + x)

    rules = []

    for c in np.unique(data['target']):
        rules_ = fim.arules(
            data.values.tolist(),
            report='(abhlQ',
            # ( => combine values in a tuple (must be first character)
            # a => absolute item set  support (number of transactions)
            #   => rule modality size
            # b => absolute body set  support (number of transactions)
            #   => rule size
            # h => absolute head item support (number of transactions)
            #   => number of samples with the modality of interest in the whole
            #      dataset
            # l => lift value of a rule (confidence/prior)
            # Q => support of the empty set (total number of transactions)
            #   => total number of samples in the whole dataset
            # Results are under the form:
            # ('modality', (item1, item2), (a, b, h, l, Q)
            # E.g.: ('l-0', ('v1-2',), (35, 35, 50, 3.0, 150))
            eval='l',
            appear={None: 'in', c: 'out'},
            **asr_params)
        rules.extend(list(map(lambda r: get_info_rules(r), rules_)))

    rules = np.array(rules)

    rules = rules[rules[:, 6] >= z_score_threshold]

    return rules
