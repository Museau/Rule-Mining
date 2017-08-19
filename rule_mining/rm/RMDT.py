# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Author:
# - Margaux Luck
#   <margaux.luck@institut-hypercube.org>, <margaux.luck@gmail.com>
# Language: python2.7


import numpy as np
import pydotplus

from collections import Counter
from collections import defaultdict

from rule_mining.rm.utils import key_with_max_val

from sklearn import tree

from math import ceil


def get_decision_tree_rules(path, x, y, dt_params={}):
    '''
    Get the matrix of rules generated using the association rules algorithm.

    Parameters:
    - path, string. Where the figure of the decision tree used for the rule
      generation must be saved.
    - x, pandas.DataFrame, shape = [n_samples, n_features]
      Training vector, where n_samples is the number of samples and
      n_features is the number of features.
    - y, pandas.Series, shape = [n_samples]
      Target vector relative to X.
    - dt_params, dict. Parameters to used for the decision tree
      algorithm. See package sklearn.tree.DecisionTreeClassifier for the
      parameters that can be used..

    Output:
    - return the relevant rules under the form of a np.array.
      shape (n_rules, n_rule_caracteristics) where n_rules is the number of
      relevant rules and n_rule_caracteristics the number of
      caracteristics of the rules(equal to 8).
    '''

    # Compute metadata
    base_mod_sizes = Counter(y)
    base_size = float(len(y))
    base_purities = {
        k: v / base_size for k, v in base_mod_sizes.iteritems()}

    # Fit decision tree classifier
    dt = tree.DecisionTreeClassifier(**dt_params)
    clf = dt.fit(x, y)

    feature_names = range(x.shape[1])

    # Plot the tree
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(path + "decision_tree_rules.pdf")

    n_nodes = clf.tree_.node_count
    left = clf.tree_.children_left
    right = clf.tree_.children_right
    threshold = clf.tree_.threshold
    features = clf.tree_.feature
    value = clf.tree_.value[:, 0]
    # get idexes of child nodes
    idx = np.argwhere(left == -1).flatten()

    def recurse(left, right, child, lineage=None):

        if lineage is None:
            lineage = [child]

        if child in left:
            parent = np.where(left == child)[0].item()
            split = 'l'

        else:
            parent = np.where(right == child)[0].item()
            split = 'r'

        lineage.append((
            parent, split, threshold[parent], feature_names[features[parent]]))

        if parent == 0:
            lineage.reverse()
            return lineage

        else:
            return recurse(left, right, parent, lineage)

    rules = []

    if n_nodes > 1:
        for child in idx:
            rule = defaultdict(list)
            # recurse(left, right, child) is a list of tuples.
            # Each tuple is under the form: (parent=index of the parent, split,
            # threshold[parent], [child])
            for node in recurse(left, right, child)[:-1]:
                if node[1] == 'l':
                    rule[node[3]].append(('<=', node[2]))

                else:
                    rule[node[3]].append(('>', node[2]))

            # Decode rule
            idx_var = list(np.sort(rule.keys()))
            inter_ = []

            for idx in idx_var:
                l = rule[idx]

                lower_limit = [item[1] for item in l if item[0] == '>']

                if lower_limit:
                    lower_limit = np.min(lower_limit) + 0.1

                else:
                    lower_limit = np.min(x[:, idx])

                upper_limit = [item[1] for item in l if item[0] == '<=']

                if upper_limit:
                    upper_limit = np.max(upper_limit)

                else:
                    upper_limit = np.max(x[:, idx])

                inter_.append([int(ceil(lower_limit)), int(upper_limit)])

            rule_mod_sizes = {k: v for k, v in enumerate(value[child])}
            rule_size = float(sum(rule_mod_sizes.values()))
            rule_purities = {
                    k: v / rule_size
                    for k, v in rule_mod_sizes.iteritems()}

            def z_score(rule_size, rule_purity, base_purity):
                '''
                Compute the z-score of a rule regarding a particular modality.

                Parameters:
                - rule_size, int. (i.e., the number of suject in the rule)
                - rule_purity, float. Rule purity.
                - base_purity, float. Base purity.

                Output:
                - z-score, float. Rule z-score.
                '''
                return np.sqrt(rule_size) * (
                        rule_purity-base_purity)/np.sqrt(
                            base_purity * (1-base_purity))

            rule_z_scores = {k: z_score(
                rule_size,
                rule_purities[k],
                base_purities[k]) for k, v in rule_purities.iteritems()}
            rule_interest_mod = key_with_max_val(rule_z_scores)

            rule_type = ['c'] * len(idx_var)

            rule = [
                idx_var,
                inter_,
                rule_size,
                rule_interest_mod,
                rule_mod_sizes[rule_interest_mod],
                rule_purities[rule_interest_mod],
                rule_z_scores[rule_interest_mod],
                rule_type]

            rules.append(rule)

    else:
        print 'No rules, only a root ...'

    rules = np.array(rules)
    return rules
