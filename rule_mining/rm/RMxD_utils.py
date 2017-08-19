# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Author:
# - Margaux Luck
#   <margaux.luck@institut-hypercube.org>, <margaux.luck@gmail.com>
# - Cecilia Damon
#   <cecilia.damon@institut-hypercube.org>, <cecilia.damon@gmail.com>
# Language: python2.7


import numpy as np

from itertools import combinations
from operator import itemgetter
from itertools import groupby, chain, product


# 1D rule-mining algorithm

def get_intervals_orm_1d_c(n_bin):
    '''
    Construct all possible candidate rule combinations made from adjacent
    bins in a particular integer ordered feature.

    Parameters:
    - n_bin: int, number of distinct possibilities for integer bin
      values in a particular "integer continuous" feature.
      The minimum value of the feature must be 0 and the maximum value
      n_bin.
      e.g., a n_bin value of 9 means the feature can take the integer
      values 0 to 9 inclusive.

    Output:
    - intervals: list, contains closed intervals of the form [min, max]
      corresponding to all possible rules.
    '''

    intervals = []

    for i in range(0, n_bin):
        intervals.append([i, i])

        for j in range(i+1, n_bin+1):
            intervals.append([i, j])

    if n_bin > 1:
        intervals.append([j, j])

    return intervals


def get_intervals_orm_1d_d(n_bin):
    '''
    Construct all possible candidate rule combinations made from
    bins in a particular integer discrete feature.

    Parameters:
    - n_bin: int, number of distinct possibilities for integer bin
      values in a particular discrete feature.
      The minimum value of the feature must be 0 and the maximum value
      n_bin.
      e.g., a n_bin value of 9 means the feature can take the integer
      values 0 to 9 inclusive.

    Output:
    - intervals: list, contains tuples of the form [(1, ), (2, ), (3, ),
      (1, 2), (1, 3), ..., (1, 2, 3), ...] corresponding to all possible rules.
    '''

    v = range(0, (n_bin+1))

    intervals = []

    for i in range(1, (n_bin+2)):
        intervals = intervals + list(combinations(v, i))

    return intervals


# 2D rule-mining algorithm


def get_cont_intervals(values_set):
    '''
    Return list of sets of continuous values from a set of values

    Parameters:
    - values_set: set of values

    Output:
    - list of sets of continuous values
    '''

    row_ranges = []

    for k, g in groupby(enumerate(values_set), lambda x: x[0] - x[1]):
        group = list(map(itemgetter(1), g))
        row_ranges.append([int(group[0]), int(group[-1])])

    return row_ranges


def get_discret_intervals(values_set):
    '''
    Return list of sets of discrete values from a set of values

    Parameters:
    - values_set: set of values

    Output:
    - list of sets of continuous values
    '''
    nbval = len(values_set)
    dranges = []

    if nbval >= 2:
        dranges.extend(
            map(lambda tup: list(tup)
                if list(tup)[0] != list(tup)[1]
                else [list(tup)[0]], product(values_set, repeat=2)))

        if nbval > 2:
            for l in range(3, nbval+1):
                dranges.extend(
                    map(lambda tup: list(tup), combinations(values_set, l)))

    else:
        dranges.append([values_set[0], values_set[0]])

    return dranges


def subsequences(iterable, length):
    '''
    Search all the subsets of continuous values with a given length

    Parameters:
    - iterable: tuple of 2 values corresponding to a range of continuous
     values
    - length: length value

    Output:
    - return:  subsets of continuous values of length 'length'
    '''

    subseq = [[iterable[i], iterable[i + length - 1]] for i in xrange(
        len(iterable) - length + 1)]

    return subseq


def get_intervals_orm_2d_cc(x, idx_var):
    '''
    Construct all possible candidate rule combinations made from adjacent bins
    in a particular combinations of two integer ordered feature.

    Parameters:
    - x, np.array. shape = [n_samples, n_features]
      Training vector, where n_samples is the number of samples and
      n_features is the number of features.
    - idx_var, tuple. Contain the index of the two integer ordered features
      to consider

    Output:
    - intervals, list of list. Contains closed intervals of the form
      [[min, max], [min, max]] corresponding to all possible rules. The first
      inter list correspond to the condition on the first feature of idx_var
      and the second inter list to the condition on the second feature of
      idx_var.
    '''

    x_var = x[:, idx_var]
    x_var = x_var[np.isfinite(x_var).all(axis=1)]
    x_var = np.asarray(x_var, dtype=int)

    values = list(set([tuple(i) for i in x_var]))

    sort_values = sorted(values)

    # Find continuous numbers in a list for each row and column
    # sets of continues values for the first variable of the couple of

    dict_row_colval = {}

    for key, val in groupby(sort_values, itemgetter(0)):
        dict_row_colval[key] = list(map(itemgetter(1), val))

    row_ranges = get_cont_intervals(dict_row_colval.keys())

    row_subranges = []

    # For each set of continuous values generated for variable1
    for rr in row_ranges:
        # extract all the subsets withing the set of continuous values 'rr'
        # (i.e. row_subranges)
        row_subranges.extend(map(
            lambda l: subsequences(range(rr[0], rr[1] + 1), l),
            range(1, rr[-1] - rr[0] + 2)))
        # row_subranges.extend(map(
        #     lambda l: subsequences(range(int(rr[0]), int(rr[1]) + 1), l),
        #     range(1, int(rr[-1]) + 2)))
        # row_subranges = list(chain.from_iterable(row_subranges))

    row_subranges = list(chain.from_iterable(row_subranges))

    intervals = []

    for rr_subseq in row_subranges:

        col_subranges = []
        # recover all the values of the second variable of the couple 'var'
        # associated to values in rr_subseq
        # and produce the corresponding set of continues values from the
        # set of recovered values (i.e. col_ranges)
        # col_ranges = get_cont_intervals(
        #     list(set(chain.from_iterable(
        #         [dict_row_colval[i] for i in range(
        #             int(rr_subseq[0]), int(rr_subseq[1]) + 1)]))))
        col_ranges = get_cont_intervals(
            list(set(chain.from_iterable(
                [dict_row_colval[i] for i in range(
                    rr_subseq[0], rr_subseq[1] + 1)]))))

        # extract all the subsets withing the set of continuous values
        # 'col_ranges' (i.e. col_subranges)
        for cr in col_ranges:
            # col_subranges.extend(
            #     map(lambda l: subsequences(
            #         range(int(cr[0]), int(cr[1]) + 1), l),
            #         range(1, int(cr[-1]) + 2)))
            col_subranges.extend(
                map(lambda l: subsequences(
                    range(cr[0], cr[1] + 1), l),
                    range(1, cr[-1]-cr[0] + 2)))

        col_subranges = list(chain.from_iterable(col_subranges))

        # Combine the subsets row_subranges and col_subranges to produce
        # all the possible rules to explore
        intervals.extend(
            [tup[0], tup[1]]
            for tup in product([rr_subseq], col_subranges))

    return intervals


def get_intervals_orm_2d_dd(x, idx_var):
    '''
    Construct all possible candidate rule combinations made from adjacent bins
    in a particular combinations of two integer ordered feature.

    Parameters:
    - x, np.array. shape = [n_samples, n_features]
      Training vector, where n_samples is the number of samples and
      n_features is the number of features.
    - idx_var, tuple. Contain the index of the two integer ordered features
      to consider

    Output:
    - intervals, list of list. Contains closed intervals of the form
      [[min, max], [min, max]] corresponding to all possible rules. The first
      inter list correspond to the condition on the first feature of idx_var
      and the second inter list to the condition on the second feature of
      idx_var.
    '''

    x_var = x[:, idx_var]
    x_var = x_var[np.isfinite(x_var).all(axis=1)]
    x_var = np.asarray(x_var, dtype=int)

    values = list(set([tuple(i) for i in x_var]))

    sort_values = sorted(values)

    # Find continuous numbers in a list for each row and column
    # sets of continues values for the first variable of the couple of

    dict_row_colval = {}

    for key, val in groupby(sort_values, itemgetter(0)):
        dict_row_colval[key] = list(map(itemgetter(1), val))

    row_ranges = get_discret_intervals(dict_row_colval.keys())

    intervals = []

    for rr_subseq in row_ranges:
        # recover all the values of the second variable of the couple 'var'
        # associated to values in rr_subseq
        # and produce the corresponding set of discrete values from the
        # set of recovered values (i.e. col_ranges)
        if len(rr_subseq) == 1:
            rr_subseq = [rr_subseq[0], rr_subseq[0]]
        col_ranges = get_discret_intervals(
            list(set(chain.from_iterable(
                [dict_row_colval[i] for i in rr_subseq]))))
        # Combine the subsets row_subranges and col_ranges to produce
        # all the possible rules to explore
        intervals.extend(
            [tup[0], tup[1]]
            for tup in product([rr_subseq], col_ranges))

    return intervals


def get_intervals_orm_2d_cd(x, idx_var):
    '''
    Construct all possible candidate rule combinations made from adjacent bins
    in a particular combinations of two integer ordered feature.

    Parameters:
    - x, np.array. shape = [n_samples, n_features]
      Training vector, where n_samples is the number of samples and
      n_features is the number of features.
    - idx_var, tuple. Contain the index of the two integer ordered features
      to consider

    Output:
    - intervals, list of list. Contains closed intervals of the form
      [[min, max], [min, max]] corresponding to all possible rules. The first
      inter list correspond to the condition on the first feature of idx_var
      and the second inter list to the condition on the second feature of
      idx_var.
    '''

    x_var = x[:, idx_var]
    x_var = x_var[np.isfinite(x_var).all(axis=1)]
    x_var = np.asarray(x_var, dtype=int)

    values = list(set([tuple(i) for i in x_var]))

    sort_values = sorted(values)

    # Find continuous numbers in a list for each row and column
    # sets of continues values for the first variable of the couple of

    dict_row_colval = {}

    for key, val in groupby(sort_values, itemgetter(0)):
        dict_row_colval[key] = list(map(itemgetter(1), val))

    row_ranges = get_cont_intervals(dict_row_colval.keys())

    subs = subsequences
    row_subranges = []

    # For each set of continuous values generated for variable1
    for rr in row_ranges:
        # extract all the subsets withing the set of continuous values 'rr'
        # (i.e. row_subranges)
        row_subranges.extend(map(
            lambda l: subs(
                # range(rr[0], rr[1] + 1), l), range(1, rr[-1]+2)))
                range(rr[0], rr[1] + 1), l), range(1, rr[-1] - rr[0] + 2)))
        # row_subranges = list(chain.from_iterable(row_subranges))

    row_subranges = list(chain.from_iterable(row_subranges))

    intervals = []

    for rr_subseq in row_subranges:
        # recover all the values of the second variable of the couple 'var'
        # associated to values in rr_subseq
        # and produce the corresponding set of discrete values from the
        # set of recovered values (i.e. col_ranges)
        col_ranges = get_discret_intervals(
            list(set(chain.from_iterable(
                [dict_row_colval[i] for i in range(
                    rr_subseq[0], rr_subseq[1] + 1)]))))
        # Combine the subsets row_subranges and col_ranges to produce
        # all the possible rules to explore
        intervals.extend(
            [tup[0], tup[1]]
            for tup in product([rr_subseq], col_ranges))

    return intervals
