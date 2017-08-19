# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Author:
# - Cecilia Damon
#   <cecilia.damon@institut-hypercube.org>, <cecilia.damon@gmail.com>
# - Margaux Luck
#   <margaux.luck@institut-hypercube.org>, <margaux.luck@gmail.com>
# Language: python2.7


import numpy as np
np.random.seed(1337)  # make the results reproductible
from math import floor


'''
Synthetic dataset for classification rules

x = (x1, x2, x3, x4)
x1, x2 ~ U[0, 1]
x3 = {0, 1}
x4 = {blue -1, white 0, red 1}

y = 0, 1 ou 2

y = 0 <=  r01 = {x4: red, x3: 1};
          r02 = {x4: red, x3: 0, x2 <= 0.5};
          r03 = {x4: blue or white, x1 >= 0.7, x3: 0, x2 > 0.2};
          r04 = {x4: white, x1: ]0.5, 0.7[}

y = 1 <=  r11 = {x4: red, x3: 0, x2 > 0.5};
          r12 = {x4: blue ou white, x1 >= 0.7, x3: 0, x2 <= 0.2};
          r13 = {x4: white, x1 <= 0.5}

y = 2 <= r21 = {x4: blue ou white, x1 >= 0.7, x3: 1}

                                    [x4-red]
                            |                       |
                        [x1>=0.7]                [x3=0]
                    |                 |        |        |
                 [x1-blue]         [x3=1]    [y=0]  [x2<=0.5]
               |         |         |    |        |     |
           [x1<=0.5]   [y=2]   [x2<=2][y=2]    [y=1] [y=0]
            |     |             |   |
          [y=0] [y=1]         [y=0][y=1]
 '''


class DecisionTreeSamples():

    def __init__(self, n, e=None):
        '''
        Initialize the synthetic dataset (X, y) based on the synthetic decision
        tree.

        Parameters:
        - n: number of samples to generate
        - e: noisy or impurity degree to integrate in the rules under the form
          of classification error expressed in percentage, None by default.
          The same or a different amount of noise can be introduced for each
          rule:
          *To introduce different amount of noise for each rule, e must be
          as a dictionary with {'r01': e1, 'r02': e2, 'r03': e3, 'r04': e04,
          'r11': e11, 'r12': e12, 'r13': e13, 'r21': e21} where 'r01', 'r02',
          etc. correspond to the name of the rules (see above) and e1, e2, etc.
          the noise specific to each one of these rules.
          *To introfuce the same amount of noise, e must be a int or a float.

        '''
        # Generate synthetic dataset (X, y)
        X = np.zeros((n, 4))
        X[:, 0] = np.random.uniform(0, 1, n)
        X[:, 1] = np.random.uniform(0, 1, n)
        X[:, 2] = np.random.randint(0, 2, n)
        X[:, 3] = np.random.randint(-1, 2, n)

        # Replace y values by the labels {0, 1, 2} according to the rules.
        y = np.zeros(n)
        indr13 = np.where(np.logical_and(X[:, 3] == 0, X[:, 0] <= 0.5))[0]
        indr11 = np.where(np.logical_and.reduce((
            X[:, 3] == 1, X[:, 2] == 0, X[:, 1] > 0.5)))[0]
        indr12 = np.where(np.logical_and.reduce((
            np.in1d(X[:, 3], [-1, 0]),
            X[:, 2] == 0, X[:, 1] <= 0.2, X[:, 0] >= 0.7)))[0]
        indr21 = np.where(np.logical_and.reduce((
            np.in1d(X[:, 3], [-1, 0]),
            X[:, 2] == 1, X[:, 0] >= 0.7)))[0]
        indr01 = np.where(np.logical_and(X[:, 3] == 1, X[:, 2] == 1))[0]
        indr02 = np.where(np.logical_and.reduce((
            X[:, 3] == 1, X[:, 2] == 0, X[:, 1] <= 0.5)))[0]
        indr03 = np.where(np.logical_and.reduce((
            np.in1d(X[:, 3], [-1, 0]), X[:, 2] == 0,
            X[:, 1] > 0.2, X[:, 0] >= 0.7)))[0]
        indr04 = np.where(np.logical_and.reduce((
            X[:, 3] == 0, X[:, 0] < 0.7, X[:, 0] > 0.5)))[0]
        y[indr13] = 1
        y[indr11] = 1
        y[indr12] = 1
        y[indr21] = 2

        if e is not None:

            # Introduce some noise in the rules
            if isinstance(e, (int, float)):
                r01 = r02 = r03 = r04 = r11 = r12 = r13 = r21 = e

            else:
                r01 = e['r01']
                r02 = e['r02']
                r03 = e['r03']
                r04 = e['r04']
                r11 = e['r11']
                r12 = e['r12']
                r13 = e['r13']
                r21 = e['r21']

            y[indr01[np.random.randint(
                0, len(indr01), int(floor(len(indr01)*r01/100.)))]] =\
                np.random.choice([1, 2], int(floor(len(indr01)*r01/100.)))
            y[indr02[np.random.randint(
                0, len(indr02), int(floor(len(indr02)*r02/100.)))]] =\
                np.random.choice([1, 2], int(floor(len(indr02)*r02/100.)))
            y[indr03[np.random.randint(
                0, len(indr03), int(floor(len(indr03)*r03/100.)))]] =\
                np.random.choice([1, 2], int(floor(len(indr03)*r03/100.)))
            y[indr04[np.random.randint(
                0, len(indr04), int(floor(len(indr04)*r04/100.)))]] =\
                np.random.choice([1, 2], int(floor(len(indr04)*r04/100.)))
            y[indr13[np.random.randint(
                0, len(indr13), int(floor(len(indr13)*r13/100.)))]] =\
                np.random.choice([0, 2], int(floor(len(indr13)*r13/100.)))
            y[indr11[np.random.randint(
                0, len(indr11), int(floor(len(indr11)*r11/100.)))]] =\
                np.random.choice([0, 2], int(floor(len(indr11)*r11/100.)))
            y[indr12[np.random.randint(
                0, len(indr12), int(floor(len(indr12)*r12/100.)))]] =\
                np.random.choice([0, 2], int(floor(len(indr12)*r12/100.)))
            y[indr21[np.random.randint(
                0, len(indr21), int(floor(len(indr21)*r21/100.)))]] =\
                np.random.choice([0, 1], int(floor(len(indr21)*r21/100.)))

        self.X = X
        self.y = y
