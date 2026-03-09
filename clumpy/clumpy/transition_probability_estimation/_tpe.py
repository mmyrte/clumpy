#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 14:03:33 2021

@author: frem
"""

# READ ME
# Transition Probability Estimators (TPE) must have these methods :
#   - fit()
#   - transition_probability()
#   - _check()


class TransitionProbabilityEstimator:
    """
    Transition probability estimator base class.
    """

    def __init__(self, verbose=0):
        self.verbose = verbose
