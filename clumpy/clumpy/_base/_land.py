#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

from ..allocation import _methods as allocation_methods
from ..ev_selection import EVSelectors
from ..ev_selection import _methods as ev_selection_methods
from ..patch import _methods as patch_methods
from ..transition_probability_estimation import (
    _methods as transition_probability_estimation_methods,
)

logger = logging.getLogger("clumpy")


class Land:
    """
    Land object which refers to a given initial state.

    Parameters
    ----------
    state : State
        The initial state of this land.

    final_states : list
        List of possible final states.

    transition_probability_estimator : TransitionProbabilityEstimator or str, default=None
        Transition probability estimator. If a string, looked up from registered methods.
        If ``None``, fit, transition_probabilities and allocate are not available.

    ev_selectors : EVSelectors or str, default=None
        Explanatory variable selectors.

    allocator : Allocator or str, default=None
        Allocator. If ``None``, the allocation is not available.

    patcher : Patcher or str or list, default=None
        Patch placement strategy.

    verbose : int, default=0
        Verbosity level.

    verbose_heading_level : int, default=1
        Verbose heading level for markdown titles. If ``0``, no markdown title are printed.
    """

    def __init__(
        self,
        state,
        final_states,
        transition_probability_estimator=None,
        ev_selectors=None,
        allocator=None,
        patcher=None,
        verbose=0,
        verbose_heading_level=1,
    ):

        self.state = state
        self.final_states = final_states

        if type(transition_probability_estimator) is str:
            self.transition_probability_estimator = (
                transition_probability_estimation_methods[
                    transition_probability_estimator
                ](verbose=verbose - 1)
            )
        else:
            self.transition_probability_estimator = transition_probability_estimator

        if type(ev_selectors) is str:
            self.ev_selectors = EVSelectors(
                selectors={
                    v: ev_selection_methods[ev_selectors]() for v in final_states
                }
            )
        else:
            self.ev_selectors = ev_selectors

        if type(allocator) is str:
            self.allocator = allocation_methods[allocator]()
        else:
            self.allocator = allocator

        if type(patcher) is str:
            self.patcher = [patch_methods[patcher]() for _v in final_states]
        else:
            self.patcher = patcher

        self.verbose = verbose
        self.verbose_heading_level = verbose_heading_level

    def __repr__(self):
        return "Land()"
