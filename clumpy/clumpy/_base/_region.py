#!/usr/bin/env python3
# -*- coding: utf-8 -*

from . import Land


class Region:
    """
    Define a region.

    Parameters
    ----------
    label : str
        The region's label. It should be unique.

    value : int
        The region's value.

    verbose : int, default=0
        Verbosity level.

    verbose_heading_level : int, default=1
        Verbose heading level for markdown titles. If ``0``, no markdown title are printed.
    """

    def __init__(
        self,
        label,
        value,
        verbose=0,
        verbose_heading_level=1,
    ):
        self.label = label
        self.value = value
        self.lands: list[Land] = []
        self.verbose = verbose
        self.verbose_heading_level = verbose_heading_level

    def __repr__(self):
        return "Region(" + self.label + ")"

    def get_lands_states(self):
        return [land.state for land in self.lands]

    def add_land(self, land):
        if land not in self.lands:
            if land.state not in self.get_lands_states():
                self.lands.append(land)
            else:
                Warning("The land value is already in.")
        else:
            Warning("The land is already in.")

        return self

    def add_lands(self, lands):
        self.lands = lands
        return self

    def get_land_by_state(self, state):
        states = self.get_lands_states()
        return self.lands[states.index(state)]

    def get_land(self, info):
        if type(info) is int:
            return self.get_land_by_state(info)
