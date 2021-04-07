#!/usr/bin/env python

"""Main import module voor de toets Analytical Computing."""

__author__      = "Brian van der Bijl"
__copyright__   = "Copyright 2020, Hogeschool Utrecht"

from typing import Tuple, Dict, Union
import numpy as np
import matplotlib.pyplot as plot

Polynomial = Tuple[Dict[int, float], str, str, int]


def polynomial(terms: Union[list, dict], label: str = 'f', var: str = 'x', primes: int = 0) -> Polynomial:
    if isinstance(terms, np.ndarray):
        terms = terms.flatten().tolist()
    if not isinstance(terms, dict):
        terms = dict(enumerate(terms))
    return (terms, label, var, primes)


def plot_data_lr(data, slope=None, intercept=None):
    x_data = data.keys()
    y_data = [ data[x] for x in x_data ]
    plot.scatter(x_data, y_data)
    plot.title("Number of people who died by becoming tangled in their bedsheets from per capita cheese consumption")
    plot.xlabel("cheese consumption")
    plot.ylabel("Deaths by bedsheet-tangling")
    if slope and intercept:
        axes = plot.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = intercept + slope * x_vals
        plot.plot(x_vals, y_vals, '--')
    plot.show()
