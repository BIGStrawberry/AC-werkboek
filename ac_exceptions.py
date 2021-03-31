#!/usr/bin/env python

"""Errors in een aparte module om import-cycli te vermijden."""

__author__      = "Brian van der Bijl"
__copyright__   = "Copyright 2020, Hogeschool Utrecht"


class DimensionError(Exception):
    """Een vector of matrix heeft niet de juiste dimensies voor de gevraagde operatie."""


class NonInvertibleError(Exception):
    """Een matrix is niet inverteerbaar, omdat deze niet vierkant is of een determinant van 0 heeft."""


class VariableError(Exception):
    """Geen waarde voor de gevraagde variabele opgegeven."""
