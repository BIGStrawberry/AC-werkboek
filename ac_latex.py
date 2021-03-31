#!/usr/bin/env python

"""Diverse wiskundige structuren weergeven in LaTeX in Jupyter Notebook."""

__author__      = "Brian van der Bijl"
__copyright__   = "Copyright 2020, Hogeschool Utrecht"

import re
from fractions import Fraction
from IPython.display import display, Math, Markdown

def show_num(num):
    return re.compile(r"\.(?!\d)").sub("\1", num)


def latex_formula(form, details=True):
    latex = form.simplify().to_latex(outer=True)
    if latex:
        display(Math(latex))
        if details:
            display(Markdown("<details><pre>$" + latex + "$</pre></details>"))


def latex_bmatrix(matrix, label=None, details=True): # Gebaseerd op https://stackoverflow.com/questions/17129290/numpy-2d-and-1d-array-to-latex-bmatrix
    if len(matrix.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(matrix).replace("[", "").replace("]", "").splitlines()
    if label:
        result = [label + " = "]
    else:
        result = [""]
    result += [r"\begin{bmatrix}"]
    result += ["  " + " & ".join(map(show_num, l.split())) + r"\\" for l in lines]
    result +=  [r"\end{bmatrix}"]
    display(Math("\n".join(result)))
    if details:
        display(Markdown("<details><pre>$" + " ".join(result) + "$</pre></details>"))


def latex_amatrix(matrix, labels=None, details=True):
    if len(matrix.shape) > 2:
        raise ValueError('array can at most display two dimensions')
    lines = str(matrix).replace("[", "").replace("]", "").splitlines()
    if labels and len(labels) == 2:
        result = [r"(\mathbf{" + labels[0] + r"} | \vec " + labels[1] + ") = "]
    else:
        result = [""]
    result += [r"\left[\begin{array}{ccc|c}"]
    result += ["  " + " & ".join(map(show_num, l.split())) + r"\\" for l in lines]
    result +=  [r"\end{array}\right]"]
    display(Math("\n".join(result)))
    if details:
        display(Markdown("<details><pre>$" + " ".join(result) + "$</pre></details>"))


def latex_msquare(matrix, details=True):
    if matrix.shape != (3,3):
        raise ValueError('Geen magisch vierkant')
    lines = str(matrix).replace("[", "").replace("]", "").splitlines()
    result = [r"\begin{array}{|c|c|c|}\hline"]
    result += ["  " + " & ".join(map(show_num, l.split())) + r"\\\hline" for l in lines]
    result +=  [r"\end{array}"]
    display(Math("\n".join(result)))
    if details:
        display(Markdown("<details><pre>$" + " ".join(result) + "$</pre></details>"))


def latex_ratio(rational):
    """Helper functie om breuken naar LaTeX te converteren; getallen worden alleen naar string
       geconverteerd."""
    if isinstance(rational, int):
        return str(rational)
    elif isinstance(rational, Fraction):
        if rational.numerator == rational.denominator:
            return "1"
        elif rational.numerator > 0:
            return r"\frac{" + str(abs(rational.numerator)) + "}{" + str(rational.denominator) + "}"
        else:
            return r"-\frac{" + str(abs(rational.numerator)) + "}{" + str(rational.denominator) + "}"
    else:
        numerator, denominator = rational.as_integer_ratio() # Nul buiten de breuk halen
        return ("-" if numerator < 0 else "") + r"\frac{" + str(abs(numerator)) + "}{" + str(denominator) + "}"


def latex_polynomial(poly, details=True):
    terms, label, var, primes = poly # Bind parameters uit tuple

    def power(exp):
        """Print een term (e.g. x^2). x^1 is gewoon x, x^0 is 1, maar n × 1 is gewoon n dus verberg de 1.
           In alle andere gevallen wordt de variabele met het juiste exponent opgeleverd."""
        if exp == 1:
            return var
        elif exp == 0:
            return ""
        else:
            return var + r"^{" + latex_ratio(exp) + "}"


    # Print f(x) met het juiste aantal primes
    if primes < 1:
        result = label.upper() + "(" + var + ") = "
    elif primes == 0:
        result = label + "(" + var + ") = "
    else:
        result = label + "^{" + r"\prime"*primes + "}" + "(" + var + ") = "
    first = True # Na de eerste moet er "+" tussen de termen komen

    for key, val in reversed(sorted(terms.items())): # Voor iedere term, van groot (hoog exponent) naar klein
        if val > 0 and not first: # Koppel met een plus, tenzij het de eerste term is
            result += "+"
        elif val < 0: # Koppel met een min als de term negatief is, ook de eerste term
            result += "-"

        if val != 0: # Zet first op False na de eerste keer
            first = False

        if key == 0:
            result += str(val)
        elif abs(val) == 1: # Print x in plaats van 1x en -x in plaats van -1x
            result += str(power(key))
        elif val != 0: # Print iedere term die niet 0 of 1 is op de gebruikelijke manier, zonder min want die staat
            result += latex_ratio(abs(val)) + str(power(key))  #   erboven al

    display(Math(result))
    if details:
        display(Markdown("<details><pre>$" + result + "$</pre></details>"))
