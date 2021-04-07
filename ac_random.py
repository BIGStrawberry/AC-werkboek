#!/usr/bin/env python

"""Generen van random matrices, vectoren, etc. voor studentenopgaven lineaire algebra en calculus."""

__author__      = "Brian van der Bijl"
__copyright__   = "Copyright 2020, Hogeschool Utrecht"

import numpy as np
from IPython.display import display, Math, Markdown
from ac_latex import latex_amatrix, latex_bmatrix

class RNG:
    # pylint: disable=R0903
    class InnerRNG:
        def __init__(self, studentnr):
            self.seed = studentnr
            np.random.seed(self.seed)


        def set(self, offset):
            np.random.seed(self.seed + offset)
            return self


        def consume_entropy(self, number, lower, upper):
            np.random.randint(lower, upper, size=number)
            return self

    instance = None


    def __init__(self, studentnr=None):
        if not RNG.instance:
            try:
                RNG.instance = RNG.InnerRNG(int(studentnr))
                display(Markdown("<h2 style=\"color:#00cccc;\">Seed geïnitialiseerd.</h2>"))

            except ValueError:
                display(Markdown("<h2 style=\"color:red;\">Je bent vergeten je studentnummer in te vullen, doe dat op de eerste regel!</h2>"))
                raise ValueError("Je bent vergeten je studentnummer in te vullen, doe dat op de eerste regel!") from None

        elif studentnr:
            display(Markdown("<h2 style=\"color:orange;\">Seed reeds geïnitialiseerd.</h2>"))


    def __getattr__(self, name):
        return getattr(self.instance, name)


DEGENERATE =  1
DONT_CARE =  0
NONDEGENERATE = -1


def random_tensor(label=None, size=None, singular=DONT_CARE,\
                  interval=None, ret=False, details=True):
    # pylint: disable=R0913

    def generate_tensor(size, interval):
        if not interval:
            interval = (-20, 20)
        if size and isinstance(size, int):
            size=(size,1)
        elif not size or not isinstance(size, tuple):
            size=(np.random.randint(2,6),1)
        return np.random.randint(interval[0], interval[1], size=size)
    candidate = generate_tensor(size, interval)
    while (singular == DEGENERATE and np.linalg.det(candidate) != 0)\
       or (singular == NONDEGENERATE and np.linalg.det(candidate) == 0):
        candidate = generate_tensor(size, interval)
    latex_bmatrix(candidate, label, details=details)
    return candidate if ret else None



def random_scalar(label="", ret=False):
    scalar = np.random.randint(-10,10)
    display(Math(label + " = " + str(scalar)))
    return scalar if ret else None



def random_sys_of_eq(ret=False, details=True):
    y_vec = np.random.choice(9,3, False)
    matrix = np.random.choice(3,(3,3))
    while np.linalg.det(matrix) == 0:
        matrix = np.random.choice(3,(3,3))
    latex_amatrix(np.concatenate((matrix, np.reshape(np.linalg.det(matrix)*y_vec, (3,1))), 1)
                    .astype(int), ("A", "b"), details=details)
    return (matrix, y_vec) if ret else None



def hide_one(scalar, nbsp=False):
    if scalar == 1:
        return "\\!" if nbsp else ""
    else:
        return str(int(scalar))


def halve(integer):
    if integer % 2 == 0:
        return str(integer/2)
    else:
        return r"\frac{" + str(integer) + "}{2}"


def random_derivatives():
    # pylint: disable=C0103
    a, b, c = np.random.randint(2, 7, 3)
    text = f"Gegeven $f(x) = ({a-1}- {hide_one(b)}x)^{hide_one(c)}$, bepaal $f^\\prime(x)$"
    display(Markdown("**(a)** " + text))
    display(Markdown("<details><pre>" + text + "</pre></details>"))

    a, b, c, d = np.random.randint(2, 7, 4)
    text = f"Gegeven $g(x) = {hide_one(a-1)}x^{hide_one(b)}\\ \\text{{tan}}({hide_one(c)}x^{hide_one(d)})$, geef $g^\\prime(x)$"
    display(Markdown("**(b)** " + text))
    display(Markdown("<details><pre>" + text + "</pre></details>"))

    a, b, c, d = np.random.randint(2, 7, 4)
    text = f"Gegeven $h(x) = \\text{{log}}_{a}({b-1}x-{c}x^{d})$, geef $h^\\prime(x)$"
    display(Markdown("**(c)** " + text))
    display(Markdown("<details><pre>" + text + "</pre></details>"))
    
def random_double_derivatives():
    a, b = np.random.randint(2, 7, 2)
    text = f"Gegeven $k(x) = \\frac{{{a}}}{{x^{b}}}$, geef $k^{{\\prime\\prime}}(x)$"
    display(Markdown("**(a)** " + text))
    display(Markdown("<details><pre>" + text + "</pre></details>"))

    a, b = np.random.randint(2, 7, 2)
    text = f"Gegeven $\\frac{{dy}}{{dx}} = x^{a} - {hide_one(b-1)}y$, geef $\\frac{{d^2y}}{{dx^2}}$"
    display(Markdown("**(b)** " + text))
    display(Markdown("<details><pre>" + text + "</pre></details>"))

def random_implicit_diff():
    a, b, c, d = np.random.randint(2, 7, 4)
    text = f"Gegeven ${hide_one(a-1)}x^3y - {hide_one(b-1)}x^2 + {hide_one(c-1)}y^4 = {2*d}$, geef $\\frac{{dy}}{{dx}}$"
    display(Markdown("**(a)** " + text))
    display(Markdown("<details><pre>" + text + "</pre></details>"))


def random_integrals():
    # pylint: disable=C0103
    a, b = np.random.randint(2, 7, 2)
    text = f"$$\\int \\sqrt[{a}]x^{b}\\ dx$$"
    display(Markdown("**(a)** Bereken " + text))
    display(Markdown("<details><pre>" + text + "</pre></details>"))

    a, b, c, d, e = np.random.randint(3, 9, 5)
    a = (a % 3) + 2
    b = (b % 3) + 1
    a = a if a != b else a+1
    a, b = max(a, b), min(a, b)
    e = e if e != d else d+e
    d, e = min(d,e), max(d,e)
    text = f"$$\\int_{{{d}}}^{{{e}}} {a*b}x^{{{a-1}}} - {b*c}x^{{{b-1}}}\\ dx$$"
    display(Markdown("**(b)** Bereken" + text))
    display(Markdown("<details><pre>" + text + "</pre></details>"))


def random_integrals_extra():
    # pylint: disable=C0103
    a, b, c = np.random.randint(2, 7, 3)
    text = f"$$\\int_{min(a,b)}^{max(a,b)+2} {c}e^x\\ dx$$"
    display(Markdown("**(a)** Bereken " + text))
    display(Markdown("<details><pre>" + text + "</pre></details>"))

    a, b, c = np.random.randint(2, 5, 3)
    text = f"$$\\int_{{{halve(min(a,b))}\\pi}}^{{{halve(max(a,b)+2)}\\pi}} -{c} \\text{{sin}}(x)\\ dx$$"
    display(Markdown("**(b)** Bereken " + text))
    display(Markdown("<details><pre>" + text + "</pre></details>"))

    a, b, c, d = np.random.randint(3, 9, 4)
    text = f"$$\\int ({a*b}x^{b-1})({a}x^{b}+{c})^{d}\\ dx$$"
    display(Markdown("**(c)** Bereken " + text))
    display(Markdown("<details><pre>" + text + "</pre></details>"))

    a, b, c = np.random.randint(2, 7, 3)
    text = f"$$\\int ({a}x^{b})\\text{{log}}_{c}(x)\\ dx$$"
    display(Markdown("**(d)** Bereken " + text))
    display(Markdown("<details><pre>" + text + "</pre></details>"))


def random_de():
    # pylint: disable=C0103
    a, b, c, d, e, f, g, h = np.random.randint(2,9,8)
    b = int(b/2)
    c = c*d
    d = 3*e*f
    e = 2*(g-4)
    f = h*2 -  1
    deriv = f"f^\\prime(x) = {a*b}x^{b-1}+{c}e^x"
    val = f"f({e}) = {a*(e**b)-d}+{c}e^{{{e}}}"
    display(Markdown(f"Vind $f({f})$ gegeven de volgende afgeleidde en waarde:\n\n$${deriv},\\ {val}$$"))
    display(Markdown("LaTeX-code van de afgeleide: <details><pre>" + deriv + "</pre></details>"))
    display(Markdown(f"LaTeX-code van de functie op {e}: <details><pre>" + val + "</pre></details>"))
