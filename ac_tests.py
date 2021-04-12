#!/usr/bin/env python

"""Testsuite voor de toets Analytical Computing."""

__author__      = "Brian van der Bijl"
__copyright__   = "Copyright 2020, Hogeschool Utrecht"

import math
import unittest as tst
from fractions import Fraction
import numpy as np

from IPython.display import display, Markdown

from ac_exceptions import NonInvertibleError, DimensionError
from ac import polynomial
import ac_formula

# pylint: disable=W0613,R0201,C0103,R0913
# W0613 Unused argument: false positives due to defining classes in side functions
# R0201 Method could be a function: because of how the unittest library works
# C0103 Argument name not conforming to snake case: to match variable names used in maths
# R0913 To many arguments: all defined functions and objects need to be passed to tests

suite = tst.TestSuite()


def run_tests(test):
    try:
        _ = get_ipython()
        ipython = True
    except NameError:
        ipython = False

    test_loader = tst.TestLoader().loadTestsFromTestCase(test)

    if ipython:
        testsuite = tst.TestSuite()
        testsuite.addTest(test_loader)
        tst.TextTestRunner(verbosity=2).run(testsuite)
    else:
        suite.addTest(test_loader)


def test_vector_addition(vector_addition):
    class TestVectorAddition(tst.TestCase): # pylint: disable=C0103
        u = np.array((1, 2, 3))
        v = np.array((4, 5, 6))
        w = np.array((7,8))

        def test_valid_addition(self):
            np.testing.assert_array_equal(vector_addition(self.u, self.v), np.array((5,7,9)))
        def test_additive_unit(self):
            np.testing.assert_array_equal(vector_addition(self.v, np.zeros(3)), self.v)
        def test_additive_unit_2(self):
            np.testing.assert_array_equal(vector_addition(np.zeros(3), self.v), self.v)
        def test_numpy_array(self):
            np.testing.assert_equal(str(type(vector_addition(self.u, self.v))), "<class 'numpy.ndarray'>", "Return type must be a numpy array!")
        def test_invalid_addition(self):
            with self.assertRaises(DimensionError):
                vector_addition(self.v, self.w)
    display(Markdown(r"""### Tests
| **Test** | **Betekenis** |
|---:|:---|
| `test_valid_addition` | $\vec u + \vec v$ klopt  |
| `test_additive_unit` | $\vec u + \vec 0 = \vec u$ ($\vec 0$ is de vector van de juiste lengte met alleen maar 0 waardes) |
| `test_additive_unit2` | $\vec 0 + \vec u = \vec u$ |
| `test_test_invalid_addition` | $\vec v + \vec w$ geeft een `DimensionError` als de dimensies niet kloppen |
"""))
    run_tests(TestVectorAddition)

def test_negative_of_vector(negative_of_vector, vector_addition):
    class TestNegativeOfVector(tst.TestCase): # pylint: disable=C0103
        v = np.array((1, -2, 3))
        z = np.zeros(3)

        def test_negative_of_vector(self):
            np.testing.assert_array_equal(negative_of_vector(self.v), -self.v)
        def test_negative_of_zero(self):
            np.testing.assert_array_equal(negative_of_vector(self.z), self.z)
        def test_sum_vector_negative_is_zero(self):
            np.testing.assert_array_equal(vector_addition(self.v, negative_of_vector(self.v)), self.z)
        def test_numpy_array(self):
            np.testing.assert_equal(str(type(negative_of_vector(self.v))), "<class 'numpy.ndarray'>", "Return type must be a numpy array!")
    display(Markdown(r"""### Tests
| **Test** | **Betekenis** |
|---:|:---|
| `test_negative_of_vector` | $-\vec v$ klopt  |
| `test_negative_of_zero` | $- \vec 0 = \vec 0$ (de vector van de juiste lengte met alleen maar 0 waardes) |
| `test_sum_vector_negative_is_zero` | $\vec v + (-\vec v) = \vec 0$ |
"""))
    run_tests(TestNegativeOfVector)

def test_scalar_product(scalar_product, vector_addition):
    class TestScalarProduct(tst.TestCase): # pylint: disable=C0103
        a = 4
        b = 9
        u = np.array((1, 2, 3))
        v = np.array((5, 6, 7))
        w = np.array((24, 32, 40))
        x = np.array((65, 78, 91))
        z = np.zeros(3)

        def test_zero(self):
            np.testing.assert_array_equal(scalar_product(0, self.v), self.z)
        def test_unit(self):
            np.testing.assert_array_equal(scalar_product(1, self.v), self.v)
        def test_double(self):
            np.testing.assert_array_equal(scalar_product(2, self.v), vector_addition(self.v, self.v))
        def test_distributive_vector_a(self):
            np.testing.assert_array_equal(scalar_product(self.a, vector_addition(self.u, self.v)),
                                          self.w)
        def test_distributive_vector_b(self):
            np.testing.assert_array_equal(self.w,
                                          vector_addition(scalar_product(self.a, self.u),
                                                          scalar_product(self.a, self.v)))
        def test_distributive_field_a(self):
            np.testing.assert_array_equal(scalar_product(self.a + self.b, self.v),
                                          self.x)
        def test_distributive_field_b(self):
            np.testing.assert_array_equal(self.x,
                                          vector_addition(scalar_product(self.a, self.v),
                                                          scalar_product(self.b, self.v)))
        def test_numpy_array(self):
            np.testing.assert_equal(str(type(scalar_product(1, self.v))), "<class 'numpy.ndarray'>", "Return type must be a numpy array!")
    display(Markdown(r"""### Tests
| **Test** | **Betekenis** |
|---:|:---|
| `test_zero` | $0\vec v = \vec 0$ (de vector van de juiste lengte met alleen maar 0 waardes) |
| `test_unit` | $1\vec v = \vec v$ |
| `test_double` | $2\vec v = \vec v + \vec v$ |
| `test_distributive_vector` | $a(\vec u + \vec v) = a \vec u + a \vec v$ |
| `test_distributive_field` | $(a+b)\vec v = a \vec v + b \vec v$|
"""))
    run_tests(TestScalarProduct)

def test_inner_product(inner_product, scalar_product):
    class TestInnerProduct(tst.TestCase): # pylint: disable=C0103
        a = 4
        u = np.array((1, -2, 3))
        v = np.array((-5, 6, -7))
        w = np.array((4,8))
        z = np.zeros(3)

        def test_valid_product(self):
            np.testing.assert_equal(inner_product(self.u, self.v), -38)
        def test_invalid_product(self):
            with self.assertRaises(DimensionError):
                inner_product(self.v, self.w)
        def test_zero(self):
            np.testing.assert_equal(inner_product(self.v, self.z), 0)
        def test_positive_definite(self):
            np.testing.assert_array_less(0, inner_product(self.v, self.v))
        def test_commutative(self):
            np.testing.assert_equal(inner_product(self.u, self.v), inner_product(self.v, self.u))
        def test_linear1(self):
            np.testing.assert_equal(inner_product(scalar_product(self.a, self.v), self.u),
                                    self.a * inner_product(self.v, self.u))
        def test_linear2(self):
            np.testing.assert_equal(inner_product(self.v, scalar_product(self.a, self.u)),
                                    self.a * inner_product(self.v, self.u))
    display(Markdown(r"""### Tests
| **Test** | **Betekenis** |
|---:|:---|
| `test_valid_product` | $\langle \vec u | \vec v \rangle $ werkt |
| `test_invalid_product` | $\langle \vec v | \vec w \rangle$ geeft een `DimensionError` als de dimensies niet kloppen |
| `test_zero` | $\langle \vec v | \vec 0 \rangle = 0$ |
| `test_positive_definite` | $\langle \vec v | \vec v \rangle > 0$ zolang $\vec v$ niet $\vec 0$ is |
| `test_commutative` | $\langle \vec u | \vec v \rangle = \langle \vec v | \vec u \rangle$ |
| `test_linear1` | $\langle a\vec v | \vec u \rangle = a \langle \vec v | \vec u \rangle$ |
| `test_linear2` | $\langle \vec v | a\vec u \rangle = a \langle \vec v | \vec u \rangle$|
"""))
    run_tests(TestInnerProduct)

def test_vector_matrix_product(matrix_product):
    class TestVectorMatrixProduct(tst.TestCase): # pylint: disable=C0103
        a = 4
        v = np.array((1, 2, 3))
        M = np.array(((6, 5, 4), (9, 8, 7)))
        z = np.zeros(3)

        def test_valid_product(self):
            np.testing.assert_array_equal(matrix_product(self.M, self.v), np.array((28, 46)))
        def test_invalid_product(self):
            with self.assertRaises(DimensionError):
                matrix_product(self.M.T, self.v)
        def test_scalar_product(self):
            np.testing.assert_array_equal(matrix_product(2*self.M, self.v), 2*matrix_product(self.M, self.v))
        def test_scalar_product2(self):
            np.testing.assert_array_equal(matrix_product(self.M, 2*self.v), 2*matrix_product(self.M, self.v))
    display(Markdown(r"""### Tests
| **Test** | **Betekenis** |
|---:|:---|
| `test_valid_product` | $\mathbf{M} \vec v$ werkt |
| `test_invalid_product` | $\mathbf{M} \vec w$ geeft een `DimensionError` als de dimensies niet kloppen |
| `test_scalar_product` | $(2\mathbf{M}) \vec v = 2(\mathbf{M} \vec v)$ |
| `test_scalar_product2` | $\mathbf{M} (2\vec v) = 2(\mathbf{M} \vec v)$ |
"""))
    run_tests(TestVectorMatrixProduct)

def test_matrix_product(matrix_product):
    class TestMatrixProduct(tst.TestCase): # pylint: disable=C0103
        a = 4
        v = np.array((1, 2, 3))
        M = np.array(((6, 5, 4), (9, 8, 7)))
        N = np.array(((4, 0), (4, 5), (8, 3)))
        z = np.zeros(3)

        def test_vector_product(self):
            np.testing.assert_array_equal(matrix_product(self.M, self.v).reshape(2), np.array((28, 46)))
        def test_matrix_product(self):
            np.testing.assert_array_equal(matrix_product(self.M, self.N), np.array(((76, 37), (124, 61))))
        def test_invalid_product(self):
            with self.assertRaises(DimensionError):
                matrix_product(self.M, self.M)
    display(Markdown(r"""### Tests
| **Test** | **Betekenis** |
|---:|:---|
| `test_vector_product` | $\mathbf{M} \vec v$ werkt nog steeds (uitvoer mag een 1-D array zijn of een 2D-array met 1 kolom) |
| `test_matrix_product` | $\mathbf{M} \mathbf{N}$ werkt |
| `test_invalid_product` | $\mathbf{M} \mathbf{N}$ geeft een `DimensionError` als de dimensies niet kloppen |
"""))
    run_tests(TestMatrixProduct)

def test_neural_network(read_network, run_network, matrix_product):
    class TestNeuralNetwork(tst.TestCase):

        layer1 = np.array(((0.5,  0.2,  0  ,  0  , -0.2),
                           (0.2, -0.5, -0.1,  0.9, -0.8),
                           (0  ,  0.2,  0  ,  0.1, -0.1),
                           (0.1,  0.8,  0.3,  0  , -0.7)))
        layer2 = np.array(((0.5,  0.2, -0.1,  0.9),
                           (0.2, -0.5,  0.3,  0.1)))

        def test_read_singlelayer(self):
            np.testing.assert_array_almost_equal(read_network(r"example.json"), self.layer1)
        def test_read_duallayer(self):
            np.testing.assert_array_almost_equal(read_network(r"example-2layer.json"),
                                                 matrix_product(self.layer2, self.layer1))
        def test_run_singlelayer(self):
            np.testing.assert_array_almost_equal(run_network(r"example.json", np.ones(5)).reshape(4),
                                                 np.array((0.5, -0.3, 0.2, 0.5)))
        def test_run_duallayer(self):
            np.testing.assert_array_almost_equal(run_network(r"example-2layer.json", np.ones(5)).reshape(2),
                                                 np.array((0.62, 0.36)))
    display(Markdown(r"""### Tests
| **Test** | **Betekenis** |
|---:|:---|
| `test_read_singlelayer` | Het voorbeeld-netwerk met 1 layer wordt goed ingelezen |
| `test_read_duallayer` | Het voorbeeld-netwerk met 2 layers wordt goed ingelezen |
| `test_run_singlelayer` | Het voorbeeld-netwerk met 1 layer wordt goed uitgevoerd |
| `test_run_duallayer` | Het voorbeeld-netwerk met 2 layers wordt goed uitgevoerd |
"""))
    run_tests(TestNeuralNetwork)

def test_determinant(determinant_2, determinant_3, determinant):
    class TestDeterminant(tst.TestCase): # pylint: disable=C0103
        M = np.array(((1,2),(3,4)))
        N = np.array(((1,-2,0),(0,1,-1),(1,-5,1)))
        O = np.array(((1,2),(3,4),(5,6)))
        P = np.array(((1,-1),(2,-2)))
        Q = np.array(((1,2,3),(4,5,6),(7,8,9)))
        R = np.array(((1,2,3,4),(5,6,7,8),(9,10,11,12)))

        def test_determinant2_nonzero(self):
            np.testing.assert_equal(determinant_2(self.M), -2)
        def test_determinant2_zero(self):
            np.testing.assert_equal(determinant_2(self.P), 0)
        def test_determinant2_invalid(self):
            with self.assertRaises(DimensionError):
                determinant_2(self.N)

        def test_determinant3_nonzero(self):
            np.testing.assert_equal(determinant_3(self.N), -2)
        def test_determinant3_zero(self):
            np.testing.assert_equal(determinant_3(self.Q), 0)
        def test_determinant3_invalid(self):
            with self.assertRaises(DimensionError):
                determinant_3(self.M)

        def test_determinant_nonzero2(self):
            np.testing.assert_equal(determinant(self.M), -2)
        def test_determinant_nonzero3(self):
            np.testing.assert_equal(determinant(self.N), -2)
        def test_determinant_zero2(self):
            np.testing.assert_equal(determinant(self.P), 0)
        def test_determinant_zero3(self):
            np.testing.assert_equal(determinant(self.Q), 0)
        def test_determinant_scalar(self):
            np.testing.assert_equal(determinant(np.array(((42,),))), 42)
        def test_determinant_too_large(self):
            with self.assertRaises(DimensionError):
                determinant(self.R)
        def test_determinant_invalid(self):
            with self.assertRaises(DimensionError):
                determinant(self.O)
    display(Markdown(r"""### Tests
| **Test** | **Betekenis** |
|---:|:---|
| `test_determinant2_nonzero` | `determinant_2` werkt op omkeerbare $2 \times 2$ matrix (determinant is niet 0) |
| `test_determinant2_zero` | `determinant_2` werkt op onomkeerbare $2 \times 2$ matrix (determinant is 0)  |
| `test_determinant2_invalid` | `determinant_2` handelt foute invoer goed af |
| `test_determinant3_nonzero` | `determinant_3` werkt op omkeerbare $3 \times 3$ matrix (determinant is niet 0)  |
| `test_determinant3_zero` | `determinant_3` werkt op onomkeerbare $3 \times 3$ matrix (determinant is 0) |
| `test_determinant3_invalid` | `determinant_3` handelt foute invoer goed af |
| `test_determinant_scalar` | `determinant` werkt correct op $1 \times 1$ matrix |
| `test_determinant_nonzero2` | `determinant` werkt op omkeerbare $2 \times 2$ matrix (determinant is niet 0)  |
| `test_determinant_nonzero3` | `determinant` werkt op omkeerbare $3 \times 3$ matrix (determinant is niet 0)  |
| `test_determinant_zero2` | `determinant` werkt op onomkeerbare $2 \times 2$ matrix (determinant is 0) |
| `test_determinant_zero3` | `determinant` werkt op onomkeerbare $3 \times 3$ matrix (determinant is 0) |
| `test_determinant_too_large` | `determinant` handelt te grote invoer goed af |
| `test_determinant_invalid` | `determinant` handelt niet-vierkante invoer goed af |
"""))
    run_tests(TestDeterminant)

def test_inverse_2(inverse_matrix_2):
    class TestInverse2(tst.TestCase): # pylint: disable=C0103
        M = np.array(((1,2),(3,4)))
        N = np.array(((1,-1),(2,-2)))
        O = np.array(((1,2),(3,4),(5,6)))

        def test_inverse2(self):
            np.testing.assert_equal(inverse_matrix_2(self.M), np.array(((-2,1),(1.5,-0.5))))
        def test_inverse2_no_inverse(self):
            with self.assertRaises(NonInvertibleError):
                inverse_matrix_2(self.N)
        def test_inverse2_invalid(self):
            with self.assertRaises(DimensionError):
                inverse_matrix_2(self.O)
    display(Markdown(r"""### Tests
| **Test** | **Betekenis** |
|---:|:---|
| `test_inverse2` | Functie werkt voor een inverteerbare $2\times 2$ matrix |
| `test_inverse2_invalid` | Correcte error voor een matrix met verkeerde dimensies |
| `test_inverse2_no_inverse` | Correcte error voor een niet-omkeerbare $2\times 2$ matrix |
"""))
    run_tests(TestInverse2)

def test_inverse_(inverse_matrix, transpose):
    class TestInverse(tst.TestCase): # pylint: disable=C0103
        M = np.array(((4,6,3,4),(4,7,2,6),(3,3,3,1),(3,7,1,6)))
        Mi = np.array(((-11,3,8,3),(-2,-1,2,2),(11,-2,-8,-4),(6,0,-5,-3)))
        N = np.array(((1,2,3),(4,5,6),(7,8,9)))
        O = np.array(((1,2),(3,4),(5,6)))
        Ot = np.array(((1,3,5),(2,4,6)))

        def test_transpose(self):
            np.testing.assert_equal(transpose(self.O), self.Ot)
        def test_inverse(self):
            np.testing.assert_equal(inverse_matrix(self.M), self.Mi)
        def test_inverse_no_inverse(self):
            with self.assertRaises(NonInvertibleError):
                inverse_matrix(self.N)
        def test_inverse_invalid(self):
            with self.assertRaises(DimensionError):
                inverse_matrix(self.O)
    display(Markdown(r"""### Tests
| **Test** | **Betekenis** |
|---:|:---|
| `test_transpose` | Tranpose functie werkt op zichzelf voor een $3\times 2$ test-matrix |
| `test_inverse` | Functie werkt voor een inverteerbare $4\times 4$ test-matrix |
| `test_inverse_invalid` | Correcte error voor een matrix met verkeerde dimensies |
| `test_inverse_no_inverse` | Correcte error voor een niet-omkeerbare vierkante matrix |"""))
    run_tests(TestInverse)

def test_magisch_vierkant(magisch_vierkant):
    class TestMagischVierkant(tst.TestCase): # pylint: disable=C0103
        i = np.array(((0, 3, 4), (0, 0, 0), (0, 7, 0)))
        i2 = np.array(((8, 3, 4), (1, 5, 9), (6, 7, 2)))
        r = np.array(((5, 0, 0), (0, 0, 4), (0, 0, 6)))
        r2 = np.array(((5, 5, 6.5), (7, 5.5, 4), (4.5, 6, 6)))

        def test_integer(self):
            np.testing.assert_almost_equal(magisch_vierkant(self.i), self.i2)
        def test_rational(self):
            np.testing.assert_almost_equal(magisch_vierkant(self.r), self.r2)
    display(Markdown(r"""### Tests
`test_integer`:
$$\begin{array}{|c|c|c|}\hline   0 & 3 & 4\\\hline   0 & 0 & 0\\\hline   0 & 7 & 0\\\hline \end{array} \mapsto \begin{array}{|c|c|c|}\hline   8 & 3 & 4\\\hline   1 & 5 & 9\\\hline   6 & 7 & 2\\\hline \end{array}$$

`test_rational`:
$$\begin{array}{|c|c|c|}\hline   5 & 0 & 0\\\hline   0 & 0 & 4\\\hline   0 & 0 & 6\\\hline \end{array} \mapsto
\begin{array}{|c|c|c|}\hline   5 & 5 & 6.5\\\hline   7 & 5.5 & 4\\\hline   4.5 & 6 & 6\\\hline \end{array}$$"""))
    run_tests(TestMagischVierkant)

def test_limit(limit_left, limit_right, limit):
    class TestLimit(tst.TestCase):
        def discontinuous_function(self, x: float) -> float:
            if x == 72:
                return -10
            elif x % 13 == 0:
                return None
            else:
                return 2.5 * x

        def holes_function(self, x: float) -> float:
            if x % 13 == 0:
                return None
            else:
                return 2.5 * x

        def single_discontinuity_function(self, x: float) -> float:
            if x == 72:
                return -10
            else:
                return 2.5 * x

        def right_undefined_function(self, x: float) -> float:
            if x >= 10:
                return None
            else:
                return x+3

        def left_undefined_function(self, x: float) -> float:
            if x <= 10:
                return None
            else:
                return x+3

        def piecewise_function(self, x: float) -> float:
            if x < -2:
                return -1.5*x -2
            elif -2 <= x <= 1:
                return -1/3 * (x-1) + 2
            else:
                return x-2

        def test_single_discontinuity(self):
            np.testing.assert_equal(limit(self.single_discontinuity_function,72), 180)
        def test_holes(self):
            np.testing.assert_equal(limit(self.holes_function,78), 195)
        def test_left_undefined_below(self):
            np.testing.assert_almost_equal(limit_left(self.right_undefined_function,10), 13, 3)
        def test_left_undefined_above(self):
            np.testing.assert_almost_equal(limit_right(self.left_undefined_function,10), 13, 3)
        def test_jump_left(self):
            np.testing.assert_almost_equal(limit_left(self.piecewise_function,1), 2, 3)
        def test_jump_right(self):
            np.testing.assert_almost_equal(limit_right(self.piecewise_function,1), -1, 3)
        def test_jump(self):
            np.testing.assert_equal(limit(self.piecewise_function,1), None)

    display(Markdown(r"""### Tests
| **Test** | **Betekenis** |
|---:|:---|
| `test_holes` | Zoekt de limiet van een functie die op een enkel punt undefined is |
| `test_single_discontinuity` | Zoekt de limiet van een functie met een enkel discontinu punt |
| `test_left_undefined_above` | Zoekt de limiet van een functie op het punt waar deze ophoudt te bestaan (rechts) |
| `test_left_undefined_below` | Zoekt de limiet van een functie op het punt waar deze begint te bestaan (links) |
| `test_jump` | Zoekt een niet-bestaande limiet in een piecewise functie |
| `test_jump_left` | Zoekt de limiet van links in de piecewise functie |
| `test_jump_right` | Zoekt de limiet van rechts in de piecewise functie |"""))
    run_tests(TestLimit)

def test_numeric_derivative(get_derivative_at):
    class TestNumericDerivative(tst.TestCase):
        def square(self, x: float) -> float:
            return x**2

        def double(self, x: float) -> float:
            return x*2

        def succ(self, x: float) -> float:
            return x+1

        def test_square(self):
            np.testing.assert_almost_equal(get_derivative_at(self.square, 2), 4, 3)
        def test_double(self):
            np.testing.assert_almost_equal(get_derivative_at(self.double, 2), 2, 3)
        def test_succ(self):
            np.testing.assert_almost_equal(get_derivative_at(self.succ, 2), 1, 3)
    display(Markdown(r"""### Tests
| **Test** | **Betekenis** |
|---:|:---|
| `test_square` | Zoekt de afgeleide van `square` op $x=2$ |
| `test_double` | Zoekt de afgeleide van `double` op $x=2$ |
| `test_succ` | Zoekt de afgeleide van `succ` op $x=2$ |"""))
    run_tests(TestNumericDerivative)

def test_verkeer_snelheden(get_data, bereken_deltas):
    display(Markdown(r"""### Tests
| **Test** | **Betekenis** |
|---:|:---|
| `test_min_1` | De minimale waarde van de berekende snelheden klopt voor voertuig 1 |
| `test_max_1` | De maximale waarde van de berekende snelheden klopt voor voertuig 1 |
| `test_min_1` | De minimale waarde van de berekende snelheden klopt voor voertuig 2 |
| `test_max_1` | De maximale waarde van de berekende snelheden klopt voor voertuig 2 |"""))
    class TestVerkeerSnelheden(tst.TestCase):
        time, car1, car2 = get_data()

        def test_min_1(self):
            car1_speed = bereken_deltas(self.time, self.car1)
            np.testing.assert_almost_equal(min(car1_speed), -2.5)
        def test_max_1(self):
            car1_speed = bereken_deltas(self.time, self.car1)
            np.testing.assert_almost_equal(max(car1_speed), 37.4)
        def test_min_2(self):
            car2_speed = bereken_deltas(self.time, self.car2)
            np.testing.assert_almost_equal(min(car2_speed), 0)
        def test_max_2(self):
            car2_speed = bereken_deltas(self.time, self.car2)
            np.testing.assert_almost_equal(max(car2_speed), 30.7)
    run_tests(TestVerkeerSnelheden)

def test_polynomial_derivative(get_derivative):
    class TestNumericDerivative(tst.TestCase):
        x_squared = polynomial({1: 0, 2: 1})
        x_recip = polynomial({-1: 1})
        x_root = polynomial({1/2: 1})

        def test_squared(self):
            np.testing.assert_equal(get_derivative(self.x_squared)[0][1], 2)
        def test_primes(self):
            np.testing.assert_equal(get_derivative(self.x_squared)[3], 1)
        def test_recip(self):
            np.testing.assert_equal(get_derivative(self.x_recip)[0][-2], -1)
        def test_root(self):
            np.testing.assert_equal(get_derivative(self.x_root)[0][-0.5], 0.5)
    display(Markdown(r"""### Tests
| **Test** | **Betekenis** |
|---:|:---|
| `test_squared` | Zoekt de afgeleide van `x_squared` ($x^2$) |
| `test_recip` | Zoekt de afgeleide van `test_recip` ($\frac{1}{x}$) |
| `test_root` | Zoekt de afgeleide van `x_root` ($\sqrt x$) |
| `test_primes` | Checkt of de variabele `primes` opgehoogd is |"""))
    run_tests(TestNumericDerivative)

def test_matrix_derivative(deriv_matrix):
    f_x = np.array((2,1,3))
    class TestMatrixDerivative(tst.TestCase):

        def test_derivative(self):
            np.testing.assert_array_equal(deriv_matrix(f_x).flatten(), np.array((1,6,0)))
    run_tests(TestMatrixDerivative)

def deriv_message(src, answer):
    if src.deriv() and src.deriv().body:
        return f"Differentiating {str(src)}, I was expecting {str(answer.body)}, but got {str(src.deriv())} 😕"
    else:
        return "I got nothing..."

def test_symbolic_differentiation_alfa(Constant, Variable, Sum, Product, Power):
    class TestSymbolicDifferentiationAlfa(tst.TestCase):

        def test_variable(self):
            form = ac_formula.Function('f', Variable('x'))
            deriv = ac_formula.Function('f', Constant(1), 1)
            np.testing.assert_equal(form.deriv(), deriv, deriv_message(form, deriv))
        def test_product(self):
            form = ac_formula.Function('f', Product(Variable('x'), Variable('x')))
            deriv = ac_formula.Function('f', Sum(Variable('x'),Variable('x')) ,1)
            np.testing.assert_equal(form.deriv(), deriv, deriv_message(form, deriv))
        def test_2x_plus_3(self):
            form = ac_formula.Function('f', Sum(Product(Constant(2), Power(Variable('x'),1)), Constant(3)))
            deriv = ac_formula.Function('f', Constant(2), 1)
            np.testing.assert_equal(form.deriv(), deriv, deriv_message(form, deriv))
        def test_recip_x(self):
            form = ac_formula.Function('f', Power(Variable('x'), -1))
            deriv = ac_formula.Function('f',ac_formula.Negative(Power(Variable('x'),-2)),1)
            np.testing.assert_equal(form.deriv(), deriv, deriv_message(form, deriv))
    display(Markdown(r"""### Tests
| **Test** | **Betekenis** |
|---:|:---|
| `test_variable` | Zoekt de afgeleide van $x$, test `Variable` |
| `test_product` | Zoekt de afgeleide van $x_1 \cdot x_2$, test `Product` |
| `test_recip_x` | Zoekt de afgeleide van $x^{-1}$, test `Power` |
| `test_2x_plus_3` | Zoekt de afgeleide van $2x+3$, combineert alles |"""))
    run_tests(TestSymbolicDifferentiationAlfa)

def test_symbolic_differentiation_bravo(Constant, Variable, Sum, Product, Power, Sin, Cos, Tan):
    class TestSymbolicDifferentiationBravo(tst.TestCase):

        def test_sin(self):
            form = ac_formula.Function('f', Sin(Variable('x')))
            deriv = ac_formula.Function('f', Cos(Variable('x')), 1)
            np.testing.assert_equal(form.deriv(), deriv, deriv_message(form, deriv))
        def test_cos(self):
            form = ac_formula.Function('f', Cos(Variable('x')))
            deriv = ac_formula.Function('f',ac_formula.Negative(Sin(Variable('x'))),1)
            np.testing.assert_equal(form.deriv(), deriv, deriv_message(form, deriv))
        def test_tan(self):
            form = ac_formula.Function('f', Tan(Variable('x')))
            deriv = ac_formula.Function('f',Power(ac_formula.Sec(Variable('x')),2),1)
            np.testing.assert_equal(form.deriv(), deriv, deriv_message(form, deriv))
    display(Markdown(r"""### Tests
| **Test** | **Betekenis** |
|---:|:---|
| `test_sin` | Zoekt de afgeleide van $\text{sin}(x)$ |
| `test_cos` | Zoekt de afgeleide van $\text{cos}(x)$ |
| `test_tan` | Zoekt de afgeleide van $\text{tan}(x)$ |"""))
    run_tests(TestSymbolicDifferentiationBravo)

def test_symbolic_differentiation_charlie(Constant, Variable, Sum, Product, Power, Sin, Cos, Tan, E, Exponent, Ln, Log):
    class TestSymbolicDifferentiationCharlie(tst.TestCase):

        def test_e(self):
            form = ac_formula.Function('f', E(Variable('x')))
            deriv = ac_formula.Function(label='f',body=E(exponent=Variable(label='x')),deriv_order=1)
            np.testing.assert_equal(form.deriv(), deriv, deriv_message(form, deriv))
        def test_exponent(self):
            form = ac_formula.Function('f', Exponent(Constant(2), Variable('x')))
            deriv = ac_formula.Function(label='f',body=Product(left=Exponent(base=Constant(value=2),exponent=Variable(label='x')),right=Ln(argument=Constant(value=2))),deriv_order=1)
            np.testing.assert_equal(form.deriv(), deriv, deriv_message(form, deriv))
        def test_ln(self):
            form = ac_formula.Function('f', Ln(Variable('x')))
            deriv = ac_formula.Function(label='f',body=Power(base=Variable(label='x'),exponent=-1),deriv_order=1)
            np.testing.assert_equal(form.deriv(), deriv, deriv_message(form, deriv))
        def test_log(self):
            form = ac_formula.Function('f', Log(Constant(2), Variable('x')))
            deriv = ac_formula.Function(label='f',body=Power(base=Product(left=Variable(label='x'),right=Ln(argument=Constant(value=2))),exponent=-1),deriv_order=1)
            np.testing.assert_equal(form.deriv(), deriv, deriv_message(form, deriv))
    display(Markdown(r"""### Tests

| **Test** | **Betekenis** |
|---:|:---|
| `test_exponent` | Zoekt de afgeleide van $a^x$, met $a$ als constante |
| `test_log` | Zoekt de afgeleide van $\text{log}_a(x)$, met $a$ als constante |
| `test_e` | Zoekt de afgeleide van $e^x$ |
| `test_ln` | Zoekt de afgeleide van $\text{ln}(x)$ |"""))
    run_tests(TestSymbolicDifferentiationCharlie)

def test_symbolic_differentiation_charlie_eq(Constant, Variable, Sum, Product, Power, Sin, Cos, Tan, E, Exponent, Ln, Log):
    class TestSymbolicDifferentiationCharlieEq(tst.TestCase):
        def test_e_equivalent(self):
            form = ac_formula.Function('f', E(Variable('x')))
            np.testing.assert_almost_equal(form.deriv().eval({'x':-1}), 0.368, 3)
        def test_exponent_equivalent(self):
            form = ac_formula.Function('f', Exponent(Constant(2), Variable('x')))
            np.testing.assert_almost_equal(form.deriv().eval({'x':7}), 88.723, 3)
        def test_ln_equivalent(self):
            form = ac_formula.Function('f', Ln(Variable('x')))
            np.testing.assert_almost_equal(form.deriv().eval({'x':3}), 0.333, 3)
        def test_log_equivalent(self):
            form = ac_formula.Function('f', Log(Constant(2), Variable('x')))
            np.testing.assert_almost_equal(form.deriv().eval({'x':5}), 0.289, 3)
    run_tests(TestSymbolicDifferentiationCharlieEq)

def test_symbolic_differentiation_delta(Constant, Variable, Sum, Product, Power, Sin, Cos, Tan, E, Exponent, Ln, Log):
    class TestSymbolicDifferentiationDelta(tst.TestCase):

        def test_e_x_squared(self):
            form = ac_formula.Function('f', E(Power(Variable('x'),2)))
            deriv = ac_formula.Function(label='f',body=Product(left=Product(left=Constant(value=2),right=Variable(label='x')),right=E(exponent=Power(base=Variable(label='x'),exponent=2))),deriv_order=1)
            np.testing.assert_equal(form.deriv(), deriv, deriv_message(form, deriv))
        def test_five_to_the_e_x(self):
            form = ac_formula.Function('f', Exponent(Constant(5), E(Variable('x'))))
            deriv = ac_formula.Function(label='f',body=Product(left=E(exponent=Variable(label='x')),right=Product(left=Exponent(base=Constant(value=5),exponent=E(exponent=Variable(label='x'))),right=Ln(argument=Constant(value=5)))),deriv_order=1)
            np.testing.assert_equal(form.deriv(), deriv, deriv_message(form, deriv))
        def test_ln_x_squared(self):
            form = ac_formula.Function('f', Ln(Power(Variable('x'),2)))
            deriv = ac_formula.Function(label='f',body=Product(left=Constant(value=2),right=Power(base=Variable(label='x'),exponent=-1)),deriv_order=1)
            np.testing.assert_equal(form.deriv(), deriv, deriv_message(form, deriv))
        def test_five_log_e_x(self):
            form = ac_formula.Function('f', Log(Constant(5), E(Variable('x'))))
            deriv = ac_formula.Function(label='f',body=Power(base=Ln(argument=Constant(value=5)),exponent=-1),deriv_order=1)
            np.testing.assert_equal(form.deriv(), deriv, deriv_message(form, deriv))
        def test_sin_squared_x(self):
            form = ac_formula.Function('f', Power(Sin(Variable('x')), 2))
            deriv = ac_formula.Function(label='f',body=Product(left=Product(left=Constant(value=2),right=Cos(argument=Variable(label='x'))),right=Sin(argument=Variable(label='x'))),deriv_order=1)
            np.testing.assert_equal(form.deriv(), deriv, deriv_message(form, deriv))
    display(Markdown(r"""### Tests
#### `test_sin_squared_x`
$$f^{}(x) =\text{sin}^{2}x \mapsto f^{\prime}(x) =2 \  \text{cos}(x) \  \text{sin}(x)$$

#### `test_e_x_squared`
$$f^{}(x) =e^{x^{2}} \mapsto f^{\prime}(x) =2x \  e^{x^{2}}$$

#### `test_five_to_the_e_x`
$$f^{}(x) =5^{e^{x}} \mapsto f^{\prime}(x) =e^{x} \  5^{e^{x}} \  \text{ln}(5)$$

#### `test_ln_x_squared`
$$f^{}(x) =\text{ln}(x^{2}) \mapsto f^{\prime}(x) =\frac{2}{x}$$

#### `test_five_log_e_x`
$$f^{}(x) =\text{log}_{5}(e^{x}) \mapsto f^{\prime} =\frac{1}{\text{ln}(5)}$$"""))
    run_tests(TestSymbolicDifferentiationDelta)

def test_symbolic_differentiation_delta_eq(Constant, Variable, Sum, Product, Power, Sin, Cos, Tan, E, Exponent, Ln, Log):
    class TestSymbolicDifferentiationDeltaEq(tst.TestCase):

        def test_e_x_squared_equivalent(self):
            form = ac_formula.Function('f', E(Power(Variable('x'),2)))
            np.testing.assert_almost_equal(form.deriv().eval({'x':5}), 720048993373.859, 3)
        def test_five_to_the_e_x_equivalent(self):
            form = ac_formula.Function('f', Exponent(Constant(5), E(Variable('x'))))
            np.testing.assert_almost_equal(form.deriv().eval({'x':-1}), 1.07, 3)
        def test_ln_x_squared_equivalent(self):
            form = ac_formula.Function('f', Ln(Power(Variable('x'),2)))
            np.testing.assert_equal(form.deriv().eval({'x':8}), 0.25)
        def test_five_log_e_x_equivalent(self):
            form = ac_formula.Function('f', Log(Constant(5), E(Variable('x'))))
            np.testing.assert_almost_equal(form.deriv().eval({'x':3}), 0.621, 3)
        def test_sin_squared_x_equivalent(self):
            form = ac_formula.Function('f', Power(Sin(Variable('x')), 2))
            np.testing.assert_almost_equal(form.deriv().eval({'x': 1.5 * math.pi}), 0, 3)
    run_tests(TestSymbolicDifferentiationDeltaEq)

def test_regressie(train, gradient, cost, data):
    class TestRegressie(tst.TestCase):

        def test_convergence(self):
            s, i = train(data)
            np.testing.assert_array_less(cost(data, s, i), 3200)
    run_tests(TestRegressie)

def test_verkeer_posities(get_data, bereken_posities, vind_botsing):
    display(Markdown(r"""### Tests
| **Test** | **Betekenis** |
|---:|:---|
| `test_time` | Controlleert de tijd van de botsing |
| `test_car_a` | Controlleert de identiteit van de eerste auto |
| `test_car_a_pos` | Controlleert de positie van de eerste auto |
| `test_car_b` | Controlleert de identiteit van de tweede auto |
| `test_car_b_pos` | Controlleert de positie van de tweede auto |"""))

    class TestVerkeerPosities(tst.TestCase):
        time, car1, car2, car3 = get_data()
        car1_position = bereken_posities(time, car1)
        car2_position = bereken_posities(time, car2)
        car3_position = bereken_posities(time, car3)

        (t,ca,cap,cb,cbp) = vind_botsing(time,car1_position,car2_position,car3_position)

        def test_time(self):
            np.testing.assert_equal(self.t, 28.2)
        def test_car_a(self):
            np.testing.assert_equal(self.ca, 2)
        def test_car_b(self):
            np.testing.assert_equal(self.cb, 3)
        def test_car_a_pos(self):
            np.testing.assert_almost_equal(self.cap, 657.4, 1)
        def test_car_b_pos(self):
            np.testing.assert_almost_equal(self.cbp, 658.5, 1)
    run_tests(TestVerkeerPosities)


def test_numeric_integral(get_integral_between):
    class TestNumericIntegral(tst.TestCase):
        def pi(self, x: float) -> float:
            return 4 / (1+x**2)
        def gauss(self, x: float) -> float:
            return math.e **(-x**2)
        def bizarre(self, x: float) -> float:
            return (math.sin(x)**2 / -math.cos(x**4)) + math.e**x

        def test_gauss(self):
            np.testing.assert_almost_equal(get_integral_between(self.gauss, -100, 100, 0.1), math.sqrt(math.pi), 0.01)
        def test_pi(self):
            np.testing.assert_almost_equal(get_integral_between(self.pi, 0, 1), math.pi, 0.01)
        def test_bizarre(self):
            np.testing.assert_almost_equal(get_integral_between(self.bizarre, -0.74, 1.07), 1.86, 0.01)
    display(Markdown(r"""### Tests
| **Test** | **Betekenis** |
|---:|:---|
| `test_pi` | Zoekt de integraal van `pi` op $[0,1]$, dit zou $\pi$ moeten benaderen  |
| `test_gauss` | Zoekt de integraal van `gauss` op $[-100,100]$, dit zou $\sqrt\pi$ moeten benaderen |
| `test_bizarre` | Zoekt de integraal van `bizarre` voor de middelste heuvel, dit zou $1.86$ moeten benaderen |"""))
    run_tests(TestNumericIntegral)


def test_polynomial_integral(get_integral):
    class TestNumericIntegral(tst.TestCase):
        x_squared = polynomial({1: 0, 2: 1})
        x_recip_sq = polynomial({-2: 1})
        x_root = polynomial({Fraction(1,2): 1})

        def test_squared(self):
            np.testing.assert_almost_equal(get_integral(self.x_squared)[0][3], Fraction(1,3), 0.001)
        def test_recip_sq(self):
            np.testing.assert_equal(get_integral(self.x_recip_sq)[0][-1], -1)
        def test_root(self):
            np.testing.assert_equal(get_integral(self.x_root)[0][Fraction(3,2)], Fraction(2,3))
    display(Markdown(r"""### Tests
| **Test** | **Betekenis** |
|---:|:---|
| `test_squared` | Zoekt de integraal van `x_squared` ($x^2$) |
| `test_recip_sq` | Zoekt de integraal van `test_recip_sq` ($\frac{1}{x^2}$) |
| `test_root` | Zoekt de integraal van `x_root` ($\sqrt x$) |"""))
    run_tests(TestNumericIntegral)


def integrate_message(src, answer):
    integral = src.integrate('x')
    if integral and integral.body:
        return f"Integrating {str(src)}, I was expecting {str(answer.body)}, but got {str(integral)} 😕"
    else:
        return "I got nothing..."

def test_symbolic_integration_alfa(Constant, Variable, Sum, Product, Power):
    class TestSymbolicIntegrationAlfa(tst.TestCase):

        def test_variable_x(self):
            form = ac_formula.Function('f', Variable('x'))
            integral = ac_formula.Function('f', Sum(Product(Constant(0.5),Power(Variable('x'),2)),Variable('C')), -1)
            np.testing.assert_equal(form.integrate('x'), integral, integrate_message(form, integral))
        def test_variable_y(self):
            form = ac_formula.Function('f', Variable('y'))
            integral = ac_formula.Function('f', Sum(Product(Variable('x'),Variable('y')),Variable('C')), -1)
            np.testing.assert_equal(form.integrate('x'), integral, integrate_message(form, integral))
        def test_product(self):
            form = ac_formula.Function('f', Product(Variable('x'), Variable('y')))
            integral = ac_formula.Function('f', Sum(Product(Variable('y'),Product(Constant(0.5),Power(Variable('x'),2))),Variable('C')), -1)
            np.testing.assert_equal(form.integrate('x'), integral, integrate_message(form, integral))
        def test_sum(self):
            form = ac_formula.Function('f', Sum(Variable('x'), Variable('y')))
            integral = ac_formula.Function('f', Sum(Sum(Product(Constant(0.5),Power(Variable('x'),2)),Product(Variable('x'),Variable('y'))),Variable('C')), -1)
            np.testing.assert_equal(form.integrate('x'), integral, integrate_message(form, integral))
        def test_power(self):
            form = ac_formula.Function('f', Power(Variable('x'), 3))
            integral = ac_formula.Function('f',Sum(Product(Constant(0.25),Power(Variable('x'),4)),Variable('C')),-1)
            np.testing.assert_equal(form.integrate('x'), integral, integrate_message(form, integral))

    run_tests(TestSymbolicIntegrationAlfa)

def test_symbolic_integration_alfa_eq(Constant, Variable, Sum, Product, Power):
    class TestSymbolicIntegrationAlfaEq(tst.TestCase):

        def test_variable_x_equivalent(self):
            form = ac_formula.Function('f', Variable('x'))
            np.testing.assert_equal(form.integrate('x').eval({'x': 12}), 72)
        def test_variable_y_equivalent(self):
            form = ac_formula.Function('f', Variable('y'))
            np.testing.assert_equal(form.integrate('x').eval({'x': 3, 'y': 4}), 12)
        def test_product_equivalent(self):
            form = ac_formula.Function('f', Product(Variable('x'), Variable('y')))
            np.testing.assert_equal(form.integrate('x').eval({'x': 8, 'y': 2}), 64)
        def test_sum_equivalent(self):
            form = ac_formula.Function('f', Sum(Variable('x'), Variable('y')))
            np.testing.assert_equal(form.integrate('x').eval({'x': 1, 'y': 9}), 9.5)
        def test_power_equivalent(self):
            form = ac_formula.Function('f', Power(Variable('x'), 3))
            np.testing.assert_equal(form.integrate('x').eval({'x': 11}), 3660.25)

    run_tests(TestSymbolicIntegrationAlfaEq)

def test_symbolic_integration_bravo(Constant, Variable, Sum, Product, Power, Sin, Cos, Tan, E, Exponent, Ln, Log):
    class TestSymbolicIntegrationBravo(tst.TestCase):

        def test_sin(self):
            form = ac_formula.Function('f', Sin(Variable('x')))
            integral = ac_formula.Function('f',Sum(ac_formula.Negative(Cos(Variable('x'))),Variable('C')),-1)
            np.testing.assert_equal(form.integrate('x'), integral, integrate_message(form, integral))
        def test_cos(self):
            form = ac_formula.Function('f', Cos(Variable('x')))
            integral = ac_formula.Function('f',Sum(Sin(Variable('x')),Variable('C')),-1)
            np.testing.assert_equal(form.integrate('x'), integral, integrate_message(form, integral))
        def test_tan(self):
            form = ac_formula.Function('f', Tan(Variable('x')))
            integral = ac_formula.Function('f',Sum(ac_formula.Negative(Ln(Cos(Variable('x')))),Variable('C')),-1)
            np.testing.assert_equal(form.integrate('x'), integral, integrate_message(form, integral))
        def test_e(self):
            form = ac_formula.Function('f', E(Variable('x')))
            integral = ac_formula.Function('f',Sum(E(Variable('x')),Variable('C')),-1)
            np.testing.assert_equal(form.integrate('x'), integral, integrate_message(form, integral))
        def test_exponent(self):
            form = ac_formula.Function('f', Exponent(Constant(3), Variable('x')))
            integral = ac_formula.Function('f',Sum(Product(Exponent(Constant(3),Variable('x')),
                                                Power(Ln(Constant(3)),-1)),Variable('C')),-1)
            np.testing.assert_equal(form.integrate('x'), integral, integrate_message(form, integral))
        def test_ln(self):
            form = ac_formula.Function('f', Ln(Variable('x')))
            integral = ac_formula.Function('f',Sum(Product(Variable('x'),Sum(Ln(Variable('x')),
                         ac_formula.Negative(Constant(1)))),Variable('C')),-1)
            np.testing.assert_equal(form.integrate('x'), integral, integrate_message(form, integral))
        def test_log(self):
            form = ac_formula.Function('f', Log(Constant(3),Variable('x')))
            integral = ac_formula.Function('f',Sum(Product(Product(Variable('x'),Sum(Ln(Variable('x')),
                         ac_formula.Negative(Constant(1)))),Power(Ln(Constant(3)),-1)),Variable('C')),-1)
            np.testing.assert_equal(form.integrate('x'), integral, integrate_message(form, integral))

    run_tests(TestSymbolicIntegrationBravo)


def test_symbolic_integration_bravo_eq(Constant, Variable, Sum, Product, Power, Sin, Cos, Tan, E, Exponent, Ln, Log):
    class TestSymbolicIntegrationBravoEq(tst.TestCase):
        def test_sin_equivalent(self):
            form = ac_formula.Function('f', Sin(Variable('x')))
            np.testing.assert_almost_equal(form.integrate('x').eval({'x': -1}), -0.54,3)
        def test_cos_equivalent(self):
            form = ac_formula.Function('f', Cos(Variable('x')))
            np.testing.assert_almost_equal(form.integrate('x').eval({'x': 7}), 0.657,3)
        def test_tan_equivalent(self):
            form = ac_formula.Function('f', Tan(Variable('x')))
            np.testing.assert_almost_equal(form.integrate('x').eval({'x': 13}), 0.097,3)
        def test_e_equivalent(self):
            form = ac_formula.Function('f', E(Variable('x')))
            np.testing.assert_almost_equal(form.integrate('x').eval({'x': 5}), 148.413,3)
        def test_exponent_equivalent(self):
            form = ac_formula.Function('f', Exponent(Constant(3), Variable('x')))
            np.testing.assert_almost_equal(form.integrate('x').eval({'x': 7}), 1990.693,3)
        def test_ln_equivalent(self):
            form = ac_formula.Function('f', Ln(Variable('x')))
            np.testing.assert_almost_equal(form.integrate('x').eval({'x': 42}), 114.982,3)
        def test_log_equivalent(self):
            form = ac_formula.Function('f', Log(Constant(3),Variable('x')))
            np.testing.assert_almost_equal(form.integrate('x').eval({'x': 72}), 214.744,3)

    run_tests(TestSymbolicIntegrationBravoEq)

def test_euler(afgeleide_functie, euler):
    class TestEuler(tst.TestCase):

        def test_result(self):
            np.testing.assert_almost_equal(euler(afgeleide_functie, 0, 1, 0.025, 0.95), 129.876, 3)

    run_tests(TestEuler)
