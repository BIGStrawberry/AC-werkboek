#!/usr/bin/env python

"""Opgaven voor de toets Analytical Computing."""

__author__      = "Brian van der Bijl"
__copyright__   = "Copyright 2020, Hogeschool Utrecht"

from ac_random import *

def opdracht_0():
    RNG().set(0)

    random_tensor(r"\vec u")
    random_tensor(r"\vec v")
    random_tensor(r"\vec w", 3)
    random_tensor(r"\vec x", 3)
    random_tensor(r"\vec y", 5)
    random_tensor(r"\vec z", 5)

def opdracht_3():
    RNG().set(1)

    random_tensor(r"\vec u")
    random_tensor(r"\vec v")

    RNG().set(2)

    random_tensor(r"\vec w", 2)
    random_tensor(r"\vec x", 2)
    random_tensor(r"\vec y", 4)
    random_tensor(r"\vec z", 4)

def opdracht_5():
    RNG().set(3)

    random_tensor(r"\vec u", 2)
    random_tensor(r"\vec v", 3)
    random_tensor(r"\vec w", 2)
    random_tensor(r"\vec x", 4)
    random_tensor(r"\vec y", 4)

def opdracht_7():
    RNG().set(4)

    random_tensor(r"\vec u", 3)
    random_tensor(r"\vec v", 2)
    random_tensor(r"\mathbf{M}", (3,2))
    random_tensor(r"\mathbf{N}", (2,3))
    random_tensor(r"\mathbf{O}", (2,2))

    RNG().set(2)

    random_tensor(r"\vec {p_a}", 2)
    random_tensor(r"\vec {p_b}", 2)
    random_tensor(r"\vec {q_a}", 4)
    random_tensor(r"\vec {q_b}", 4)

def opdracht_9():
    RNG().set(4).consume_entropy(0x02, -0x14, 0x14)

    random_tensor(r"\vec u",3)
    random_tensor(r"\mathbf{M}", (3,2))
    random_tensor(r"\mathbf{N}", (2,3))
    random_tensor(r"\mathbf{O}", (2,2))

def opdracht_12():
    RNG().set(5)
    random_sys_of_eq()

def opdracht_13():
    RNG().set(6)

    random_tensor(r"\textbf{M}", (2,2), singular=matrix_nd)
    random_tensor(r"\textbf{N}", (2,2), singular=matrix_gd)
    random_tensor(r"\textbf{O}", (2,2), singular=matrix_nd)
    random_tensor(r"\textbf{P}", (3,3), singular=matrix_nd, interval=(0,5))
    random_tensor(r"\textbf{Q}", (3,3), singular=matrix_gd, interval=(0,5))
    
def opdracht_15():
    RNG().set(6)

    random_tensor(r"\textbf{M}", (2,2), singular=matrix_nd)
    random_tensor(r"\textbf{N}", (2,2), singular=matrix_gd)
    random_tensor(r"\textbf{O}", (2,2), singular=matrix_nd)
    
def opdracht_18():
    RNG().set(7)

    random_tensor(r"\textbf{M}", (2,2), singular=matrix_nd, interval=(0,5))
    random_tensor(r"\textbf{N}", (2,2), singular=matrix_gd, interval=(0,5))
    random_tensor(r"\textbf{O}", (3,3), singular=matrix_nd, interval=(0,5))
    random_tensor(r"\textbf{P}", (3,3), singular=matrix_gd, interval=(0,5))
    random_tensor(r"\textbf{Q}", (2,3), singular=matrix_ns, interval=(0,5))
    random_tensor(r"\textbf{R}", (2,3), singular=matrix_ns, interval=(0,5))
    
def opdracht_30():
    RNG().set(8)
    random_derivatives()

def opdracht_34():
    RNG().set(9)
    random_integrals()

def opdracht_35():
    RNG().set(10)
    random_de()