#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 11:02:08 2021

@author: veit
"""

import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

ξ = sym.Symbol("ξ")
ell = sym.Symbol("ell")
ρ = sym.Symbol("ρ")
A = sym.Symbol("A")

N = sym.Matrix([[1-ξ,               0,              0, ξ,             0,              0],
                [  0, 1-3*ξ**2+2*ξ**3, ξ*ell*(1-ξ)**2, 0,  ξ**2*(3-2*ξ), ξ**2*ell*(ξ-1)],
                [  0,   6*ξ/ell*(ξ-1),   1-4*ξ+3*ξ**2, 0, 6*ξ/ell*(1-ξ),      ξ*(3*ξ-2)]])

D = sym.zeros(3, 6)
D[0, :] = sym.diff(N[0, :]/ell, ξ)
D[1, :] = -sym.diff(N[1, :]/ell**2, ξ, 2)
D[2, :] = -sym.diff(N[2, :]/ell, ξ)
