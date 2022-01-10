#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 11:02:08 2021

@author: veit
"""

import sympy as sym
import numpy as np

ξ = sym.Symbol("ξ")
ell = sym.Symbol("ell")
ρ = sym.Symbol("ρ")
A = sym.Symbol("A")

N = sym.Matrix([[1-ξ,                   0,                    0],
                [  0,     1-3*ξ**2+2*ξ**3,                    0],
                [  0,                   0,      1-3*ξ**2+2*ξ**3],
                [  0,                   0,                    0],
                [  0,                   0, -ell*(ξ-2*ξ**2+ξ**3)],
                [  0, ell*(ξ-2*ξ**2+ξ**3),                    0],
                [  ξ,                   0,                    0],
                [  0,       3*ξ**2-2*ξ**3,                    0],
                [  0,                   0,        3*ξ**2-2*ξ**3],
                [  0,                   0,                    0],
                [  0,                   0,    -ell*(-ξ**2+ξ**3)],
                [  0,    ell*(-ξ**2+ξ**3),                    0]]).T

S = sym.integrate(ρ*A*ell*N, (ξ, 0, 1))
S00 = sym.integrate(ρ*A*ell*N[0, :].T*N[0, :], (ξ, 0, 1))
S11 = sym.integrate(ρ*A*ell*N[1, :].T*N[1, :], (ξ, 0, 1))
S22 = sym.integrate(ρ*A*ell*N[2, :].T*N[2, :], (ξ, 0, 1))
S01 = sym.integrate(ρ*A*ell*N[0, :].T*N[1, :], (ξ, 0, 1))
S02 = sym.integrate(ρ*A*ell*N[0, :].T*N[2, :], (ξ, 0, 1))
S12 = sym.integrate(ρ*A*ell*N[1, :].T*N[2, :], (ξ, 0, 1))

D = sym.zeros(3, 12)
D[0, :] = sym.diff(N/ell, ξ)[0, :]
D[1, :] = -sym.diff(N/ell**2, ξ, 2)[1, :]
D[2, :] = -sym.diff(N/ell**2, ξ, 2)[2, :]
