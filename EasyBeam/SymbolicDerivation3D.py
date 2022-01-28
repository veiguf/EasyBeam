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

N = sym.Matrix([[1 - ξ,                       0,                        0,     0,                  0,                  0],
                [    0,     2*ξ**3 - 3*ξ**2 + 1,                        0,     0,                  0,  6*(-ξ**2 + ξ)/ell],
                [    0,                       0,      2*ξ**3 - 3*ξ**2 + 1,     0, -6*(-ξ**2 + ξ)/ell,                  0],
                [    0,                       0,                        0, 1 - ξ,                  0,                  0],
                [    0,                       0, -ell*(ξ**3 - 2*ξ**2 + ξ),     0,   3*ξ**2 - 4*ξ + 1,                  0],
                [    0, ell*(ξ**3 - 2*ξ**2 + ξ),                        0,     0,                  0,   3*ξ**2 - 4*ξ + 1],
                [    ξ,                       0,                        0,     0,                  0,                  0],
                [    0,        -2*ξ**3 + 3*ξ**2,                        0,     0,                  0, -6*(-ξ**2 + ξ)/ell],
                [    0,                       0,         -2*ξ**3 + 3*ξ**2,     0,  6*(-ξ**2 + ξ)/ell,                  0],
                [    0,                       0,                        0,     ξ,                  0,                  0],
                [    0,                       0,       -ell*(ξ**3 - ξ**2),     0,       3*ξ**2 - 2*ξ,                  0],
                [    0,       ell*(ξ**3 - ξ**2),                        0,     0,                  0,       3*ξ**2 - 2*ξ]]).T

S__ = sym.integrate(ρ*A*ell*N[0:3, :], (ξ, 0, 1))
S00 = sym.integrate(ρ*A*ell*N[0, :].T*N[0, :], (ξ, 0, 1))
S11 = sym.integrate(ρ*A*ell*N[1, :].T*N[1, :], (ξ, 0, 1))
S22 = sym.integrate(ρ*A*ell*N[2, :].T*N[2, :], (ξ, 0, 1))
S01 = sym.integrate(ρ*A*ell*N[0, :].T*N[1, :], (ξ, 0, 1))
S02 = sym.integrate(ρ*A*ell*N[0, :].T*N[2, :], (ξ, 0, 1))
S12 = sym.integrate(ρ*A*ell*N[1, :].T*N[2, :], (ξ, 0, 1))
S10 = sym.integrate(ρ*A*ell*N[1, :].T*N[0, :], (ξ, 0, 1))
S20 = sym.integrate(ρ*A*ell*N[2, :].T*N[0, :], (ξ, 0, 1))
S21 = sym.integrate(ρ*A*ell*N[2, :].T*N[1, :], (ξ, 0, 1))

D = sym.zeros(6, 12)
D[0, :] = sym.diff(N[0, :]/ell, ξ)
D[1, :] = -sym.diff(N[1, :]/ell**2, ξ, 2)
D[2, :] = -sym.diff(N[2, :]/ell**2, ξ, 2)
D[3, :] = sym.diff(N[3, :]/ell, ξ)
D[4, :] = -sym.diff(N[4, :]/ell, ξ)
D[5, :] = -sym.diff(N[5, :]/ell, ξ)

u = np.zeros([12])
u[6] = 1
u[5] = -np.pi
u[11] = np.pi
ell = 1
def S(ξ):
    A = np.array([[1 - ξ,                       0,                        0,     0,                  0,                  0],
                  [    0,     2*ξ**3 - 3*ξ**2 + 1,                        0,     0,                  0,  6*(-ξ**2 + ξ)/ell],
                  [    0,                       0,      2*ξ**3 - 3*ξ**2 + 1,     0, -6*(-ξ**2 + ξ)/ell,                  0],
                  [    0,                       0,                        0, 1 - ξ,                  0,                  0],
                  [    0,                       0, -ell*(ξ**3 - 2*ξ**2 + ξ),     0,   3*ξ**2 - 4*ξ + 1,                  0],
                  [    0, ell*(ξ**3 - 2*ξ**2 + ξ),                        0,     0,                  0,   3*ξ**2 - 4*ξ + 1],
                  [    ξ,                       0,                        0,     0,                  0,                  0],
                  [    0,        -2*ξ**3 + 3*ξ**2,                        0,     0,                  0, -6*(-ξ**2 + ξ)/ell],
                  [    0,                       0,         -2*ξ**3 + 3*ξ**2,     0,  6*(-ξ**2 + ξ)/ell,                  0],
                  [    0,                       0,                        0,     ξ,                  0,                  0],
                  [    0,                       0,       -ell*(ξ**3 - ξ**2),     0,       3*ξ**2 - 2*ξ,                  0],
                  [    0,       ell*(ξ**3 - ξ**2),                        0,     0,                  0,       3*ξ**2 - 2*ξ]], dtype=float).T
    return A

xi = np.linspace(0, 1, 101)
x = np.zeros([len(xi), 6])
for i in range(len(xi)):
    x[i, :] = S(xi[i])@u
for i in range(6):
    plt.figure()
    plt.plot(xi, x[:, i])
    plt.xlim(0, 1)
plt.figure()
plt.plot(x[:, 0], x[:, 1])
plt.axis("equal")
