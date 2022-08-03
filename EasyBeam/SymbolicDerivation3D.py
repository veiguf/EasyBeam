#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 11:02:08 2021

@author: veit
"""

import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

ξ = sym.Symbol("xi")
ell = sym.Symbol("ell")
ρ = sym.Symbol("rho")
A = sym.Symbol("A")

u0 = ell*sym.Matrix([ξ, 0, 0])

def Skew(v):
    s = sym.zeros(3, 3)
    s[0, 1] = -v[2]
    s[1, 0] = v[2]
    s[0, 2] = v[1]
    s[2, 0] = -v[1]
    s[1, 2] = -v[0]
    s[2, 1] = v[0]
    return(s)

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

I1 = sym.Matrix([sym.integrate(ρ*A*ell, (ξ, 0, 1))])
I2 = sym.integrate(ρ*A*ell*u0, (ξ, 0, 1))
I3 = sym.integrate(ρ*A*ell*N[0:3, :], (ξ, 0, 1))

Is = [[]]*9
Iv = [[]]*9
Im = [[]]*9
for i in range(3):
    for j in range(3):
        Is[3*i+j] = sym.integrate(ρ*A*ell*u0[i]*u0[j], (ξ, 0, 1))
        Iv[3*i+j] = sym.integrate(ρ*A*ell*u0[i]*N[j, :], (ξ, 0, 1))
        Im[3*i+j] = sym.integrate(ρ*A*ell*N[i, :].T*N[j, :], (ξ, 0, 1))
        print(3*i+j, i+1, j+1)

I4 = sym.Matrix([[Iv[5]-Iv[7]],
                 [Iv[6]-Iv[2]],
                 [Iv[1]-Iv[3]]])

I5 = sym.Matrix([[Im[5]-Im[7]],
                 [Im[6]-Im[2]],
                 [Im[1]-Im[3]]])

I6 = Im[0]+Im[4]+Im[8]

I7 = sym.Matrix([[Is[4]+Is[8], -Is[1], -Is[2]],
                 [-Is[3], Is[0]+Is[8], -Is[5]],
                 [-Is[6], -Is[7], Is[0]+Is[4]]])

I8 = sym.Matrix([[2*(Iv[4]+Iv[8]), -(Iv[1]+Iv[3]), -(Iv[2]+Iv[6])],
                 [-(Iv[1]+Iv[3]), 2*(Iv[0]+Iv[8]), -(Iv[5]+Iv[7])],
                 [-(Iv[2]+Iv[6]), -(Iv[5]+Iv[7]), 2*(Iv[0]+Iv[4])]])

I9 = sym.Matrix([[Im[4]+Im[8], -Im[1], -Im[2]],
                 [-Im[3], Im[0]+Im[8], -Im[5]],
                 [-Im[6], -Im[7], Im[0]+Im[4]]])

D = sym.zeros(6, 12)
D[0, :] = sym.diff(N[0, :]/ell, ξ)
D[1, :] = -sym.diff(N[1, :]/ell**2, ξ, 2)
D[2, :] = -sym.diff(N[2, :]/ell**2, ξ, 2)
D[3, :] = sym.diff(N[3, :]/ell, ξ)
D[4, :] = -sym.diff(N[4, :]/ell, ξ)
D[5, :] = -sym.diff(N[5, :]/ell, ξ)

# u = np.zeros([12])
# u[6] = 1
# u[5] = -np.pi
# u[11] = np.pi
# ell = 1
# def S(ξ):
#     A = np.array([[1 - ξ,                       0,                        0,     0,                  0,                  0],
#                   [    0,     2*ξ**3 - 3*ξ**2 + 1,                        0,     0,                  0,  6*(-ξ**2 + ξ)/ell],
#                   [    0,                       0,      2*ξ**3 - 3*ξ**2 + 1,     0, -6*(-ξ**2 + ξ)/ell,                  0],
#                   [    0,                       0,                        0, 1 - ξ,                  0,                  0],
#                   [    0,                       0, -ell*(ξ**3 - 2*ξ**2 + ξ),     0,   3*ξ**2 - 4*ξ + 1,                  0],
#                   [    0, ell*(ξ**3 - 2*ξ**2 + ξ),                        0,     0,                  0,   3*ξ**2 - 4*ξ + 1],
#                   [    ξ,                       0,                        0,     0,                  0,                  0],
#                   [    0,        -2*ξ**3 + 3*ξ**2,                        0,     0,                  0, -6*(-ξ**2 + ξ)/ell],
#                   [    0,                       0,         -2*ξ**3 + 3*ξ**2,     0,  6*(-ξ**2 + ξ)/ell,                  0],
#                   [    0,                       0,                        0,     ξ,                  0,                  0],
#                   [    0,                       0,       -ell*(ξ**3 - ξ**2),     0,       3*ξ**2 - 2*ξ,                  0],
#                   [    0,       ell*(ξ**3 - ξ**2),                        0,     0,                  0,       3*ξ**2 - 2*ξ]], dtype=float).T
#     return A

# xi = np.linspace(0, 1, 101)
# x = np.zeros([len(xi), 6])
# for i in range(len(xi)):
#     x[i, :] = S(xi[i])@u
# for i in range(6):
#     plt.figure()
#     plt.plot(xi, x[:, i])
#     plt.xlim(0, 1)
# plt.figure()
# plt.plot(x[:, 0], x[:, 1])
# plt.axis("equal")
