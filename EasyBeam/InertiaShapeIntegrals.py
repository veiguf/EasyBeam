#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 11:02:08 2021

@author: veit
"""

import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from sympy.physics.quantum import TensorProduct as kron

def Skew(v):
    s = sym.zeros(3, 3)
    s[0, 1] = -v[2]
    s[1, 0] = v[2]
    s[0, 2] = v[1]
    s[2, 0] = -v[1]
    s[1, 2] = -v[0]
    s[2, 1] = v[0]
    return(s)

def SkewOfMatrix(m):
    S = sym.zeros(3, 3*12)
    for i in range(12):
        S[:, 3*i:3*(i+1)] = Skew(m[:, i])
    return S

ξ = sym.Symbol("xi")
ell = sym.Symbol("ell")
ρ = sym.Symbol("rho")
A = sym.Symbol("A")
x1, y1, z1 = sym.symbols('x1 y1 z1')
x2, y2, z2 = sym.symbols('x2 y2 z2')
ζ1, ζ2, ζ3, ζ4, ζ5, ζ6, ζ7, ζ8, ζ9, ζ10, ζ11, ζ12 = sym.symbols('ζ1 ζ2 ζ3 ζ4 ζ5 ζ6 ζ7 ζ8 ζ9 ζ10 ζ11 ζ12')

u1 = sym.Matrix([x1, y1, z1])
u2 = sym.Matrix([x2, y2, z2])

# uO = ell*sym.Matrix([ξ, 0, 0])

uO = u1+ξ*(u2-u1)
uS = Skew(uO)

S = sym.Matrix([[1 - ξ,                       0,                        0],
                [    0,     2*ξ**3 - 3*ξ**2 + 1,                        0],
                [    0,                       0,      2*ξ**3 - 3*ξ**2 + 1],
                [    0,                       0,                        0],
                [    0,                       0, -ell*(ξ**3 - 2*ξ**2 + ξ)],
                [    0, ell*(ξ**3 - 2*ξ**2 + ξ),                        0],
                [    ξ,                       0,                        0],
                [    0,        -2*ξ**3 + 3*ξ**2,                        0],
                [    0,                       0,         -2*ξ**3 + 3*ξ**2],
                [    0,                       0,                        0],
                [    0,                       0,       -ell*(ξ**3 - ξ**2)],
                [    0,       ell*(ξ**3 - ξ**2),                        0]]).T

SS = SkewOfMatrix(S)

e = sym.diag(1, 1, 1)
ζ = sym.Matrix([ζ1, ζ2, ζ3, ζ4, ζ5, ζ6, ζ7, ζ8, ζ9, ζ10, ζ11, ζ12])
R = sym.diag(1, 1, 1)

m = sym.integrate(ρ*A*ell, (ξ, 0, 1))               # scalar
Xo = sym.integrate(ρ*A*ell*uO, (ξ, 0, 1))           # 3x1
Θo = sym.integrate(ρ*A*ell*uS.T*uS, (ξ, 0, 1))      # 3x3
Iψ = sym.integrate(ρ*A*ell*S, (ξ, 0, 1))            # 3x12
IψS = sym.integrate(ρ*A*ell*SS, (ξ, 0, 1))          # 3x36
IuSψ = sym.integrate(ρ*A*ell*uS.T*S, (ξ, 0, 1))     # 3x12
IuSψS = sym.integrate(ρ*A*ell*uS.T*SS, (ξ, 0, 1))   # 3x36
Iψψ = sym.integrate(ρ*A*ell*S.T*S, (ξ, 0, 1))       # 12x12
IψSψ = sym.integrate(ρ*A*ell*SS.T*S, (ξ, 0, 1))     # 36x12
IψSψS = sym.integrate(ρ*A*ell*SS.T*SS, (ξ, 0, 1))   # 36x36

mtt1 = m*e
mtr1 = -R*(Skew(Xo)+IψS*kron(ζ, e))
mtf1 = R*Iψ
mrr1 = Θo+IuSψS*kron(ζ, e)+kron(ζ, e).T*IuSψS.T+kron(ζ, e).T*IψSψS*kron(ζ, e)
mrf1 = -(IuSψ+kron(ζ, e).T*IψSψ)
mff1 = Iψψ


I1 = sym.integrate(ρ*A*ell, (ξ, 0, 1))              # scalar
I2 = sym.integrate(ρ*A*ell*uO, (ξ, 0, 1))           # 3x1
I3 = sym.integrate(ρ*A*ell*S, (ξ, 0, 1))            # 3x12

Is = [[]]*9
Iv = [[]]*9
Im = [[]]*9
for i in range(3):
    for j in range(3):
        Is[3*i+j] = sym.integrate(ρ*A*ell*uO[i]*uO[j], (ξ, 0, 1))
        Iv[3*i+j] = sym.integrate(ρ*A*ell*uO[i]*S[j, :], (ξ, 0, 1))
        Im[3*i+j] = sym.integrate(ρ*A*ell*S[i, :].T*S[j, :], (ξ, 0, 1))
        print(3*i+j, i+1, j+1)

I4 = sym.Matrix([[Iv[5]-Iv[7]],
                 [Iv[6]-Iv[2]],
                 [Iv[1]-Iv[3]]])                    # 3x12

I5 = sym.Matrix([[Im[5]-Im[7]],
                 [Im[6]-Im[2]],
                 [Im[1]-Im[3]]])                    # 36x12

I6 = Im[0]+Im[4]+Im[8]                              # 12x12

I7 = sym.Matrix([[Is[4]+Is[8], -Is[1], -Is[2]],
                  [-Is[3], Is[0]+Is[8], -Is[5]],
                  [-Is[6], -Is[7], Is[0]+Is[4]]])   # 3x3

I8 = sym.Matrix([[2*(Iv[4]+Iv[8]), -(Iv[1]+Iv[3]), -(Iv[2]+Iv[6])],
                 [-(Iv[1]+Iv[3]), 2*(Iv[0]+Iv[8]), -(Iv[5]+Iv[7])],
                 [-(Iv[2]+Iv[6]), -(Iv[5]+Iv[7]), 2*(Iv[0]+Iv[4])]])    # 3x36

I9 = sym.Matrix([[Im[4]+Im[8],      -Im[1],      -Im[2]],
                 [     -Im[3], Im[0]+Im[8],      -Im[5]],
                 [     -Im[6],      -Im[7], Im[0]+Im[4]]])              # 36x36

mtt2 = I1*e
mtr2 = -R*Skew(I2+I3*ζ)
mtf2 = R*I3
mrr2 = I7+I8*kron(e, ζ)+kron(e, ζ).T*I9*kron(e, ζ)
mrf2 = I4+kron(e, ζ).T*I5
mff2 = I6

v1, v2, v3, v4, v5, v6, v7, v8, v9 = sym.symbols('v1 v2 v3 v4 v5 v6 v7 v8 v9')
x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = sym.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12')
y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12 = sym.symbols('y1 y2 y3 y4 y5 y6 y7 y8 y9 y10 y11 y12')
z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12 = sym.symbols('z1 z2 z3 z4 z5 z6 z7 z8 z9 z10 z11 z12')

v = sym.Matrix([[v1, v2, v3],
                [v4, v5, v6],
                [v7, v8, v9]])

w = sym.Matrix([[x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12],
                [y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12],
                [z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12]])

# Testing
V = SkewOfMatrix(v@w)
W = Skew(v[0, :]).T*SkewOfMatrix(w)
