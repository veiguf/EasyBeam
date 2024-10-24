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
    x = len(m[0, :])
    S = sym.zeros(3, 3*x)
    for i in range(x):
        S[:, 3*i:3*(i+1)] = Skew(m[:, i])
    return S

# ξ, η, ζ = sym.Symbol("xi eta zeta")
x, y, z = sym.symbols("x y z")
w, h, ell = sym.symbols("w h ell")
ρ = sym.Symbol("rho")
A = sym.Symbol("A")
Ix = sym.Symbol("Ix")
Iy = sym.Symbol("Iy")
Iz = sym.Symbol("Iz")

x1, y1, z1 = sym.symbols('x1 y1 z1')
# x2, y2, z2 = sym.symbols('x2 y2 z2')

# c1 = (x2-x1)/ell
# c2 = (y2-y1)/ell
# c3 = (z2-z1)/ell

# T = sym.Matrix([[c1, -(c1*c2)/sym.sqrt(c1**2+c3**2), -c3/sym.sqrt(c1**2+c3**2)],
#                 [c2,          sym.sqrt(c1**2+c3**2),                     0],
#                 [c3, -(c2*c3)/sym.sqrt(c1**2+c3**2),  c1/sym.sqrt(c1**2+c3**2)]])

T11, T12, T13, T21, T22, T23, T31, T32, T33 = sym.symbols("T11 T12 T13 T21 T22 T23 T31 T32 T33")
T = sym.Matrix([[T11, T12, T13],
                [T21, T22, T23],
                [T31, T32, T33]])

ξ = x/ell
η = y/ell  # 0
ζ = z/ell  # 0

u1 = sym.Matrix([x1, y1, z1])
u1s = Skew(u1)

uL = ell*sym.Matrix([ξ, η, ζ])
uLs = Skew(uL)

uX = T*uL
uXs = Skew(uX)

uO = u1+uX
uOs = Skew(uO)

S = sym.Matrix([[                  1-ξ,                   0,                    0],
                [         6*(ξ-ξ**2)*η,     1-3*ξ**2+2*ξ**3,                    0],
                [         6*(ξ-ξ**2)*ζ,                   0,      1-3*ξ**2+2*ξ**3],
                [                    0,        -(1-ξ)*ell*ζ,          (1-ξ)*ell*η],
                [ (1-4*ξ+3*ξ**2)*ell*ζ,                   0, (-ξ+2*ξ**2-ξ**3)*ell],
                [(-1+4*ξ-3*ξ**2)*ell*η, (ξ-2*ξ**2+ξ**3)*ell,                    0],
                [                    ξ,                   0,                    0],
                [        6*(-ξ+ξ**2)*η,       3*ξ**2-2*ξ**3,                    0],
                [        6*(-ξ+ξ**2)*ζ,                   0,        3*ξ**2-2*ξ**3],
                [                    0,            -ell*ξ*ζ,              ell*ξ*η],
                [  (-2*ξ+3*ξ**2)*ell*ζ,                   0,      (ξ**2-ξ**3)*ell],
                [   (2*ξ-3*ξ**2)*ell*η,    (-ξ**2+ξ**3)*ell,                    0]]).T

SS = SkewOfMatrix(S)

printMatrices = True

m = sym.integrate(ρ, (x, 0, ell), (y, -w/2, w/2), (z, -h/2, h/2))               # scalar
m = m.simplify()
m = m.subs(w*h, A)

Xo = sym.integrate(ρ*uO, (x, 0, ell), (y, -w/2, w/2), (z, -h/2, h/2))           # 3x1
Xo = Xo.simplify()
Xo = Xo.subs(w*h, A)
if printMatrices:
    print("\n Xo \n", Xo)

Θo = sym.integrate(ρ*uOs.T*uOs, (x, 0, ell), (y, -w/2, w/2), (z, -h/2, h/2))      # 3x3
Θo = Θo.simplify()
Θo = Θo.subs(w*h, A)
Θo = Θo.subs(h**2, 12*Iy/A)
Θo = Θo.subs(w**2, 12*Iz/A)
Θo = Θo.simplify()
Θo = Θo.subs(Iy+Iz, Ix)

if printMatrices:
    print("\n Θo \n", Θo)

Iψ = sym.integrate(ρ*S, (x, 0, ell), (y, -w/2, w/2), (z, -h/2, h/2))            # 3x12
Iψ = Iψ.simplify()
Iψ = Iψ.subs(w*h, A)
if printMatrices:
    print("\n Iψ.T \n", Iψ.T)

IψS = sym.integrate(ρ*SS, (x, 0, ell), (y, -w/2, w/2), (z, -h/2, h/2))          # 3x36
IψS = IψS.simplify()
IψS = IψS.subs(w*h, A)
if printMatrices:
    print("\n IψS.T \n", IψS.T)

IuSψ = sym.integrate(ρ*uOs.T*S, (x, 0, ell), (y, -w/2, w/2), (z, -h/2, h/2))     # 3x12
IuSψ = IuSψ.simplify()
IuSψ = IuSψ.subs(w*h, A)
IuSψ = IuSψ.subs(h**2, 12*Iy/A)
IuSψ = IuSψ.subs(w**2, 12*Iz/A)
IuSψ = IuSψ.simplify()
IuSψ = IuSψ.subs(Iy+Iz, Ix)
if printMatrices:
    print("\n IuSψ.T \n", IuSψ.T)

IuSψS = sym.integrate(ρ*uOs.T*SS, (x, 0, ell), (y, -w/2, w/2), (z, -h/2, h/2))   # 3x36
IuSψS = IuSψS.simplify()
IuSψS = IuSψS.subs(w*h, A)
IuSψS = IuSψS.subs(h**2, 12*Iy/A)
IuSψS = IuSψS.subs(w**2, 12*Iz/A)
IuSψS = IuSψS.simplify()
IuSψS = IuSψS.subs(Iy+Iz, Ix)
if printMatrices:
    print("\n IuSψS.T \n", IuSψS.T)

Iψψ = sym.integrate(ρ*S.T*S, (x, 0, ell), (y, -w/2, w/2), (z, -h/2, h/2))       # 12x12
# Iψψ = Iψψ/(A*ell*ρ)
Iψψ = Iψψ.simplify()
Iψψ = Iψψ.subs(w*h, A)
Iψψ = Iψψ.subs(h**2, 12*Iy/A)
Iψψ = Iψψ.subs(w**2, 12*Iz/A)
Iψψ = Iψψ.simplify()
Iψψ = Iψψ.subs(Iy+Iz, Ix)

# IψψTorsion = sym.zeros(12, 12)
# IψψTorsion[3, 3] = IψψTorsion[9, 9] = A*ell*ρ*Ix/(3*A)
# IψψTorsion[3, 9] = IψψTorsion[9, 3] = A*ell*ρ*Ix/(6*A)
# Iψψ += IψψTorsion
if printMatrices:
    print("\n Iψψ \n", Iψψ)

IψSψ = sym.integrate(ρ*SS.T*S, (x, 0, ell), (y, -w/2, w/2), (z, -h/2, h/2))     # 36x12
IψSψ = IψSψ.simplify()
IψSψ = IψSψ.subs(w*h, A)
IψSψ = IψSψ.subs(h**2, 12*Iy/A)
IψSψ = IψSψ.subs(w**2, 12*Iz/A)
IψSψ = IψSψ.simplify()
IψSψ = IψSψ.subs(Iy+Iz, Ix)
if printMatrices:
    print("\n IψSψ \n", IψSψ)

IψSψS = sym.integrate(ρ*SS.T*SS, (x, 0, ell), (y, -w/2, w/2), (z, -h/2, h/2))   # 36x36
IψSψS = IψSψS.simplify()
IψSψS = IψSψS.subs(w*h, A)
IψSψS = IψSψS.subs(h**2, 12*Iy/A)
IψSψS = IψSψS.subs(w**2, 12*Iz/A)
IψSψS = IψSψS.simplify()
IψSψS = IψSψS.subs(Iy+Iz, Ix)
if printMatrices:
    print("\n IψSψS \n", IψSψS)

# HRZ lumping from Felippa et al (2014) Appendix 4.1:
mlum = sym.zeros(12, 12)
for i in range(3):
    Sum = Iψψ[i, i]+Iψψ[6+i, 6+i]
    mlum[i, i] = (Iψψ[i, i]/Sum).simplify()
    mlum[3+i, 3+i] = (Iψψ[3+i, 3+i]/Sum).simplify()
    mlum[6+i, 6+i] = (Iψψ[6+i, 6+i]/Sum).simplify()
    mlum[9+i, 9+i] = (Iψψ[9+i, 9+i]/Sum).simplify()

"""
###############################################################################

e = sym.diag(1, 1, 1)
ζ1, ζ2, ζ3, ζ4, ζ5, ζ6, ζ7, ζ8, ζ9, ζ10, ζ11, ζ12 = sym.symbols('ζ1 ζ2 ζ3 ζ4 ζ5 ζ6 ζ7 ζ8 ζ9 ζ10 ζ11 ζ12')
ζ = sym.Matrix([ζ1, ζ2, ζ3, ζ4, ζ5, ζ6, ζ7, ζ8, ζ9, ζ10, ζ11, ζ12])
R = sym.diag(1, 1, 1)

mtt1 = m*e
mtr1 = -R*(Skew(Xo)+IψS*kron(ζ, e))
mtf1 = R*Iψ
mrr1 = Θo+IuSψS*kron(ζ, e)+kron(ζ, e).T*IuSψS.T+kron(ζ, e).T*IψSψS*kron(ζ, e)
mrf1 = -(IuSψ+kron(ζ, e).T*IψSψ)
mff1 = Iψψ

###############################################################################

I1 = sym.integrate(ρ, (x, 0, ell), (y, -w/2, w/2), (z, -h/2, h/2))              # scalar
I1 = I1.simplify()
I1 = I1.subs(w*h, A)

I2 = sym.integrate(ρ*uO, (x, 0, ell), (y, -w/2, w/2), (z, -h/2, h/2))           # 3x1
I2 = I2.simplify()
I2 = I2.subs(w*h, A)

I3 = sym.integrate(ρ*S, (x, 0, ell), (y, -w/2, w/2), (z, -h/2, h/2))            # 3x12
I3 = I3.simplify()
I3 = I3.subs(w*h, A)

Is = [[]]*9
Iv = [[]]*9
Im = [[]]*9
for i in range(3):
    for j in range(3):
        Is[3*i+j] = sym.integrate(ρ*uO[i]*uO[j], (x, 0, ell), (y, -w/2, w/2), (z, -h/2, h/2))
        Iv[3*i+j] = sym.integrate(ρ*uO[i]*S[j, :], (x, 0, ell), (y, -w/2, w/2), (z, -h/2, h/2))
        Im[3*i+j] = sym.integrate(ρ*S[i, :].T*S[j, :], (x, 0, ell), (y, -w/2, w/2), (z, -h/2, h/2))
        # print(3*i+j, i+1, j+1)

I4 = sym.Matrix([[Iv[5]-Iv[7]],
                 [Iv[6]-Iv[2]],
                 [Iv[1]-Iv[3]]])                    # 3x12
# I4 = I4.simplify()
I4 = I4.subs(w*h, A)

I5 = sym.Matrix([[Im[5]-Im[7]],
                 [Im[6]-Im[2]],
                 [Im[1]-Im[3]]])                    # 36x12
# I5 = I5.simplify()
I5 = I5.subs(w*h, A)

I6 = Im[0]+Im[4]+Im[8]                              # 12x12
# I6 = I6.simplify()
I6 = I6.subs(w*h, A)

I7 = sym.Matrix([[Is[4]+Is[8], -Is[1], -Is[2]],
                  [-Is[3], Is[0]+Is[8], -Is[5]],
                  [-Is[6], -Is[7], Is[0]+Is[4]]])   # 3x3
# I7 = I7.simplify()
I7 = I7.subs(w*h, A)

I8 = sym.Matrix([[2*(Iv[4]+Iv[8]), -(Iv[1]+Iv[3]), -(Iv[2]+Iv[6])],
                 [-(Iv[1]+Iv[3]), 2*(Iv[0]+Iv[8]), -(Iv[5]+Iv[7])],
                 [-(Iv[2]+Iv[6]), -(Iv[5]+Iv[7]), 2*(Iv[0]+Iv[4])]])    # 3x36
# I8 = I8.simplify()
I8 = I8.subs(w*h, A)

I9 = sym.Matrix([[Im[4]+Im[8],      -Im[1],      -Im[2]],
                 [     -Im[3], Im[0]+Im[8],      -Im[5]],
                 [     -Im[6],      -Im[7], Im[0]+Im[4]]])              # 36x36
# I9 = I9.simplify()
I9 = I9.subs(w*h, A)

###############################################################################

mtt2 = I1*e
mtr2 = -R*Skew(I2+I3*ζ)
mtf2 = R*I3
mrr2 = I7+I8*kron(e, ζ)+kron(e, ζ).T*I9*kron(e, ζ)
mrf2 = I4+kron(e, ζ).T*I5
mff2 = I6

###############################################################################

print(mtt1-mtt2)
print(mtr1-mtr2)
print(mtf1-mtf2)
print((mrr1-mrr2).simplify())
print((mrf1-mrf2).simplify())
print((mff1-mff2).simplify())
"""
