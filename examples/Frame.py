#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 19:09:52 2020

@author: veit
"""

from EasyBeam import Beam2D
import numpy as np

Cantilever = Beam2D()
# Knoten [mm]
Cantilever.N = np.array([[   0,    0],
                         [2000,    0],
                         [4000,    0],
                         [6000,    0],
                         [1000, 1000],
                         [3000, 1000],
                         [5000, 1000]])
# Elemente: welche Knoten werden verbunden?
Cantilever.El = np.array([[0, 1],
                          [0, 4],
                          [1, 2],
                          [1, 4],
                          [1, 5],
                          [4, 5],
                          [2, 3],
                          [2, 5],
                          [2, 6],
                          [5, 6],
                          [3, 6]])
# Randbedingungen und Belastung [N] bzw. [Nmm]
Cantilever.BC = [0, 1, 10]
Cantilever.Load = [[16, -20000]]
Cantilever.Initialize()
# Querschnitte
Cantilever.eU = np.ones([Cantilever.nEl, 1])*20    # mm
Cantilever.eL = np.ones([Cantilever.nEl, 1])*-20    # mm
Cantilever.A = np.ones([Cantilever.nEl, 1])*550     # mm^2
Cantilever.I = np.ones([Cantilever.nEl, 1])*85900   # mm^4
# E-Modul
Cantilever.E = np.ones([Cantilever.nEl, 1])*210000  # MPa

Cantilever.Solve()
Cantilever.nStep = 8
Cantilever.ComputeStress()
Cantilever.Scale = 100
Cantilever.PlotResults()
