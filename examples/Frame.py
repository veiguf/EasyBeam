#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 19:09:52 2020

@author: veit
"""

from EasyBeam import Beam2D
import numpy as np

Frame = Beam2D()
# Knoten [mm]
Frame.N = np.array([[   0,    0],
                    [2000,    0],
                    [4000,    0],
                    [6000,    0],
                    [1000, 1000],
                    [3000, 1000],
                    [5000, 1000]])
# Elemente: welche Knoten werden verbunden?
Frame.El = np.array([[0, 1],
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
Frame.BC = [0, 1, 10]
Frame.Load = [[16, -20000]]
Frame.Initialize()
# Querschnitte
Frame.eU = np.ones([Frame.nEl, 1])*20    # mm
Frame.eL = np.ones([Frame.nEl, 1])*-20    # mm
Frame.A = np.ones([Frame.nEl, 1])*550     # mm^2
Frame.I = np.ones([Frame.nEl, 1])*85900   # mm^4
# E-Modul
Frame.E = np.ones([Frame.nEl, 1])*210000  # MPa

Frame.Solve()
Frame.nStep = 8
Frame.Scale = 100
Frame.ComputeStress()
Frame.PlotStressUpperFibre()
Frame.PlotStressLowerFibre()
Frame.PlotStressMaximum()
Frame.PlotDisplacement()
