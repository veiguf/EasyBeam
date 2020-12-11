from EasyBeam import Beam2D
import numpy as np


# Initialisiern des Problems
Frame = Beam2D()

# Knoten [mm]
Frame.N = [[   0,    0],
           [2000,    0],
           [4000,    0],
           [6000,    0],
           [1000, 1000],
           [3000, 1000],
           [5000, 1000]]

# Elemente: verbindet die Knoten
Frame.El = [[0, 1],
            [0, 4],
            [1, 2],
            [1, 4],
            [1, 5],
            [4, 5],
            [2, 3],
            [2, 5],
            [2, 6],
            [5, 6],
            [3, 6]]

# Randbedingungen und Belastung [N] bzw. [Nmm]
Frame.BC = [0, 1, 10]
Frame.Load = [[16, -20000]]

# Initialisieren des Modells
Frame.Initialize()

# Querschnittgeometrie und Werkstoff
Frame.eU = np.ones([Frame.nEl, 1])*20    # mm
Frame.eL = np.ones([Frame.nEl, 1])*-20    # mm
Frame.A = np.ones([Frame.nEl, 1])*550     # mm^2
Frame.I = np.ones([Frame.nEl, 1])*85900   # mm^4
Frame.E = np.ones([Frame.nEl, 1])*210000  # MPa
Frame.rho = np.ones([Frame.nEl, 1])*7.85e-9   # t/mm^3

# Lösen
Frame.StaticAnalysis()
Frame.Scale = 100
Frame.ComputeStress()
Frame.EigenvalueAnalysis(nEig=len(Frame.DoF))

# Grafische Darstellung
Frame.PlotMesh()
Frame.PlotStress(stress="all")
Frame.PlotDisplacement()
Frame.ScalePhi = 30
Frame.PlotMode()
