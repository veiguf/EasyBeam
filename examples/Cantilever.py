from EasyBeam import Beam2D
import numpy as np

Cantilever = Beam2D()
# Knoten [mm]
Cantilever.N = np.array([[  0, 0],
                         [300, 0]])
# Elemente: welche Knoten werden verbunden?
Cantilever.El = np.array([[0, 1]])
# Randbedingungen und Belastung [N] bzw. [Nmm]
Cantilever.BC = [0, 1, 2]
Cantilever.Load = [[4, -1000]]
Cantilever.Initialize()
# Querschnitte
b = 10      # mm
h = 20      # mm
Cantilever.eU = np.ones([Cantilever.nEl, 1])*h/2    # mm
Cantilever.eL = np.ones([Cantilever.nEl, 1])*-h/2    # mm
Cantilever.A = np.ones([Cantilever.nEl, 1])*b*h     # mm^2
Cantilever.I = np.ones([Cantilever.nEl, 1])*b*h**3/12   # mm^4
# E-Modul
Cantilever.E = np.ones([Cantilever.nEl, 1])*210000      # MPa

Cantilever.Solve()
Cantilever.nStep = 100
Cantilever.Scale = 10
Cantilever.ComputeStress()

Cantilever.PlotMesh()
Cantilever.PlotStress()
Cantilever.PlotDisplacement()
