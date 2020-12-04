from EasyBeam import Beam2D
import numpy as np


# Initialisiern des Problems
Cantilever = Beam2D()

# Knoten [mm]
Cantilever.N = np.array([[  0, 0],
                         [300, 0]])

# Elemente: verbindet die Knoten
Cantilever.El = np.array([[0, 1]])

# Randbedingungen und Belastung [N] bzw. [Nmm]
Cantilever.BC = [0, 1, 2]
Cantilever.Load = [[4, -1000]]

# Initialisieren des Modells
Cantilever.Initialize()

# Querschnittgeometrie und Werkstoff
b = 10      # mm
h = 20      # mm
Cantilever.eU = np.ones([Cantilever.nEl, 1])*h/2    # mm
Cantilever.eL = np.ones([Cantilever.nEl, 1])*-h/2    # mm
Cantilever.A = np.ones([Cantilever.nEl, 1])*b*h     # mm^2
Cantilever.I = np.ones([Cantilever.nEl, 1])*b*h**3/12   # mm^4
Cantilever.E = np.ones([Cantilever.nEl, 1])*210000      # MPa

# Lösen
Cantilever.StaticAnalysis()
Cantilever.Scale = 10
Cantilever.ComputeStress()

# Grafische Darstellung
Cantilever.PlotMesh()
Cantilever.PlotStress()
Cantilever.PlotDisplacement()
