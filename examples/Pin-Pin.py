from EasyBeam import Beam2D
import numpy as np


# Initialisiern des Problems
PinPin = Beam2D()

# Knoten [mm]
PinPin.N = np.array([[  0, 0],
                     [150, 0],
                     [300, 0]])

# Elemente: verbindet die Knoten
PinPin.El = np.array([[0, 1],
                      [1, 2]])

# Randbedingungen und Belastung [N] bzw. [Nmm]
PinPin.BC = [0, 1, 7]
PinPin.Load = [[4, -1000]]

# Initialisieren des Modells
PinPin.Initialize()

# Querschnittgeometrie und Werkstoff
b = 10      # mm
h = 20      # mm
PinPin.eU = np.ones([PinPin.nEl, 1])*h/2    # mm
PinPin.eL = np.ones([PinPin.nEl, 1])*-h/2   # mm
PinPin.A = np.ones([PinPin.nEl, 1])*b*h     # mm^2
PinPin.I = np.ones([PinPin.nEl, 1])*b*h**3/12   # mm^4
PinPin.E = np.ones([PinPin.nEl, 1])*210000      # MPa

# Lösen
PinPin.StaticAnalysis()
PinPin.Scale = 100
PinPin.ComputeStress()

# Grafische Darstellung
PinPin.PlotMesh()
PinPin.PlotStress()
PinPin.PlotDisplacement()
