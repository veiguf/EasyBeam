from EasyBeam import Beam2D
import numpy as np

PinPin = Beam2D()
# Knoten [mm]
PinPin.N = np.array([[  0, 0],
                     [150, 0],
                     [300, 0]])
# Elemente: welche Knoten werden verbunden?
PinPin.El = np.array([[0, 1],
                      [1, 2]])
# Randbedingungen und Belastung [N] bzw. [Nmm]
PinPin.BC = [0, 1, 7]
PinPin.Load = [[4, -1000]]
PinPin.Initialize()
# Querschnitte
b = 10      # mm
h = 20      # mm
PinPin.eU = np.ones([PinPin.nEl, 1])*h/2    # mm
PinPin.eL = np.ones([PinPin.nEl, 1])*-h/2   # mm
PinPin.A = np.ones([PinPin.nEl, 1])*b*h     # mm^2
PinPin.I = np.ones([PinPin.nEl, 1])*b*h**3/12   # mm^4
# E-Modul
PinPin.E = np.ones([PinPin.nEl, 1])*210000      # MPa

PinPin.Solve()
PinPin.nStep = 100
PinPin.Scale = 100
PinPin.ComputeStress()
PinPin.PlotMesh()
PinPin.PlotStress()
PinPin.PlotDisplacement()
