from EasyBeam import Beam2D
import numpy as np


# Initialisiern des Problems
PinPin = Beam2D()

# Werkstoff und Querschnitt: ID, rho, E, A, I, eU, eL
b = 10      # mm
h = 20      # mm
PinPin.Properties = [['Prop1', 7.85e-9, 210000, 0.3, 1, b, h],
                     ['Prop2', 2.70e-9,  70000, 0.3, 1, b, h]]

# Knoten [mm]
PinPin.Nodes = [[  0, 0],
                [150, 0],
                [300, 0]]

# Elemente: verbindet die Knoten
PinPin.El = [[0, 1],
             [1, 2]]

PinPin.PropID = ['Prop1', 'Prop2']

# Randbedingungen und Belastung [N] bzw. [Nmm]
PinPin.Disp = [[0, [0, 0, 'f']],
               [2, [1, 0, 0.1]]]
PinPin.Load = []

# Initialisieren des Modells
PinPin.Initialize()

# Lösen
PinPin.StaticAnalysis()
PinPin.Scale = 10
PinPin.ComputeStress()
PinPin.EigenvalueAnalysis(nEig=len(PinPin.DoF))

# Grafische Darstellung
PinPin.PlotMesh()
PinPin.PlotStress()
PinPin.PlotDisplacement()
PinPin.ScalePhi = 1
PinPin.PlotMode()
