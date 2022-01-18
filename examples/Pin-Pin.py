from EasyBeam import Beam2D
import numpy as np


# Initialisiern des Problems
PinPin = Beam2D()

# Werkstoff und Querschnitt: ID, rho, E, A, I, eU, eL
b = 10      # mm
h = 20      # mm
PinPin.Properties = [['Prop1', 7.85e-9, 210000, 0.3, "rect", b, h],
                     ['Prop2', 2.70e-9,  70000, 0.3, "rect", b, h]]

# Knoten [mm]
PinPin.Nodes = [[  0, 0],
                [150, 0],
                [300, 0]]

# Elemente: verbindet die Knoten
PinPin.El = [[1, 2],
             [2, 3]]

PinPin.PropID = ['Prop1', 'Prop2']

# Randbedingungen und Belastung [N] bzw. [Nmm]
PinPin.Disp = [[1, [0, 0, 'f']],
               [3, [1, 0, 0.1]]]
PinPin.Load = []

PinPin.nSeg = 100

# Lösen
PinPin.StaticAnalysis()
PinPin.EigenvalueAnalysis(nEig=len(PinPin.DoF))

# Grafische Darstellung
PinPin.PlotMesh(FontMag=2)
PinPin.PlotStress(stress='all', scale=10)
PinPin.PlotDisplacement(component='all', scale=10)
PinPin.PlotMode(scale=1)
