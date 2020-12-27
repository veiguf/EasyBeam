from EasyBeam import Beam2D
import numpy as np


# Initialisiern des Problems
Frame = Beam2D()
Frame.Properties = [['Q12x12', 7.85e-9, 210000, 144, 12**4/12,  6,  -6],
                    ['Q16x16', 7.85e-9, 210000, 256, 16**4/12,  8,  -8],
                    ['Q20x20', 7.85e-9, 210000, 400, 20**4/12, 10, -10],
                    ['QR40x3', 7.85e-9, 210000, 434,    97800, 20, -20]]

# Knoten [mm]
Frame.N = [[   0,    0],
           [2000,    0],
           [4000,    0],
           [6000,    0],
           [1000, 1000],
           [3000, 1000],
           [5000, 1000]]

# Elemente: verbindet die Knoten
Frame.El = [[0, 1, 'Q12x12'],
            [0, 4, 'QR40x3'],
            [1, 2, 'Q20x20'],
            [1, 4, 'Q16x16'],
            [1, 5, 'QR40x3'],
            [4, 5, 'QR40x3'],
            [2, 3, 'Q12x12'],
            [2, 5, 'QR40x3'],
            [2, 6, 'Q16x16'],
            [5, 6, 'QR40x3'],
            [3, 6, 'QR40x3']]

# Randbedingungen und Belastung [N] bzw. [Nmm]
Frame.Disp = [[0, [  0, 0, 'f']],
              [3, ['f', 0, 'f']]]
Frame.Load = [[5, [0, -20000, 0]]]

# Initialisieren des Modells
Frame.Initialize()

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
