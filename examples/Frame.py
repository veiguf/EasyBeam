from EasyBeam import Beam2D
import numpy as np


# Initialisiern des Problems
Frame = Beam2D()
Frame.Properties = [['Q12x12', 7.85e-9, 210000, 0.3, 1, 12, 12],
                    ['Q16x16', 7.85e-9, 210000, 0.3, 1, 16, 16],
                    ['Q20x20', 7.85e-9, 210000, 0.3, 1, 20, 10],
                    ['QR20x3', 7.85e-9, 210000, 0.3, 3, 20,  3]]

# Knoten [mm]
Frame.Nodes= [[   0,    0],
              [2000,    0],
              [4000,    0],
              [6000,    0],
              [1000, 1000],
              [3000, 1000],
              [5000, 1000]]

# Elemente: verbindet die Knoten
Frame.El = [[1, 2],
            [1, 5],
            [2, 3],
            [2, 5],
            [2, 6],
            [5, 6],
            [3, 4],
            [3, 6],
            [3, 7],
            [6, 7],
            [4, 7]]

Frame.PropID = ['Q12x12', 'QR20x3', 'Q20x20', 'Q16x16', 'QR20x3', 'QR20x3',
                'Q12x12', 'QR20x3', 'Q16x16', 'QR20x3', 'QR20x3']

# Randbedingungen und Belastung [N] bzw. [Nmm]
Frame.Disp = [[1, [  0, 0, 'f']],
              [4, ['f', 0, 'f']]]
Frame.Load = [[6, [0, -20000, 0]]]

Frame.nSeg = 20

# Lösen
Frame.StaticAnalysis()
Frame.EigenvalueAnalysis(nEig=len(Frame.DoF))

# Grafische Darstellung
Frame.PlotMesh(FontMag=2)
Frame.PlotStress(stress="all", scale=100)
Frame.PlotDisplacement(component="all", scale=100)
Frame.PlotMode(scale=30)
