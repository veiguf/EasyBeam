from EasyBeam import Beam2D
import numpy as np


# Initialisiern des Problems
Logo = Beam2D()
b = 10      # mm
h = 20      # mm
Logo.Properties = [['Prop1', 7.85e-9, 210000, 0.3, 1, h, b]]

# Knoten
Logo.Nodes = [[ 100, 100],  # 0 E
              [   0, 100],  # 1 E
              [   0,  50],  # 2 E
              [ 100,  50],  # 3 E
              [   0,   0],  # 4 E
              [ 100,   0],  # 5 E/A
              [ 150,  50],  # 6 A
              [ 200, 100],  # 7 A
              [ 200,   0],  # 8 A/S
              [ 350,   0],  # 9 S/Y
              [ 350,  50],  # 10 S
              [ 250,  50],  # 11 S
              [ 250, 100],  # 12 S
              [ 350, 100],  # 13 S/Y
              [ 400,  50],  # 14 Y
              [ 450, 100],  # 15 Y/B
              [ 450,  50],  # 16 B
              [ 450,   0],  # 17 B
              [ 550,   0],  # 18 B/E
              [ 550,  50],  # 19 B/E
              [ 550, 100],  # 20 B/E
              [ 650, 100],  # 21 E
              [ 650,  50],  # 22 E
              [ 650,   0],  # 23 E/A
              [ 750, 100],  # 24 A/M
              [ 750,   0],  # 25 A/M
              [ 850,   0],  # 26 M
              [ 850, 100],  # 27 M
              [ 950,   0]]  # 28 M

# Elemente: verbindet die Knoten
Logo.El = [[ 1,  2],
           [ 2,  3],
           [ 3,  4],
           [ 3,  5],
           [ 5,  6],
           [ 6,  7],
           [ 7,  8],
           [ 8,  9],
           [ 9, 10],
           [10, 11],
           [11, 12],
           [12, 13],
           [13, 14],
           [14, 15],
           [15, 16],
           [15, 10],
           [16, 17],
           [17, 18],
           [18, 19],
           [19, 20],
           [17, 20],
           [21, 20],
           [16, 21],
           [22, 21],
           [23, 20],
           [19, 24],
           [24, 25],
           [26, 25],
           [25, 27],
           [27, 28],
           [28, 29]]

Logo.PropID = ["Prop1"]*len(Logo.El)

# Boundary conditions and loads
Logo.Disp = [[ 5, [0, 0, 0]]]
Logo.Load = [[29, [0, -10, 0]]]

# Lösen
Logo.StaticAnalysis()
Logo.EigenvalueAnalysis(nEig=10)

# Grafische Darstellung
Logo.PlotMesh(ElementNumber=False, FontMag=1.5)
Logo.PlotMesh(NodeNumber=False, FontMag=1.5)
Logo.PlotStress(stress="all", scale=5)
Logo.PlotDisplacement(component="all", scale=5)
Logo.PlotMode(scale=1)
