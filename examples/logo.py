from EasyBeam import Beam2D
import numpy as np


# Initialisiern des Problems
Logo = Beam2D()

# Knoten
Logo.N = np.array([[ 100, 100],  # 0 E
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
                   [ 950,   0]])  # 28 M

# Elemente: verbindet die Knoten
Logo.El = np.array([[0, 1],
                    [1, 2],
                    [2, 3],
                    [2, 4],
                    [4, 5],
                    [5, 6],
                    [6, 7],
                    [7, 8],
                    [8, 9],
                    [9, 10],
                    [10, 11],
                    [11, 12],
                    [12, 13],
                    [13, 14],
                    [14, 15],
                    [14,  9],
                    [15, 16],
                    [16, 17],
                    [17, 18],
                    [18, 19],
                    [16, 19],
                    [20, 19],
                    [15, 20],
                    [21, 20],
                    [22, 19],
                    [18, 23],
                    [23, 24],
                    [25, 24],
                    [24, 26],
                    [26, 27],
                    [27, 28]])

# Boundary conditions and loads
Logo.BC = [0, 1, 2]
Logo.Load = [[26, -10],
             [27, -10],
             [18, -10],
             [19, -10]]

# Initialisieren des Modells
Logo.Initialize()

# Querschnittgeometrie und Werkstoff
b = 10      # mm
h = 10      # mm
Logo.eU = np.ones([Logo.nEl, 1])*h/2
Logo.eL = np.ones([Logo.nEl, 1])*-h/2
Logo.A = np.ones([Logo.nEl, 1])*b*h     # mm^2
Logo.I = np.ones([Logo.nEl, 1])*b*h**3/12    # mm^4
Logo.E = np.ones([Logo.nEl, 1])*210000        # MPa

# Lösen
Logo.StaticAnalysis()
Logo.Scale = 10
Logo.ComputeStress()

# Grafische Darstellung
Logo.PlotMesh(ElementNumber=False)
Logo.PlotMesh(NodeNumber=False)
Logo.PlotStress(stress="all")
Logo.PlotDisplacement()
Logo.PlotDisplacement()
