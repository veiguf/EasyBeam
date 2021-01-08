from EasyBeam import Beam2D
import numpy as np


# Initialisiern des Problems
Logo = Beam2D()
b = 10      # mm
h = 20      # mm
Logo.Properties = [['Prop1', 7.85e-9, 210000, b*h, b*h**3/12, h/2, -h/2]]

# Knoten
Logo.N = [[ 100, 100],  # 0 E
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
Logo.El = [[ 0,  1, 'Prop1'],
           [ 1,  2, 'Prop1'],
           [ 2,  3, 'Prop1'],
           [ 2,  4, 'Prop1'],
           [ 4,  5, 'Prop1'],
           [ 5,  6, 'Prop1'],
           [ 6,  7, 'Prop1'],
           [ 7,  8, 'Prop1'],
           [ 8,  9, 'Prop1'],
           [ 9, 10, 'Prop1'],
           [10, 11, 'Prop1'],
           [11, 12, 'Prop1'],
           [12, 13, 'Prop1'],
           [13, 14, 'Prop1'],
           [14, 15, 'Prop1'],
           [14,  9, 'Prop1'],
           [15, 16, 'Prop1'],
           [16, 17, 'Prop1'],
           [17, 18, 'Prop1'],
           [18, 19, 'Prop1'],
           [16, 19, 'Prop1'],
           [20, 19, 'Prop1'],
           [15, 20, 'Prop1'],
           [21, 20, 'Prop1'],
           [22, 19, 'Prop1'],
           [18, 23, 'Prop1'],
           [23, 24, 'Prop1'],
           [25, 24, 'Prop1'],
           [24, 26, 'Prop1'],
           [26, 27, 'Prop1'],
           [27, 28, 'Prop1']]

# Boundary conditions and loads
Logo.Disp = [[ 4, [0, 0, 0]]]
Logo.Load = [[28, [0, -10, 0]]]

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
Logo.rho = np.ones([Logo.nEl, 1])*7.85e-9   # t/mm^3

# Lösen
Logo.StaticAnalysis()
Logo.Scale = 5
Logo.ComputeStress()
Logo.EigenvalueAnalysis(nEig=10)

# Grafische Darstellung
Logo.PlotMesh(ElementNumber=False)
Logo.PlotMesh(NodeNumber=False)
Logo.PlotStress(stress="all")
Logo.PlotDisplacement()
Logo.PlotMode()
