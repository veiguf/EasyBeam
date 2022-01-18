from EasyBeam import Beam2D

# Initialisiern des Problems
Logo = Beam2D()
b = 10      # mm
h = 20      # mm
Logo.Properties = [['Prop1', 7.85e-9, 210000, 0.3, "rect", h, b]]

# Knoten
Logo.Nodes = [[ 100, 100],  # 1 E
              [   0, 100],  # 2 E
              [   0,  50],  # 3 E
              [ 100,  50],  # 4 E
              [   0,   0],  # 5 E
              [ 100,   0],  # 6 E
              [ 125,   0],  # 7 A
              [ 225, 100],  # 8 A
              [ 225,   0],  # 9 A
              [ 250,   0],  # 10 S
              [ 350,   0],  # 11 S
              [ 350,  50],  # 12 S
              [ 250,  50],  # 13 S
              [ 250, 100],  # 14 S
              [ 350, 100],  # 15 S
              [ 375, 100],  # 16 Y
              [ 425,  50],  # 17 Y
              [ 375,   0],  # 18 Y
              [ 475, 100],  # 19 Y
              [ 500, 100],  # 20 B
              [ 500,  50],  # 21 B
              [ 500,   0],  # 22 B
              [ 600, 100],  # 23 B
              [ 600,  50],  # 24 B
              [ 600,   0],  # 25 B
              [ 625,  75],  # 26 B
              [ 625,  25],  # 27 B
              [ 650,   0],  # 30 E
              [ 650,  50],  # 31 E
              [ 650, 100],  # 32 E
              [ 770, 100],  # 33 E
              [ 750,  50],  # 34 E
              [ 750,   0],  # 35 E
              [ 775,   0],  # 36 A
              [ 875, 100],  # 37 A
              [ 875,   0],  # 38 A
              [ 900,   0],  # 39 M
              [ 900, 100],  # 40 M
              [1000,   0],  # 41 M
              [1000, 100],  # 42 M
              [1100,   0]]  # 43 M

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
           [16, 17],
           [17, 18],
           [17, 19],
           [19, 20],
           [20, 21],
           [21, 22],
           [20, 23],
           [21, 24],
           [22, 25],
           [23, 26],
           [26, 24],
           [24, 27],
           [27, 25],
           [25, 28],
           [28, 29],
           [29, 30],
           [30, 31],
           [29, 32],
           [28, 33],
           [33, 34],
           [34, 35],
           [35, 36],
           [36, 37],
           [37, 38],
           [38, 39],
           [39, 40],
           [40, 41]]

Logo.PropID = ["Prop1"]*len(Logo.El)

# Boundary conditions and loads
Logo.Disp = [[ 5, [0, 0, 0]]]
Logo.Load = [[20, [0, -100, 0]]]

# Lösen
Logo.StaticAnalysis()
Logo.EigenvalueAnalysis(nEig=10)

# Grafische Darstellung
Logo.PlotMesh(ElementNumber=False, FontMag=0.8)
Logo.PlotMesh(NodeNumber=False, FontMag=0.8)
Logo.PlotStress(stress="max", scale=8)
Logo.PlotDisplacement(component="mag", scale=8)
# Logo.PlotMode(scale=1)
