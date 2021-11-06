from EasyBeam import Beam2D

# Initialisiern des Problems
Logo = Beam2D()
b = 10      # mm
h = 20      # mm
Logo.Properties = [['Prop1', 7.85e-9, 210000, 0.3, 1, h, b]]

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
              [ 585, 100],  # 23 B
              [ 585,  50],  # 24 B
              [ 585,   0],  # 25 B
              [ 600,  85],  # 26 B
              [ 600,  65],  # 27 B
              [ 600,  35],  # 28 B
              [ 600,  15],  # 29 B
              [ 625,   0],  # 30 E
              [ 625,  50],  # 31 E
              [ 625, 100],  # 32 E
              [ 725, 100],  # 33 E
              [ 725,  50],  # 34 E
              [ 725,   0],  # 35 E
              [ 750,   0],  # 36 A
              [ 850, 100],  # 37 A
              [ 850,   0],  # 38 A
              [ 875,   0],  # 39 M
              [ 875, 100],  # 40 M
              [ 975,   0],  # 41 M
              [ 975, 100],  # 42 M
              [1075,   0]]  # 43 M

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
           [26, 27],
           [27, 24],
           [24, 28],
           [28, 29],
           [29, 25],
           [25, 30],
           [30, 31],
           [31, 32],
           [32, 33],
           [31, 34],
           [30, 35],
           [35, 36],
           [36, 37],
           [37, 38],
           [38, 39],
           [39, 40],
           [40, 41],
           [41, 42],
           [42, 43]]

Logo.PropID = ["Prop1"]*len(Logo.El)

# Boundary conditions and loads
Logo.Disp = [[ 5, [0, 0, 0]]]
Logo.Load = [[43, [0, -100, 0]]]

# Lösen
Logo.StaticAnalysis()
Logo.EigenvalueAnalysis(nEig=10)

# Grafische Darstellung
Logo.PlotMesh(ElementNumber=False, FontMag=0.8)
Logo.PlotMesh(NodeNumber=False, FontMag=0.8)
Logo.PlotStress(stress="max", scale=2)
Logo.PlotDisplacement(component="mag", scale=2)
# Logo.PlotMode(scale=1)
