from EasyBeam import Beam2D

# Initialisiern des Problems
class Model(Beam2D):
    b_x = 10          # mm
    h_x = 20          # mm
    F_x = -100        # N
    l_x = 1000        # mm
    E_x = 210000      # MPa
    rho_x = 7.85e-9   # t/mm^3
    nu_x = 0.3
    DesVar = ["h_x", "b_x"]
    def __init__(self):
        self.nEl = 1
        self.nSeg = 1
        self.stiffMatType = "Euler-Bernoulli"
        self.massMatType = "consistent"
        self.Properties = [['Prop1', self.rho_x, self.E_x, self.nu_x, 1, self.h_x, self.b_x]]
        self.Nodes = [[]]*(self.nEl+1)
        for i in range(self.nEl+1):
            self.Nodes[i] = [self.l_x*i/self.nEl, 0.0]
        self.El = [[]]*(self.nEl)
        for i in range(self.nEl):
            self.El[i] = [i+1, i+2]
        self.PropID = ["Prop1"]*self.nEl
        self.Disp = [[         1, [       0,        0, 0]]]
        self.Load = [[self.nEl+1, [self.F_x, self.F_x, 0]]]

Cantilever = Model()
# Lösen
Cantilever.StaticAnalysis()

# Darstellung
Cantilever.PlotMesh()
Cantilever.PlotDisplacement(scale=10)
Cantilever.PlotStress(scale=10)

# Sensitivitäten
Cantilever.SensitivityAnalysis()
Cantilever.ComputeStressSensitivity()

print("deformation")
print(Cantilever.u)
print("strain")
print(Cantilever.epsilonL)
print(Cantilever.epsilonU)
print("stress")
print(Cantilever.sigmaL)
print(Cantilever.sigmaU)
print()
print("deformation sensitivity")
print(Cantilever.uNabla)
print("strain sensitivity")
print(Cantilever.epsilonLNabla)
print(Cantilever.epsilonUNabla)
print("stress sensitivity")
print(Cantilever.sigmaLNabla)
print(Cantilever.sigmaUNabla)
