from EasyBeam import Beam2D
import numpy as np

# Parameter
b = 10          # mm
h = 20          # mm
Fy = -100        # N
l = 1000        # mm
E = 210000      # MPa
rho = 7.85e-9   # t/mm^3
I = b*h**3/12   # mm^4
A = b*h         # mm^2
nEl = 1
nSeg = 1
nu = 0.3

# Initialisiern des Problems
Cantilever = Beam2D()
Cantilever.SizingVariables = [["h", "b"]]
Cantilever.nSeg = 100
Cantilever.stiffMatType = "Euler-Bernoulli"
Cantilever.massMatType = "consistent"
Cantilever.Properties = [['Prop1', rho, E, nu, 1, h, b]]
Cantilever.Nodes = [[]]*(nEl+1)
for i in range(nEl+1):
    Cantilever.Nodes[i] = [l*i/nEl, 0.0]
Cantilever.El = [[]]*(nEl)
for i in range(nEl):
    Cantilever.El[i] = [i+1, i+2]
Cantilever.PropID = ["Prop1"]*nEl
Cantilever.Disp = [[    1, [0, 0, 0]]]
Cantilever.Load = [[nEl+1, [Fy, Fy, 0]]]
Cantilever.nSeg = 5

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

# Cantilever.EigenvalueAnalysis(nEig=3)
# Cantilever.PlotMode(scale=10)
# print('Eigenvalue solver:', Cantilever.EigenvalSolver)

# # Analytical values, continuous beam theory for eigenfrequencies
# print("Analytical results")
# print("maximum stress [MPa]:")
# sigmaMax = np.abs(F*l/I*h/2)
# print(sigmaMax)

# print("maximum displacement [mm]:")
# wMax = F*l**3/(3*E*I)
# print(wMax)

# print("first three bending modes [Hz]:")
# fB1 = 1.87510107**2/(2*np.pi*l**2)*((E*I)/(rho*A))**0.5
# fB2 = 4.69409113**2/(2*np.pi*l**2)*((E*I)/(rho*A))**0.5
# fB3 = 7.85475744**2/(2*np.pi*l**2)*((E*I)/(rho*A))**0.5
# print(fB1, ',', fB2, ',', fB3)

# print("first three longitudinal modes [Hz]:")
# fL1 = 1/(4*l)*(E/rho)**0.5
# fL2 = 3/(4*l)*(E/rho)**0.5
# fL3 = 5/(4*l)*(E/rho)**0.5
# print(fL1, ',', fL2, ',', fL3)
