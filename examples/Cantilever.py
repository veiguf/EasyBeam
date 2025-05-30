from EasyBeam import Beam2D
import numpy as np

# Parameter
b = 10          # mm
h = 20          # mm
F = -100        # N
l = 1000        # mm
E = 210000      # MPa
rho = 7.85e-9   # t/mm^3
I = b*h**3/12   # mm^4
A = b*h         # mm^2
nEl = 5
nu = 0.3

# Initialisiern des Problems
Cantilever = Beam2D()

Cantilever.deadLoad = True

# Setze Elementenbeschreibung
Cantilever.stiffMatType = "Euler-Bernoulli"  # Euler-Bernoulli or Timoshenko-Ehrenfest
Cantilever.massMatType = "consistent"        # lumped or consistent

# Knoten [mm]
Cantilever.Nodes = [[]]*(nEl+1)
for i in range(nEl+1):
    Cantilever.Nodes[i] = [l*i/nEl, 0.0]

# Elemente: verbindet die Knoten
Cantilever.El = [[]]*(nEl)
for i in range(nEl):
    Cantilever.El[i] = [i+1, i+2]

# Randbedingungen und Belastung [N] bzw. [Nmm]
Cantilever.Disp = [[    1, [0, 0, 0]]]
Cantilever.Load = [[nEl+1, [0, F, 0]]]

# Werkstoff und Querschnitt: ID, rho, E, A, I, eU, eL
Cantilever.Properties = [['Prop1', rho, E, nu, "rect", h, b]]
# Zuweisung auf Elemente
Cantilever.PropID = ["Prop1"]*nEl

Cantilever.nSeg = 20
# Darstellung Vernetzung
Cantilever.PlotMesh()

# Statische Analyse
Cantilever.StaticAnalysis()
Cantilever.PlotDisplacement(component='all', scale=10)
Cantilever.PlotStress(stress='all', scale=10)
Cantilever.PlotInternalForces(scale=10)

# Modalanalyse
Cantilever.EigenvalueAnalysis(nEig=3)
Cantilever.PlotMode(scale=5)
print('Eigenvalue solver:', Cantilever.EigenvalSolver)

# Analytical values, continuous beam theory for eigenfrequencies
print("Analytical results")

print("maximum stress due to F [MPa]:")
sigmaMax = np.abs(F*l/I*h/2)
print(sigmaMax)

print("maximum displacement due to F [mm]:")
wMax = F*l**3/(3*E*I)
print(wMax)

print("maximum stress due to deadLoad [MPa]:")
sigmaMax = np.abs((rho*9810*A)*l**2*h/(4*I))
print(sigmaMax)

print("maximum displacement due to deadLoad [mm]:")
wMax = (rho*9810*A)*l**4/(8*E*I)
print(wMax)

print("first three bending modes [Hz]:")
fB1 = 1.87510107**2/(2*np.pi*l**2)*((E*I)/(rho*A))**0.5
fB2 = 4.69409113**2/(2*np.pi*l**2)*((E*I)/(rho*A))**0.5
fB3 = 7.85475744**2/(2*np.pi*l**2)*((E*I)/(rho*A))**0.5
print(fB1, ',', fB2, ',', fB3)

print("first three longitudinal modes [Hz]:")
fL1 = 1/(4*l)*(E/rho)**0.5
fL2 = 3/(4*l)*(E/rho)**0.5
fL3 = 5/(4*l)*(E/rho)**0.5
print(fL1, ',', fL2, ',', fL3)
