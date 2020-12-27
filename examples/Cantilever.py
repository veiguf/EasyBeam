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

# Initialisiern des Problems
Cantilever = Beam2D()

# Setze Elementenbeschreibung
Cantilever.stiffMatType = "Euler-Bernoulli"  # Euler-Bernoulli or Timoshenko-Ehrenfest
Cantilever.massMatType = "consistent"        # lumped or consistent

# Werkstoff und Querschnitt: ID, rho, E, A, I, eU, eL
Cantilever.Properties = [['Prop1', rho, E, A, I, h/2, -h/2]]

# Knoten [mm]
Cantilever.N = [[]]*(nEl+1)
for i in range(nEl+1):
    Cantilever.N[i] = [l*i/nEl, 0.0]

# Elemente: verbindet die Knoten
Cantilever.El = [[]]*(nEl)
for i in range(nEl):
    Cantilever.El[i] = [i, i+1, 'Prop1']

# Randbedingungen und Belastung [N] bzw. [Nmm]
Cantilever.Disp = [[  0, [0, 0, 0]]]
Cantilever.Load = [[nEl, [0, F, 0]]]

# Initialisieren des Modells
Cantilever.Initialize()

# Lösen
Cantilever.StaticAnalysis()
Cantilever.Scale = 10
Cantilever.ComputeStress()
Cantilever.EigenvalueAnalysis(nEig=3)

# Grafische Darstellung
Cantilever.PlotMesh()
Cantilever.PlotStress()
Cantilever.PlotDisplacement()
Cantilever.ScalePhi = 10
Cantilever.PlotMode()
print('Eigenvalue solver:', Cantilever.EigenvalSolver)

# Analytical values, continuous beam theory for eigenfrequencies
print("Analytical results")
print("maximum stress [MPa]:")
sigmaMax = np.abs(F*l/I*h/2)
print(sigmaMax)

print("maximum displacement [mm]:")
wMax = F*l**3/(3*E*I)
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
