from EasyBeam import Beam3D
import numpy as np
import matplotlib.pyplot as plt

# Parameter
b = 10          # mm
h = 20          # mm
F = 100        # N
l = 1000        # mm
E = 210000      # MPa
rho = 7.85e-9   # t/mm^3
I = b*h**3/12   # mm^4
A = b*h         # mm^2
nEl = 5
nu = 0.3

# Initialisiern des Problems
Cantilever = Beam3D()
Cantilever.plotting = False

# Setze Elementenbeschreibung
Cantilever.stiffMatType = "Euler-Bernoulli"  # Euler-Bernoulli or Timoshenko-Ehrenfest
Cantilever.massMatType = "consistent"        # lumped or consistent

# Knoten [mm]
Cantilever.Nodes = [[]]*(nEl+1)
for i in range(nEl+1):
    Cantilever.Nodes[i] = [l*i/nEl, 0.0, 0.0]

# Elemente: verbindet die Knoten
Cantilever.El = [[]]*(nEl)
for i in range(nEl):
    Cantilever.El[i] = [i+1, i+2]

# Randbedingungen und Belastung [N] bzw. [Nmm]
Cantilever.Disp = [[    1, [0, 0,  0, 0, 0, 0]]]
Cantilever.Load = [[nEl+1, [0, 0, -F, 0, 0, 0]]]

# Werkstoff und Querschnitt: ID, rho, E, A, I, eU, eL
Cantilever.Properties = [["Prop1", rho, E, nu, 1, h, b]]
# Zuweisung auf Elemente
Cantilever.PropID = ["Prop1"]*nEl

Cantilever.nSeg = 20
# Darstellung Vernetzung
# Cantilever.PlotMesh()

Cantilever.Initialize()

# Statische Analyse
Cantilever.StaticAnalysis()
Cantilever.ComputeDisplacement()
Cantilever.ComputeInternalForces()
Cantilever.ComputeStress()
# Cantilever.PlotDisplacement(component='mag', scale=10)
# Cantilever.PlotStress(stress='max', scale=10)

# Modalanalyse
Cantilever.EigenvalueAnalysis(nEig=20)

labels = ["N", "Qy", "Qz", "Mt", "My", "Mz"]
for i in range(6):
    plt.figure()
    plt.title(labels[i])
    plt.plot(Cantilever.QS[:, :, i].flatten("C"))

labels = ["sigma_N", "sigma_by", "sigma_bz", "tau_t"]
for i in range(4):
    for j in range(9):
        plt.figure()
        plt.title(labels[i]+"SecPoint"+str(j))
        plt.plot(Cantilever.sigma[:, :, j,i].flatten("C"))

for i in range(9):
    plt.figure()
    plt.title("EquivalentStress SecPoint"+str(i))
    plt.plot(Cantilever.sigmaEqv[:, :, i].flatten("C"))

plt.figure()
plt.plot(Cantilever.sigmaMax[:, :].flatten("C"))

#Cantilever.PlotMode(scale=5)
"""
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
"""
