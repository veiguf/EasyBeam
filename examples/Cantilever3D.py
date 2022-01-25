from EasyBeam import Beam3D
import numpy as np
import matplotlib.pyplot as plt







class Model(Beam3D):
    # Parameter
    b_x = 10          # mm
    h_x = 20          # mm
    F_x = 100        # N
    l_x = 1000        # mm
    E_x = 210000      # MPa
    rho_x = 7.85e-9   # t/mm^3
    nu_x = 0.3

    DesVar = ["h_x", "b_x"]
    def __init__(self):

        self.nEl = 5
        self.nSeg = 20

        self.plotting = False
        self.stiffMatType = "Euler-Bernoulli"  # Euler-Bernoulli or Timoshenko-Ehrenfest
        self.massMatType = "consistent"        # lumped or consistent

        # Knoten [mm]
        self.Nodes = [[]]*(self.nEl+1)
        for i in range(self.nEl+1):
            self.Nodes[i] = [self.l_x*i/self.nEl, 0.0, 0.0]

        # Elemente: verbindet die Knoten
        self.El = [[]]*(self.nEl)
        for i in range(self.nEl):
            self.El[i] = [i+1, i+2]

        # Randbedingungen und Belastung [N] bzw. [Nmm]
        self.Disp = [[         1, [0, 0,         0, 0, 0, 0]]]
        self.Load = [[self.nEl+1, [0, 0, -self.F_x, 0, 0, 0]]]

        # Werkstoff und Querschnitt: ID, rho, E, A, I, eU, eL
        self.Properties = [["Prop1", self.rho_x, self.E_x, self.nu_x, "rect", self.h_x, self.b_x]]
        # Zuweisung auf Elemente
        self.PropID = ["Prop1"]*self.nEl


# Initialisiern des Problems
Cantilever = Model()

# Darstellung Vernetzung
# Cantilever.PlotMesh()

Cantilever.Initialize()
# Statische Analyse
Cantilever.StaticAnalysis()
Cantilever.ComputeDisplacement()
Cantilever.ComputeInternalForces()
Cantilever.ComputeStress()
Cantilever.SensitivityAnalysis()
Cantilever.ComputeStressSensitivity()
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
plt.plot(Cantilever.sigmaEqvMax[:, :].flatten("C"))

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
