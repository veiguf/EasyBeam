from EasyBeam import Beam3D
import numpy as np
import matplotlib.pyplot as plt


class Model(Beam3D):
    # Parameter
    b_x = 10          # mm
    h_x = 20          # mm
    F_x = 100        # N
    l_x = 1000        # mm
    E_x = 206900      # MPa
    rho_x = 7.85e-9   # t/mm^3
    nu_x = 0.29

    DesVar = ["h_x", "b_x"]
    def __init__(self):

        self.nEl = 5
        self.nSeg = 20

        self.plotting = True
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
Cantilever.Initialize()

# Statische Analyse
Cantilever.StaticAnalysis()
Cantilever.ComputeDisplacement()
print("Maximum displacement magnitude:", Cantilever.uSmag.max())
Cantilever.ComputeInternalForces()
Cantilever.ComputeStress()
print("Maximum equivalent stress:", Cantilever.sigmaEqvMax.max())
Cantilever.SensitivityAnalysis()
Cantilever.ComputeStressSensitivity()

# Modalanalyse
Cantilever.EigenvalueAnalysis(nEig=15)
print('Eigenfrequencies:', Cantilever.f0)
print('Eigenvalue solver:', Cantilever.EigenvalSolver)

# Plots
Cantilever.PlotMesh()
Cantilever.PlotDisplacement(component='mag', scale=10)
Cantilever.PlotStress(stress='max', scale=10)
Cantilever.PlotInternalForces(scale=10)
Cantilever.PlotMode(scale=5)

# Analytical values, continuous beam theory for eigenfrequencies
print()
print("Analytical results")
sigmaMax = np.abs(Cantilever.F_x*Cantilever.l_x/Cantilever.Iy[0]*Cantilever.Sec[0, 3, 1])
wMax = Cantilever.F_x*Cantilever.l_x**3/(3*Cantilever.E[0]*Cantilever.Iy[0])
print("maximum stress [MPa]:", sigmaMax)
print("maximum displacement [mm]:", wMax)

# print("first three bending modes in z [Hz]:")
fBz = np.zeros([3])
fBz[0] = 1.87510107**2/(2*np.pi*Cantilever.l_x**2)*((Cantilever.E_x*Cantilever.Iz[0])/(Cantilever.rho_x*Cantilever.A[0]))**0.5
fBz[1] = 4.69409113**2/(2*np.pi*Cantilever.l_x**2)*((Cantilever.E_x*Cantilever.Iz[0])/(Cantilever.rho_x*Cantilever.A[0]))**0.5
fBz[2] = 7.85475744**2/(2*np.pi*Cantilever.l_x**2)*((Cantilever.E_x*Cantilever.Iz[0])/(Cantilever.rho_x*Cantilever.A[0]))**0.5
print("bending Eigenfrequencies in z", fBz)

# print("first three bending modes in y [Hz]:")
fBy = np.zeros([3])
fBy[0] = 1.87510107**2/(2*np.pi*Cantilever.l_x**2)*((Cantilever.E_x*Cantilever.Iy[0])/(Cantilever.rho_x*Cantilever.A[0]))**0.5
fBy[1] = 4.69409113**2/(2*np.pi*Cantilever.l_x**2)*((Cantilever.E_x*Cantilever.Iy[0])/(Cantilever.rho_x*Cantilever.A[0]))**0.5
fBy[2] = 7.85475744**2/(2*np.pi*Cantilever.l_x**2)*((Cantilever.E_x*Cantilever.Iy[0])/(Cantilever.rho_x*Cantilever.A[0]))**0.5
print("bending Eigenfrequencies in y", fBy)

# print("first three longitudinal modes [Hz]:")
fL = np.zeros([3])
fL[0] = 1/(4*Cantilever.l_x)*(Cantilever.E_x/Cantilever.rho_x)**0.5
fL[1] = 3/(4*Cantilever.l_x)*(Cantilever.E_x/Cantilever.rho_x)**0.5
fL[2] = 5/(4*Cantilever.l_x)*(Cantilever.E_x/Cantilever.rho_x)**0.5
print("longitudinal Eigenfrequencies", fL)

# Compute lumped mass matrix from consistent mass matrix
mass = Cantilever.mass
m = Cantilever.m
M = np.zeros_like(m)
for i in range(3):
    s = 0
    for j in range(int(len(m)/6)):
        s += m[6*j+i, 6*j+i]
    idx = (np.arange(0,len(m)/3, 1, dtype=int)*3+i).tolist()
    M[idx, idx] = m[idx, idx]/s*mass
