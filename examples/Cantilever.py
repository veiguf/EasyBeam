from EasyBeam import Beam2D
import numpy as np

# Parameter
b = 10      # mm
h = 20      # mm
F = -100
l = 1000
E = 210000
rho = 7.85e-9
I = b*h**3/12
A = b*h

# Initialisiern des Problems
Cantilever = Beam2D()

# Setze Elementenbeschreibung
Cantilever.stiffMatType = "Euler-Bernoulli"

# Knoten [mm]
Cantilever.N = [[0, 0],
                [l, 0]]

# Elemente: verbindet die Knoten
Cantilever.El = [[0, 1]]

# Randbedingungen und Belastung [N] bzw. [Nmm]
Cantilever.BC = [0, 1, 2]
Cantilever.Load = [[4, F]]

# Initialisieren des Modells
Cantilever.Initialize()

# Querschnittgeometrie und Werkstoff
Cantilever.eU = np.ones([Cantilever.nEl, 1])*h/2     # mm
Cantilever.eL = np.ones([Cantilever.nEl, 1])*-h/2    # mm
Cantilever.A = np.ones([Cantilever.nEl, 1])*A        # mm^2
Cantilever.I = np.ones([Cantilever.nEl, 1])*I        # mm^4
Cantilever.E = np.ones([Cantilever.nEl, 1])*E        # MPa
Cantilever.rho = np.ones([Cantilever.nEl, 1])*rho    # t/mm^3

# Lösen
Cantilever.StaticAnalysis()
Cantilever.Scale = 10
Cantilever.ComputeStress()
Cantilever.EigenvalueAnalysis(nEig=len(Cantilever.DoF))

# Grafische Darstellung
Cantilever.PlotMesh()
Cantilever.PlotStress()
Cantilever.PlotDisplacement()
Cantilever.ScalePhi = 1
Cantilever.PlotMode()

# With Timoshenko beams
Cantilever.stiffMatType = "Timoshenko-Ehrenfest"
Cantilever.EigenvalueAnalysis(nEig=len(Cantilever.DoF))
Cantilever.PlotMode()

# Analytical values, continuous beam theory for eigenfrequencies
print('\033[1m'+"Analytical results")
print("maximum stress [MPa]:")
sigmaMax = np.abs(F*l/I*h/2)
print(sigmaMax)

print("maximum displacement [mm]:")
wMax = F*l**3/(3*E*I)
print(wMax)

print("first three bending modes [Hz]:")
fB1 = 1.875**2/(2*np.pi*l**2)*((E*I)/(rho*A))**0.5
fB2 = 4.684**2/(2*np.pi*l**2)*((E*I)/(rho*A))**0.5
fB3 = 7.069**2/(2*np.pi*l**2)*((E*I)/(rho*A))**0.5
print(fB1)
print(fB2)
print(fB3)

print("first three longitudinal modes [Hz]:")
fL1 = 1/(4*l)*(E/rho)**0.5
fL2 = 3/(4*l)*(E/rho)**0.5
fL3 = 5/(4*l)*(E/rho)**0.5
print(fL1)
print(fL2)
print(fL3)
