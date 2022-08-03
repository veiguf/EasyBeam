from EasyBeam import Beam3D, Beam2D

import numpy as np
from numpy import pi, sin, cos, sqrt
import matplotlib.pyplot as plt
import seaborn as sns

# Parameter
b = 20          # mm
h = 30          # mm
E = 210000      # MPa
rho = 7.85e-9   # t/mm^3
nu = 0.3

lC = 120
nElC = 3
lR = 180
nElR = 4

def Configuration(theta, plotScale=False):

    x1 = lC*cos(theta)
    y1 = lC*sin(theta)
    x2 = x1+sqrt(lR**2-y1**2)
    
    # Initialisiern des Problems
    Cantilever = Beam2D()
    Cantilever.plotting = True
    
    # Setze Elementenbeschreibung
    Cantilever.stiffMatType = "Euler-Bernoulli"  # Euler-Bernoulli or Timoshenko-Ehrenfest
    Cantilever.massMatType = "consistent"        # lumped or consistent

    # Knoten
    Cantilever.Nodes = []
    for i in range(nElC+1):
        Cantilever.Nodes.append([i/nElC*x1, i/nElC*y1])
    for i in range(nElR+1):
        Cantilever.Nodes.append([x1+i/nElR*(x2-x1), y1-i/nElR*y1])
    Cantilever.Nodes.append([x2, 0.0])

    # Elemente
    Cantilever.El = []
    for i in range(nElC):
        Cantilever.El.append([i+1, i+2])
    for i in range(nElR):
        Cantilever.El.append([nElC+i+2, nElC+i+3])

    # Zwangsbedingungen
    Cantilever.Disp = [[          1, [  0, 0, "f"]],
                       [nElC+nElR+3, ["f", 0,   0]]]

    # Werkstoff und Querschnitt
    Cantilever.Properties = [["Prop1", rho, E, nu, "rect", h, b]]
    # Zuweisung auf Elemente
    Cantilever.PropID = ["Prop1"]*(nElC+nElR)

    Cantilever.nSeg = 100

    Cantilever.Initialize()

    # Modalanalyse mit Drehgelenk
    kx = E*1e9 # rigid
    ky = E*1e9 # rigid
    kT = 0
    mS = 30*30*40*rho
    JS = 0
    Cantilever.EigenvalueAnalysis(nEig=(nElC+nElR)*3+1,
                                  addSpring2D=[[     nElC+1,      nElC+2, kx, ky, kT],
                                               [nElC+nElR+2, nElC+nElR+3, kx, ky, kT]],
                                  addNodalMass2D=[[nElC+nElR+3, mS, mS, JS]])
    if plotScale:
        Cantilever.PlotMesh()
        Cantilever.PlotMode(scale=plotScale)
    return Cantilever

# Eval with plots
SliderCrank = Configuration(45*pi/180, plotScale=0.5)

# Eval for all configurations
theta = np.linspace(0, 2*pi, 37) # 25 -> alle 15 grad
f = []
for itheta in theta:
    SliderCrank = Configuration(itheta, plotScale=False)
    f.append(SliderCrank.f0)
f = np.asarray(f)

nfreq = 10
tDelta = 1e-4

plt.figure()
plt.plot(theta*180/pi, f[:,:nfreq])
plt.ylabel("eigenfrequency $\\omega$ [Hz]")
plt.xlabel("crank orientation $\\theta$ [deg]")
sns.despine()
plt.show()

ratio = tDelta/(1/f[:,:nfreq])
plt.plot(theta*180/pi, ratio)
plt.ylabel("ratio $\\frac{\\Delta t}{T}$")
plt.xlabel("crank orientation $\\theta$ [deg]")
sns.despine()
plt.show()









