from EasyBeam import Beam3D
from EasyBeam import BeamAnalysis
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
        self.selfload = True
        # Knoten [mm]
        self.Nodes = [[]]*(self.nEl+1)
        for i in range(self.nEl+1):
            self.Nodes[i] = [self.l_x*i/self.nEl, 0.0, 0.0]


        self.El = [[]]*(self.nEl)
        for i in range(self.nEl):
            self.El[i] = [i+1, i+2]


        self.Disp = [[         1, [0, 0, 0, 0, 0, 0]]]
        
        self.Properties = [["Prop1", self.rho_x, self.E_x, self.nu_x, "rect", self.h_x, self.b_x]]
        self.PropID = ["Prop1"]*self.nEl

    def validate(self):
        n = 5
        self.wAnalytical = np.zeros(n,)
        self.MAnalytical = np.zeros(n,)
        self.sigmaAnalytical = np.zeros(n,)
        A = self.b_x * self.h_x
        q0 = A * self.rho_x*9.815*1000
        c1 = -q0*self.l_x
        c2 = q0*self.l_x**2/2
        I = (self.b_x * self.h_x**3)/12
        x = 0
        self.sigmaMaxAnalytical = (c1*x+c2)/I*self.h_x/2
        self.wMaxAnalytical =  1/(self.E_x*I) *q0*self.l_x**4/8
        self.wMaxNablaAnalytical_b = 0
        self.wMaxNablaAnalytical_h = 6*self.rho_x*9806.65*self.l_x**4/(2*self.E_x*self.h_x**3)
        self.sigmaMaxNablaAnalytical_h = -12*self.rho_x*9806.65*self.l_x**2/self.h_x**2

nEl = 20
nh = 25
wAnalytical = np.zeros((nEl, nh))
wCalc = np.zeros((nEl, nh))
wDelta = np.zeros((nEl, nh))
sigmaAnalytical = np.zeros((nEl, nh))
sigmaCalc = np.zeros((nEl, nh))
sigmaDelta = np.zeros((nEl, nh))
wDeltaRelative = np.zeros((nEl, nh))
sigmaDeltaRelative = np.zeros((nEl, nh))

wNabla_h = np.zeros((nEl, nh))
wNabla_b = np.zeros((nEl, nh))
wMaxNablaDelta_h = np.zeros((nEl, nh))
wMaxNablaDelta_b = np.zeros((nEl, nh))
wMaxNablaDeltaRelative_h = np.zeros((nEl, nh))
for iEl in range(0, nEl):
    for ih, h in enumerate(np.linspace(1, 20, nh)):

        Cantilever = Model()
        Cantilever.h_x = h
        Cantilever.nEl = iEl+1
        Cantilever.Initialize()
        
        Cantilever.StaticAnalysis()
        Cantilever.ComputeDisplacement()
        Cantilever.ComputeInternalForces()
        Cantilever.ComputeStress()
        Cantilever.SensitivityAnalysis(xDelta=1e-6)
        Cantilever.ComputeStressSensitivity()
        Cantilever.validate()
        
        wCalc[iEl, ih] = Cantilever.uSmag.max()
        wAnalytical[iEl, ih] = Cantilever.wMaxAnalytical
        wDelta[iEl, ih] = Cantilever.uSmag.max()-Cantilever.wMaxAnalytical
        wDeltaRelative[iEl, ih] = (Cantilever.uSmag.max()-Cantilever.wMaxAnalytical)/Cantilever.wMaxAnalytical
       
        sigmaCalc[iEl, ih] = Cantilever.sigmaEqvMax.max()
        sigmaAnalytical[iEl, ih] = Cantilever.sigmaMaxAnalytical
        sigmaDelta[iEl, ih] = Cantilever.sigmaEqvMax.max()-Cantilever.sigmaMaxAnalytical
        sigmaDeltaRelative[iEl, ih] = (Cantilever.sigmaEqvMax.max()-Cantilever.sigmaMaxAnalytical)/Cantilever.sigmaMaxAnalytical
        
        wNabla_h[iEl, ih] = Cantilever.uNabla[-4,0]
        wNabla_b[iEl, ih] = Cantilever.uNabla[-4,1]

        wMaxNablaDelta_h[iEl, ih] = Cantilever.uNabla[-4,0]-Cantilever.wMaxNablaAnalytical_h
        wMaxNablaDelta_b[iEl, ih] = Cantilever.uNabla[-4,1]-Cantilever.wMaxNablaAnalytical_b
        wMaxNablaDeltaRelative_h[iEl, ih] = (Cantilever.uNabla[-4,0]-Cantilever.wMaxNablaAnalytical_h)/Cantilever.wMaxNablaAnalytical_h
        
extent = (1, 20, 1, nEl+1)

## Primal analysis
# FE: nEl-h: sigma
fig, ax = plt.subplots()
plt.imshow(sigmaCalc, extent=extent, cmap="Blues")
plt.colorbar(label="$\\sigma_{calc}$ [MPa]")
plt.ylabel("number of elements $n_{el}$")
plt.xlabel("height $h$ [mm]")

# An: nEl-h: sigma
fig, ax = plt.subplots()
plt.imshow(sigmaAnalytical, extent=extent, cmap="Blues")
plt.colorbar(label="$\\sigma_{analytical}$ [MPa]")
plt.ylabel("number of elements $n_{el}$")
plt.xlabel("height $h$ [mm]")

# FE: nEl-h: w
fig, ax = plt.subplots()
plt.imshow(wCalc, extent=extent, cmap="Blues")
plt.colorbar(label="$u_{calc}$ [mm]")
plt.ylabel("number of elements $n_{el}$")
plt.xlabel("height $h$ [mm]")

# An: nEl-h: w
fig, ax = plt.subplots()
plt.imshow(wAnalytical, extent=extent, cmap="Blues")
plt.colorbar(label="$u_{analytical}$ [mm]")
plt.ylabel("number of elements $n_{el}$")
plt.xlabel("height $h$ [mm]")


# nEl-h: Delta_sigma
fig, ax = plt.subplots()
plt.imshow(sigmaDelta, extent=extent, cmap="Blues")
plt.colorbar(label="$\\Delta\\sigma$ [MPa]")
plt.ylabel("number of elements $n_{el}$")
plt.xlabel("height $h$ [mm]")

# nEl-h: Delta_u
fig, ax = plt.subplots()
plt.imshow(wDelta, extent=extent, cmap="Blues")
plt.colorbar(label="$\\Delta u$ [mm]")
plt.ylabel("number of elements $n_{el}$")
plt.xlabel("height $h$ [mm]")

# nEl-h: eps_sigma
fig, ax = plt.subplots()
plt.imshow(sigmaDeltaRelative, extent=extent, cmap="Blues")
plt.colorbar(label="$\\epsilon_{\\sigma}$ [-]")
plt.ylabel("number of elements $n_{el}$")
plt.xlabel("height $h$ [mm]")

# nEl-h: eps_u
fig, ax = plt.subplots()
plt.imshow(wDeltaRelative, extent=extent, cmap="Blues")
plt.colorbar(label="$\\epsilon_{u}$ [-]")
plt.ylabel("number of elements $n_{el}$")
plt.xlabel("height $h$ [mm]")


## Sensitivity analysis
# FE: nEl-h: wNabla_h
fig, ax = plt.subplots()
plt.imshow(wNabla_h, extent=extent, cmap="Blues")
plt.colorbar(label="$\\nabla_h u_{calc}$ [mm/mm]")
plt.ylabel("number of elements $n_{el}$")
plt.xlabel("height $h$ [mm]")

# FE: nEl-h: wNabla_b
fig, ax = plt.subplots()
plt.imshow(wNabla_b, extent=extent, cmap="Blues")
plt.colorbar(label="$\\nabla_b u_{calc}$ [mm/mm]")
plt.ylabel("number of elements $n_{el}$")
plt.xlabel("height $h$ [mm]")

# nEl-h: Delta_uNabla_h
fig, ax = plt.subplots()
plt.imshow(wMaxNablaDelta_h, extent=extent, cmap="Blues")
plt.colorbar(label="$\\Delta ( \\nabla_h u ) $ [mm/mm]")
plt.ylabel("number of elements $n_{el}$")
plt.xlabel("height $h$ [mm]")

# nEl-h: Delta_uNabla_b
fig, ax = plt.subplots()
plt.imshow(wMaxNablaDelta_b, extent=extent, cmap="Blues")
plt.colorbar(label="$\\Delta ( \\nabla_b u )$ [mm/mm]")
plt.ylabel("number of elements $n_{el}$")
plt.xlabel("height $h$ [mm]")
print("uNabla_b is an interesting case in which full numerical may be more accurate than semi-analytical!")

# nEl-h: eps_uNabla_h
fig, ax = plt.subplots()
plt.imshow(wMaxNablaDeltaRelative_h, extent=extent, cmap="Blues")
plt.colorbar(label="$\\epsilon_{\\nabla_h u}$ [-]")
plt.ylabel("number of elements $n_{el}$")
plt.xlabel("height $h$ [mm]")