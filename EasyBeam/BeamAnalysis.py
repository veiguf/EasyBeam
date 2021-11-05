import numpy as np
import scipy.linalg as spla
import numpy.linalg as npla
from scipy.constants import pi
from copy import deepcopy

import time

class Beam:
    from EasyBeam.BeamPlotting import (_plotting, PlotMesh, PlotDisplacement,
                              PlotStress, PlotMode)
    nSeg = 10
    massMatType = "consistent"
    stiffMatType = "Euler-Bernoulli"
    lineStyleUndeformed = "-"
    colormap = "coolwarm" # "RdBu" #"coolwarm_r" #"Blues"
    DesVar = []
    plotting = True
    Load = []
    Initialized = False
    StaticAnalyzed = False
    ComputedDisplacement = False
    ComputedStress = False

    def Initialize(self):
        self.Initialized = True
        self.Nodes = np.array(self.Nodes, dtype=float)
        self.El = np.array(self.El, dtype=int)
        self.nEl = len(self.El)     # number of elements
        self.nN = len(self.Nodes[:, 0])       # number of nodes
        self.BC = []        # boundary conditions
        self.BC_DL = []
        self.DL = []        # displacement load
        for i in range(len(self.Disp)):
            for ii in range(self.nNDoF):
                entry = self.Disp[i][1][ii]
                if isinstance(entry, int) or isinstance(entry, float):
                    self.BC_DL.append(self.nNDoF*(self.Disp[i][0]-1)+ii)
                    if entry == 0:
                        self.BC.append(self.nNDoF*(self.Disp[i][0]-1)+ii)
                    else:
                        self.DL.append(self.nNDoF*(self.Disp[i][0]-1)+ii)

        self.DoF = []       # degrees-of-freedom
        self.DoF_DL = []
        for i in range(self.nNDoF*self.nN):
            if i not in self.BC:
                self.DoF.append(i)
            if i not in self.BC_DL:
                self.DoF_DL.append(i)

        # initial displacements
        self.u = np.empty([self.nNDoF*self.nN])
        self.u[:] = np.nan
        for i in range(len(self.Disp)):
            for ii in range(self.nNDoF):
                if (isinstance(self.Disp[i][1][ii], int) or
                    isinstance(self.Disp[i][1][ii], float)):
                    self.u[self.nNDoF*(self.Disp[i][0]-1)+ii] = self.Disp[i][1][ii]

        # initial forces
        self.F = np.empty([self.nNDoF*self.nN])
        self.F[:] = np.nan
        self.F[self.DoF] = 0
        for i in range(len(self.Load)):
            for ii in range(self.nNDoF):
                if (isinstance(self.Load[i][1][ii], int) or
                    isinstance(self.Load[i][1][ii], float)):
                    self.F[self.nNDoF*(self.Load[i][0]-1)+ii] = self.Load[i][1][ii]

        self.rho = np.zeros([self.nEl])
        self.E = np.zeros([self.nEl])
        self.nu = np.zeros([self.nEl])
        self.G = np.zeros([self.nEl])
        self.A = np.zeros([self.nEl])
        self.Ix = np.zeros([self.nEl])
        self.Iy = np.zeros([self.nEl])
        self.Iz = np.zeros([self.nEl])
        self.zU = np.zeros([self.nEl])
        self.zL = np.zeros([self.nEl])
        self.ϰ = np.zeros([self.nEl])
        # lengths and rotations
        self.ell = np.zeros([self.nEl])
        self.TX = np.zeros([self.nEl, self.nNPoC, self.nNPoC])
        self.T = np.zeros([self.nEl, 2*self.nNDoF, 2*self.nNDoF])
        # self.r = np.zeros([self.nEl, 3, self.nSeg+1])
        self.mass = 0
        self.L = np.zeros([self.nEl, 2*self.nNDoF, self.nNDoF*self.nN])
        self.BL = np.zeros([self.nEl, self.nSeg+1, 2, 2*self.nNDoF])
        self.BU = np.zeros([self.nEl, self.nSeg+1, 2, 2*self.nNDoF])
        if self.nNDoF == 3:
            self.r0 = np.insert(self.Nodes, 2, 0, axis=1).flatten('C')
        elif self.nNDoF == 6:
            self.r0 = np.block([self.Nodes, np.zeros_like(self.Nodes)]).flatten('C')

        for i in range(self.nEl):
            for ii in range(len(self.Properties)):
                if self.PropID[i] == self.Properties[ii][0]:
                    self.rho[i] = self.Properties[ii][1]
                    self.E[i] = self.Properties[ii][2]
                    self.nu[i] = self.Properties[ii][3]
                    self.G[i] = self.E[i]/(2*(1+self.nu[i]))
                    # Böge & Böge (2019) Formeln und Tabellen zur Technischen Mechanik
                    if self.Properties[ii][4] in [1, "rect", "Rectangle"]:
                        h = self.Properties[ii][5]
                        b = self.Properties[ii][6]
                        self.A[i] = b*h
                        c = 1/3*(1-0.63/(h/b)+0.052/(h/b)**5)
                        if h >= b:
                            self.Ix[i] = c*h*b**3
                        else:
                            self.Ix[i] = c*b*h**3
                        self.Iy[i] = b*h**3/12
                        self.Iz[i] = h*b**3/12
                        self.zU[i] = h/2
                        self.zL[i] = -h/2
                        self.ϰ[i] = 10*(1+self.nu[i])/(12+11*self.nu[i])  #Solid rectangular cross-sectional geometry after Cowper (1966)
                    elif self.Properties[ii][4] in [2, "round"]:
                        r = self.Properties[ii][5]
                        self.A[i] = pi*r**2
                        self.Ix[i] = pi*r**4/2
                        self.Iy[i] = pi*r**4/4
                        self.Iz[i] = pi*r**4/4
                        self.zU[i] = r/2
                        self.zL[i] = -r/2
                        self.ϰ[i] = 0.847
                    elif self.Properties[ii][4] in [3, "roundtube"]:
                        r = self.Properties[ii][5]
                        t = self.Properties[ii][6]
                        self.A[i] = pi*((r+t)**2-(r)**2)
                        self.Ix[i] = pi*((r+t)**4-r**4)/2
                        self.Iy[i] = pi*((r+t)**4-r**4)/4
                        self.Iz[i] = pi*((r+t)**4-r**4)/4
                        self.zU[i] = r/2
                        self.zL[i] = -r/2
                        self.ϰ[i] = 0.847 # needs to be corrected!!!!
                    #elif self.Properties[ii][4] in [5, 'I', 'DoubleT']
                    elif self.Properties[ii][4] in [6, "C"]:
                        h = self.Properties[ii][5]
                        b = self.Properties[ii][6]
                        t = self.Properties[ii][7]
                        self.A[i] = b*h-(b-t)*(h-2*t)
                        lt1 = 2*b-t
                        lt2 = h-1.6*t
                        self.Ix[i] = 1/3*(lt1*t**3+lt2*t**3)
                        self.Iy[i] = b*h**3/12-(b-t)*(h-2*t)**3/12
                        e1 = 1/2*(2*t*b**2+(h-2*t)*t**2)/(2*t*b+(h-2*t)*t)
                        e2 = b-2*e1
                        self.Iz[i] = 1/3*(h*e1**3-(h-2*t)*(e1-t)**3+2*t*e2**3)
                        self.zU[i] = h/2
                        self.zL[i] = -h/2
                    else:
                        print("oops nothing more programmed!!!")
            self.ell[i] = np.linalg.norm(self.Nodes[self.El[i, 1]-1, :] -
                                       self.Nodes[self.El[i, 0]-1, :])
            self.mass += (self.A[i]*self.ell[i]*self.rho[i])
            self.TX[i] = self.TransXMat(i)
            self.T[i] = self.TransMat(i)
            self.L[i,          0:  self.nNDoF, self.nNDoF*(self.El[i, 0]-1):self.nNDoF*(self.El[i, 0]-1)+self.nNDoF] = np.eye(self.nNDoF)
            self.L[i, self.nNDoF:2*self.nNDoF, self.nNDoF*(self.El[i, 1]-1):self.nNDoF*(self.El[i, 1]-1)+self.nNDoF] = np.eye(self.nNDoF)
            for j in range(self.nSeg+1):
                ξ = j/(self.nSeg)
                self.BL[i, j], self.BU[i, j] = self.StrainDispMat(ξ, self.ell[i], self.zU[i], self.zL[i])

        if self.plotting:
            self.r0S = np.zeros([self.nEl, 2, self.nSeg+1])
            for i in range(self.nEl):
                for j in range(self.nSeg+1):
                    ξ = j/(self.nSeg)
                    self.r0S[i, :, j] = self.TX[i]@self.ShapeMat(ξ, self.ell[i])@self.T[i]@self.L[i]@self.r0

    def Assemble(self, MatElem):
        Matrix = np.zeros([self.nNDoF*self.nN, self.nNDoF*self.nN])
        for i in range(self.nEl):
            Matrix += self.L[i].T@self.T[i].T@MatElem(i)@self.T[i]@self.L[i]
        return Matrix

    def NMat(self, i, ξ):
        NMat = self.TX[i]@self.ShapeMat(ξ, self.ell[i])@self.T[i]@self.L[i]
        return NMat

    def AssembleOneDirection(self, MatElem):  # needed for FFRF
        Matrix = np.zeros([self.nNPoC, self.nNDoF*self.nN])
        for i in range(self.nEl):
            Matrix += self.TX[i]@MatElem(i)@self.T[i]@self.L[i]
        return Matrix

    def StaticAnalysis(self):
        self.StaticAnalyzed = True
        if not self.Initialized:
            self.Initialize()
        self.k = self.Assemble(self.StiffMatElem)
        self.u[self.DoF_DL] = np.linalg.solve(self.k[self.DoF_DL, :][:, self.DoF_DL],
                                              self.F[self.DoF_DL]-
                                              self.k[self.DoF_DL, :][:, self.DL]@self.u[self.DL])
        self.F[self.BC_DL] = self.k[self.BC_DL, :][:, self.DoF]@self.u[self.DoF]
        self.r = self.r0+self.u

    def SensitivityAnalysis(self, xDelta=1e-9):
        nx = np.size(self.DesVar)
        self.uNabla = np.zeros((len(self.u), np.size(self.DesVar)))
        self.massNabla = np.zeros((np.size(self.DesVar,)))
        FPseudo = np.zeros((len(self.F), nx))
        self.TNabla = np.zeros([self.nEl, 2*self.nNDoF, 2*self.nNDoF, nx])
        self.BLNabla = np.zeros([self.nEl, self.nSeg+1, 2, 2*self.nNDoF, nx])
        self.BUNabla = np.zeros([self.nEl, self.nSeg+1, 2, 2*self.nNDoF, nx])
        self.ENabla = np.zeros((self.nEl, nx))
        for i in range(nx):
            new = deepcopy(self)
            xPert = xDelta*(1+getattr(new, new.DesVar[i]))
            setattr(new, new.DesVar[i],
                    getattr(new, new.DesVar[i])+xPert)
            new.__init__()
            new.Initialize()
            self.TNabla[:, :, :, i] = (new.T-self.T)/xPert
            self.BLNabla[:, :, :, :, i] = (new.BL-self.BL)/xPert
            self.BUNabla[:, :, :, :, i] = (new.BU-self.BU)/xPert
            self.ENabla[:, i] = (new.E-self.E)/xPert
            kNew = new.Assemble(new.StiffMatElem)
            FPseudo[:, i] = (new.F-self.F)/xPert-((kNew-self.k)/xPert)@self.u
            self.massNabla[i] = (new.mass-self.mass)/xPert
        self.uNabla[self.DoF_DL] = np.linalg.solve(self.k[self.DoF_DL][:, self.DoF_DL],
                                                   FPseudo[self.DoF_DL])

    def ComputeDisplacement(self):
        self.ComputedDisplacement = True
        if not self.StaticAnalyzed:
            self.StaticAnalysis()
        self.uE = np.zeros([self.nEl, 2*self.nNDoF])
        self.uS = np.zeros([self.nEl, 2, self.nSeg+1])
        for iEl in range(self.nEl):
            self.uE[iEl]  = self.T[iEl]@self.L[iEl]@self.u
            for j in range(self.nSeg+1):
                ξ = j/(self.nSeg)
                self.uS[iEl, :, j] = self.TX[iEl]@self.ShapeMat(ξ, self.ell[iEl])@self.uE[iEl]

    def ComputeStress(self):
        self.ComputedStress = True
        if not self.ComputedDisplacement:
            self.ComputeDisplacement()
        self.epsilonL = np.zeros([self.nEl, self.nSeg+1, 2])
        self.epsilonU = np.zeros([self.nEl, self.nSeg+1, 2])
        self.sigmaL = np.zeros([self.nEl, self.nSeg+1, 2])
        self.sigmaU = np.zeros([self.nEl, self.nSeg+1, 2])
        self.sigmaMax = np.zeros([self.nEl, self.nSeg+1])
        for iEl in range(self.nEl):
            for j in range(self.nSeg+1):
                self.epsilonL[iEl, j] = self.BL[iEl, j]@self.uE[iEl]
                self.epsilonU[iEl, j] = self.BU[iEl, j]@self.uE[iEl]
                self.sigmaL[iEl, j] = self.epsilonL[iEl, j]*self.E[iEl]
                self.sigmaU[iEl, j] = self.epsilonU[iEl, j]*self.E[iEl]
                self.sigmaMax[iEl, j] = np.max((np.abs(np.sum(self.sigmaL[iEl, j])),
                                                np.abs(np.sum(self.sigmaU[iEl, j]))))

    def ComputeStressSensitivity(self):
        # not general enough for shape
        nx = np.size(self.DesVar)
        self.uENabla = np.zeros([self.nEl, 2*self.nNDoF, nx])
        self.epsilonLNabla = np.zeros((self.nEl, self.nSeg+1, 2, nx))
        self.epsilonUNabla = np.zeros((self.nEl, self.nSeg+1, 2, nx))
        self.sigmaLNabla = np.zeros((self.nEl, self.nSeg+1, 2, nx))
        self.sigmaUNabla = np.zeros((self.nEl, self.nSeg+1, 2, nx))
        for i in range(nx):
            for iEl in range(self.nEl):
                self.uENabla[iEl, :, i]  = self.T[iEl]@self.L[iEl]@self.uNabla[:, i]+self.TNabla[iEl, :, :, i]@self.L[iEl]@self.u
                for j in range(self.nSeg+1):
                    self.epsilonLNabla[iEl, j, :, i] = self.BLNabla[iEl, j, :, :, i]@self.uE[iEl] + self.BL[iEl, j]@self.uENabla[iEl, :, i]
                    self.epsilonUNabla[iEl, j, :, i] = self.BUNabla[iEl, j, :, :, i]@self.uE[iEl] + self.BU[iEl, j]@self.uENabla[iEl, :, i]
                    self.sigmaLNabla[iEl, j, :, i] = self.epsilonLNabla[iEl, j, :, i]*self.E[iEl] + self.epsilonL[iEl, j]*self.ENabla[iEl, i]
                    self.sigmaUNabla[iEl, j, :, i] = self.epsilonUNabla[iEl, j, :, i]*self.E[iEl] + self.epsilonU[iEl, j]*self.ENabla[iEl, i]

    def EigenvalueAnalysis(self, nEig=2, massMatType="consistent"):
        if not self.Initialized:
            self.Initialize()
        self.massMatType = massMatType
        self.k = self.Assemble(self.StiffMatElem)
        self.m = self.Assemble(self.MassMatElem)
        try:
            lambdaComplex, self.Phi = spla.eigh(self.k[self.DoF, :][:, self.DoF],
                                                self.m[self.DoF, :][:, self.DoF],
                                                eigvals=(0, nEig-1))
            self.EigenvalSolver = "scipy.linalg.eigh"
        except:
            lambdaComplex, self.Phi = spla.eig(self.k[self.DoF, :][:, self.DoF],
                                               self.m[self.DoF, :][:, self.DoF])
            self.EigenvalSolver = "scipy.linalg.eig"
        self.lambdaR = abs(lambdaComplex.real)
        iSort = self.lambdaR.real.argsort()
        self.lambdaR = self.lambdaR[iSort]
        self.omega = np.sqrt(self.lambdaR)
        self.f0 = self.omega/2/np.pi
        self.Phi = self.Phi[:, iSort]

    def EigenvalueSensitivity(self):
        pass


class Beam2D(Beam):
    from EasyBeam.Beam2D import (ShapeMat, TransXMat, TransMat,
                                 StrainDispMat, StrainDispNablah,
                                 StiffMatElem, MassMatElem)
    nNDoF = 3   # number of nodal degrees of freedom
    nNPoC = 2   # number of nodal position coordinates

class Beam3D(Beam):
    """
    I need Ix, Iy, Iz...
    """
    from EasyBeam.Beam3D import (ShapeMat, TransXMat, TransMat,
                                 StrainDispMat, StrainDispNablah,
                                 StiffMatElem, MassMatElem)
    nNDoF = 6   # number of nodal degrees of freedom
    nNPoC = 3   # number of nodal position coordinates

class BeamFFRF2D(Beam2D):
    from EasyBeam.Beam2D import (ShapeMat, StrainDispMat, StrainDispNablah,
                                 StiffMatElem, MassMatElem)
    from EasyBeam.BeamFFRF2D import (StfElem, SrfElem, FFRF_Output,
                                     FFRF_OutputSensitivities)
    nNDoF = 3   # number of nodal degrees of freedom
    nNPoC = 2   # number of nodal position coordinates

if __name__ == '__main__':

    t0 = time.time()

    class Model(Beam2D):
        b1 = 10    # mm
        h1 = 20    # mm
        b2 = 10    # mm
        h2 = 20    # mm
        x1 = 0
        x2 = 100
        x3 = 100
        y1 = 0
        y2 = 0
        y3 = 100
        DesVar = ["h1", "b1", "h2", "b2", "x1", "x2", "x3", "y1", "y2", "y3"]

        def __init__(self):
            self.stiffMatType = "Euler-Bernoulli"  # Euler-Bernoulli or Timoshenko-Ehrenfest
            self.massMatType = "consistent"        # lumped or consistent

            #Material     rho       E   nu shape, h, b
            self.Properties = [['Steel', 7.85e-9, 210000, 0.3, 1, self.h1, self.b1],
                               [  'Alu', 2.70e-9,  70000, 0.3, 1, self.h2, self.b2]]
            self.Nodes = [[self.x1, self.y1],
                          [self.x2, self.y2],
                          [self.x3, self.y3]]
            self.El = [[1, 2],
                       [2, 3]]
            self.PropID = ["Alu", "Steel"]
            self.Disp = [[1, [  0,   0, 'f']],
                         [2, [0.1,   0, 'f']]]
            self.Load = [[3, [800, 'f', 'f']]]
            self.nSeg = 2

    Test = Model()
    # Test.PlotMesh(NodeNumber=True, ElementNumber=True, Loads=True, BC=True, FontMag=2)

    Test.StaticAnalysis()
    Test.ComputeStress()
    # Test.PlotDisplacement('mag', scale=20)
    # Test.PlotStress('max', scale=20)

    Test.SensitivityAnalysis()
    Test.ComputeStressSensitivity()

    # Test.EigenvalueAnalysis(nEig=len(Test.DoF))
    # Test.PlotMode(scale=0.1)

    t1 = time.time()

    xDelta = 1e-6
    x0 = np.array([20, 10, 20, 10, 0, 100, 100, 0, 0, 100])

    def Eval(x):
        Test = Model()
        Test.h1 = x[0]
        Test.b1 = x[1]
        Test.h2 = x[2]
        Test.b2 = x[3]
        Test.x1 = x[4]
        Test.x2 = x[5]
        Test.x3 = x[6]
        Test.y1 = x[7]
        Test.y2 = x[8]
        Test.y3 = x[9]
        Test.__init__()
        Test.StaticAnalysis()
        Test.ComputeStress()
        return(Test.u, Test.sigmaL)

    u0, sigma0 = Eval(x0)
    uNabla = np.zeros([len(u0), len(x0)])
    sigmaNabla = np.zeros([sigma0.shape[0], sigma0.shape[1], sigma0.shape[2], len(x0)])
    for i in range(len(x0)):
        e = np.zeros_like(x0)
        e[i] = 1
        u1, sigma1 = Eval(x0+e*xDelta)
        uNabla[:, i] = (u1-u0)/xDelta
        sigmaNabla[:, :, :, i] = (sigma1-sigma0)/xDelta

    t2 = time.time()

    # np.set_printoptions(precision=6, suppress=True)
    for i in range(len(x0)):
        print("\ndisplacement sensitivity "+str(Test.DesVar[i]))
        print(np.linalg.norm(uNabla[:, i]-Test.uNabla[:, i]))
        print("FD:\n", uNabla[:, i])
        print("Analytical:\n", Test.uNabla[:, i])
    for i in range(len(x0)):
        print("\nstress sensitivity "+str(Test.DesVar[i]))
        print(np.linalg.norm(sigmaNabla[:, :, :, i]-Test.sigmaLNabla[:, :, :, i]))
        print("FD:\n", sigmaNabla[:, :, :, i])
        print("Analytical:\n", Test.sigmaLNabla[:, :, :, i])
    print("\ncomputation time analytical:", t1-t0)
    print("\ncomputation time numerical:", t2-t1)
