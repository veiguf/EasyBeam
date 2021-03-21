import numpy as np
import scipy.linalg as spla
import numpy.linalg as npla
from scipy.constants import pi
from copy import deepcopy


class Beam:
    from EasyBeam.BeamPlotting import (_plotting, PlotMesh, PlotDisplacement,
                              PlotStress, PlotMode)
    nSeg = 10
    massMatType = "consistent"
    stiffMatType = "Euler-Bernoulli"
    lineStyleUndeformed = "-"
    colormap = "RdBu" #"coolwarm_r" #"Blues"
    SizingVariables = []
    plotting = True

    Initialized = False
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
        self.A = np.zeros([self.nEl])
        self.I = np.zeros([self.nEl])
        self.zU = np.zeros([self.nEl])
        self.zL = np.zeros([self.nEl])
        self.ϰ = np.zeros([self.nEl])
        # lengths and rotations
        self.ell = np.zeros([self.nEl])
        self.β = np.zeros([self.nEl])
        self.T2 = np.zeros([self.nEl, 2, 2])
        self.T = np.zeros([self.nEl, 6, 6])
        # self.r = np.zeros([self.nEl, 3, self.nSeg+1])
        self.mass = 0
        self.L = np.zeros([self.nEl, 6, 3*self.nN])
        self.r0 = np.insert(self.Nodes, 2, 0, axis=1).flatten('C')

        for i in range(self.nEl):
            for ii in range(len(self.Properties)):
                if self.PropID[i] == self.Properties[ii][0]:
                    self.rho[i] = self.Properties[ii][1]
                    self.E[i] = self.Properties[ii][2]
                    self.nu[i] = self.Properties[ii][3]
                    if self.Properties[ii][4] in [1, "rect", "Rectangle"]:
                        h = self.Properties[ii][5]
                        b = self.Properties[ii][6]
                        self.A[i] = b*h
                        self.I[i] = b*h**3/12
                        self.zU[i] = h/2
                        self.zL[i] = -h/2
                        self.ϰ[i] = 10*(1+self.nu[i])/(12+11*self.nu[i])  #Solid rectangular cross-sectional geometry after Cowper (1966)
                    elif self.Properties[ii][4] in [2, "round"]:
                        r = self.Properties[ii][5]
                        self.A[i] = pi*r**2
                        self.I[i] = pi*r**4/4
                        self.zU[i] = r/2
                        self.zL[i] = -r/2
                        self.ϰ[i] = 0.847
                    elif self.Properties[ii][4] in [3, "roundtube"]:
                        r = self.Properties[ii][5]
                        t = self.Properties[ii][6]
                        self.A[i] = pi*((r+t)**2-(r)**2)
                        self.I[i] = pi*((r+t)**4-r**4)/4
                        self.zU[i] = r/2
                        self.zL[i] = -r/2
                        self.ϰ[i] = 0.847 # needs to be corrected!!!!
                    #elif self.Properties[ii][4] in [5, 'I', 'DoubleT']
                    elif self.Properties[ii][4] in [6, "C"]:
                        h = self.Properties[ii][5]
                        b = self.Properties[ii][6]
                        t = self.Properties[ii][7]
                        self.A[i] = b*h-(b-t)*(h-2*t)
                        self.I[i] = b*h**3/12-(b-t)*(h-2*t)**3/12
                        self.zU[i] = h/2
                        self.zL[i] = -h/2
                    else:
                        print("oops nothing more programmed!!!")
            self.ell[i] = np.linalg.norm(self.Nodes[self.El[i, 1]-1, :] -
                                       self.Nodes[self.El[i, 0]-1, :])
            self.mass += (self.A[i]*self.ell[i]*self.rho[i])
            if self.Nodes[self.El[i, 1]-1, 0] >= self.Nodes[self.El[i, 0]-1, 0]:
                """
                HERE is a division by zero...needs to be checked
                """
                self.β[i] = np.arctan((self.Nodes[self.El[i, 1]-1, 1] -
                                       self.Nodes[self.El[i, 0]-1, 1])/
                                      (self.Nodes[self.El[i, 1]-1, 0] -
                                       self.Nodes[self.El[i, 0]-1, 0]))
            else:
                self.β[i] = pi + np.arctan((self.Nodes[self.El[i, 1]-1, 1] -
                                            self.Nodes[self.El[i, 0]-1, 1])/
                                           (self.Nodes[self.El[i, 1]-1, 0] -
                                            self.Nodes[self.El[i, 0]-1, 0]))
            self.T2[i] = np.array([[np.cos(self.β[i]), -np.sin(self.β[i])],
                                  [np.sin(self.β[i]),  np.cos(self.β[i])]],
                                 dtype=float)
            self.T[i] = np.block([[    self.T2[i].T, np.zeros([2, 4])],
                                  [0, 0, 1, 0, 0, 0],
                                  [np.zeros([2, 3]), self.T2[i].T, np.zeros([2, 1])],
                                  [0, 0, 0, 0, 0, 1]])
            self.L[i, 0:3, 3*(self.El[i, 0]-1):3*(self.El[i, 0]-1)+3] = np.eye(3)
            self.L[i, 3:6, 3*(self.El[i, 1]-1):3*(self.El[i, 1]-1)+3] = np.eye(3)
        if self.plotting:
            self.r0S = np.zeros([self.nEl, 2, self.nSeg+1])
            for i in range(self.nEl):
                for j in range(self.nSeg+1):
                    ξ = j/(self.nSeg)
                    self.r0S[i, :, j] = self.T2[i]@self.ShapeMat(ξ, self.ell[i])@self.T[i]@self.L[i]@self.r0

    def Assemble(self, MatElem):
        Matrix = np.zeros([self.nNDoF*self.nN, self.nNDoF*self.nN])
        for i in range(self.nEl):
            Matrix += self.L[i].T@self.T[i].T@MatElem(i)@self.T[i]@self.L[i]
        return Matrix

    def StaticAnalysis(self):
        if not self.Initialized:
            self.Initialize()
        self.k = self.Assemble(self.StiffMatElem)
        self.u[self.DoF_DL] = np.linalg.solve(self.k[self.DoF_DL, :][:, self.DoF_DL],
                                              self.F[self.DoF_DL]-
                                              self.k[self.DoF_DL, :][:, self.DL]@self.u[self.DL])
        self.F[self.BC_DL] = self.k[self.BC_DL, :][:, self.DoF_DL]@self.u[self.DoF_DL]
        self.r = self.r0+self.u

    def SensitivityAnalysis(self, xDelta=1e-3):
        nx = np.size(self.SizingVariables)
        self.uNabla = np.zeros((len(self.u), np.size(self.SizingVariables)))
        self.massNabla = np.zeros((np.size(self.SizingVariables,)))
        FPseudo = np.zeros((len(self.F), nx))
        ix = 0
        for i in range(len(self.SizingVariables)):
            for j in self.SizingVariables[i]:
                new = deepcopy(self)
                if j =="h":
                    xPert = xDelta*(1+new.Properties[i][5])
                    new.Properties[i][5] += xPert
                elif j =="b":
                    xPert = xDelta*(1+new.Properties[i][6])
                    new.Properties[i][6] += xPert
                new.Initialize()
                kNew = new.Assemble(new.StiffMatElem)
                FPseudo[:, ix] = (new.F-self.F)/xPert-((kNew-self.k)/xPert)@self.u
                self.massNabla[ix] = (new.mass-self.mass)/xPert
                ix += 1
        self.uNabla[self.DoF_DL] = np.linalg.solve(self.k[self.DoF_DL][:, self.DoF_DL],
                                                   FPseudo[self.DoF_DL])

    def ComputeDisplacement(self):
        self.ComputedDisplacement = True
        self.uE = np.zeros([self.nEl, 6])
        self.uS = np.zeros([self.nEl, 2, self.nSeg+1])
        for iEl in range(self.nEl):
            self.uE[iEl]  = self.T[iEl]@self.L[iEl]@self.u
            for j in range(self.nSeg+1):
                ξ = j/(self.nSeg)
                self.uS[iEl, :, j] = self.T2[iEl]@self.ShapeMat(ξ, self.ell[iEl])@self.uE[iEl]

    def ComputeStress(self):
        self.ComputedStress = True
        if not self.ComputedDisplacement:
            self.ComputeDisplacement()
        self.sigmaU = np.zeros([self.nEl, self.nSeg+1, 2])
        self.sigmaL = np.zeros([self.nEl, self.nSeg+1, 2])
        self.epsilonU = np.zeros([self.nEl, self.nSeg+1, 2])
        self.epsilonL = np.zeros([self.nEl, self.nSeg+1, 2])
        self.sigmaMax = np.zeros([self.nEl, self.nSeg+1])
        self.epsilon = np.zeros([self.nEl, self.nSeg+1, 2])
        self.sigma = np.zeros([self.nEl, self.nSeg+1, 2])
        for iEl in range(self.nEl):
            for j in range(self.nSeg+1):
                ξ = j/(self.nSeg)
                BL, BU = self.StrainDispMat(ξ, self.ell[iEl], self.zU[iEl],
                                            self.zL[iEl])
                self.epsilonL[iEl, j] = BL@self.uE[iEl]
                self.epsilonU[iEl, j] = BU@self.uE[iEl]
                self.sigmaL[iEl, j] = self.epsilonL[iEl, j]*self.E[iEl]
                self.sigmaU[iEl, j] = self.epsilonU[iEl, j]*self.E[iEl]
                self.sigmaMax[iEl, j] = np.max((np.abs(np.sum(self.sigmaL[iEl, j])),
                                                np.abs(np.sum(self.sigmaU[iEl, j]))))

    def ComputeStressSensitivity(self):
        # not general enough for shape
        nx = np.size(self.SizingVariables)
        self.epsilonLNabla = np.zeros((self.nEl, self.nSeg+1, 2, nx))
        self.epsilonUNabla = np.zeros((self.nEl, self.nSeg+1, 2, nx))
        self.sigmaLNabla = np.zeros((self.nEl, self.nSeg+1, 2, nx))
        self.sigmaUNabla = np.zeros((self.nEl, self.nSeg+1, 2, nx))
        for iEl in range(self.nEl):
            uENabla = self.T[iEl]@self.L[iEl]@self.uNabla
            for j in range(self.nSeg+1):
                ξ = j/(self.nSeg)
                BL, BU = self.StrainDispMat(ξ, self.ell[iEl], self.zU[iEl],
                                            self.zL[iEl])
                BLNabla = np.zeros((2, 6, nx))
                BUNabla = np.zeros((2, 6, nx))
                ix = 0
                for ii in range(len(self.SizingVariables)):
                    for iVar in self.SizingVariables[ii]:
                        if iVar == "h":
                            BLNabla[:, :, ix], BUNabla[:, :, ix] = \
                                self.StrainDispNablah(ξ, self.ell[iEl])
                        ix += 1
                self.epsilonLNabla[iEl, j] = BLNabla.transpose(0, 2, 1)@self.uE[iEl] + BL@uENabla
                self.epsilonUNabla[iEl, j] = BUNabla.transpose(0, 2, 1)@self.uE[iEl] + BU@uENabla
                self.sigmaLNabla[iEl, j] = self.E[iEl]*self.epsilonLNabla[iEl, j]
                self.sigmaUNabla[iEl, j] = self.E[iEl]*self.epsilonUNabla[iEl, j]

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
    from EasyBeam.Beam2D import (ShapeMat, StrainDispMat, StrainDispNablah,
                                 StiffMatElem, MassMatElem)
    nNDoF = 3

class Beam3D(Beam):
    """
    I need Ix, Iy, Iz...
    """
    from EasyBeam.Beam3D import (StiffMatElem, ShapeMat, StrainDispMat,
                                 MassMatElem, StrainDispNablah)
    nNDoF = 6

class BeamFFRF2D(Beam2D):
    from EasyBeam.Beam2D import (ShapeMat, StrainDispMat, StrainDispNablah,
                                 StiffMatElem, MassMatElem)
    from EasyBeam.BeamFFRF2D import (NMat, StfElem, SrfElem, Assemble2x6,
                                     FFRF_Output)
    nNDoF = 3

if __name__ == '__main__':

    Test = Beam2D()

    Test.stiffMatType = "Euler-Bernoulli"  # Euler-Bernoulli or Timoshenko-Ehrenfest
    Test.massMatType = "consistent"        # lumped or consistent

    b = 10      # mm
    h = 20      # mm

    #Material     rho       E   nu shape, h, b
    Test.Properties = [['Steel', 7.85e-9, 210000, 0.3, 1, h, b],
                       [  'Alu', 2.70e-9,  70000, 0.3, 1, h, b]]

    # one list per element property set: possibilities: None, "matertial", E, nu, A, I, h, b
    Test.SizingVariables = [["h", "b"],
                            ["h", "b"]]

    Test.Nodes = [[  0,   0],
                  [100,   0],
                  [100, 100]]
    Test.El = [[1, 2],
               [2, 3]]
    Test.PropID = ["Alu", "Steel"]

    Test.Disp = [[1, [  0,   0, 'f']],
                 [2, [0.1,   0, 'f']]]
    Test.Load = [[3, [800, 'f', 'f']]]

    Test.nSeg = 100
    Test.PlotMesh(NodeNumber=True, ElementNumber=True, Loads=True, BC=True,
                  FontMag=2)

    Test.StaticAnalysis()
    Test.PlotDisplacement('mag', scale=20)
    Test.PlotStress('max', scale=20)

    Test.EigenvalueAnalysis(nEig=len(Test.DoF))
    Test.PlotMode(scale=0.1)

    Test.SensitivityAnalysis()

    CheckSens = 1
    if CheckSens:
        CheckStress = 1
        if CheckStress:
            Test.ComputeStressSensitivity()

        uNablahFD = [np.zeros_like(Test.u)]*2
        sigmaLNablahFD = [np.zeros_like(Test.sigmaL)]*2
        sigmaLNablabFD = [np.zeros_like(Test.sigmaL)]*2
        sigmaUNablahFD = [np.zeros_like(Test.sigmaL)]*2
        sigmaUNablabFD = [np.zeros_like(Test.sigmaL)]*2
        xDelta = 1
        for i in range(2):
            Test1 = deepcopy(Test)
            Test1.Properties[i][5] += xDelta
            Test1.Initialize()
            Test1.StaticAnalysis()
            uNablahFD[i] = (Test1.u-Test.u)/xDelta
            if CheckStress:
                Test1.ComputeStress()
                sigmaLNablahFD[i] = (Test1.sigmaL-Test.sigmaL)/xDelta
                sigmaUNablahFD[i] = (Test1.sigmaU-Test.sigmaU)/xDelta

        uNablabFD = [np.zeros_like(Test.u)]*2
        sigmaLNablabFD = [np.zeros_like(Test.sigmaL)]*2
        for i in range(2):
            Test1 = deepcopy(Test)
            Test1.Properties[i][6] += xDelta
            Test1.Initialize()
            Test1.StaticAnalysis()
            uNablabFD[i] = (Test1.u-Test.u)/xDelta

            if CheckStress:
                Test1.ComputeStress()
                sigmaLNablabFD[i] = (Test1.sigmaL-Test.sigmaL)/xDelta
                sigmaUNablabFD[i] = (Test1.sigmaU-Test.sigmaU)/xDelta

        print("sensitivities:")
        print("displacement")
        print("FD")
        print(uNablahFD)
        print(uNablabFD)
        print("Analytical")
        print(Test.uNabla)
        if CheckStress:
            print("stress")
            print("FD")
            #print(sigmaLNablahFD)
            print("Analytical")
            #print(Test.sigmaLNabla)

