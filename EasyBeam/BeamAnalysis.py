import numpy as np
import scipy.linalg as spla
import numpy.linalg as npla
from scipy.constants import pi
from copy import deepcopy

import time

class Beam:
    from EasyBeam.BeamPlotting import (PlotDisplacement, PlotStress,
                                       PlotInternalForces, PlotMode)
    nSeg = 10
    massMatType = "consistent"
    stiffMatType = "Euler-Bernoulli"
    lineStyleUndeformed = "-"
    colormap = "coolwarm" # "RdBu" #"coolwarm_r" #"Blues"
    plotting = True
    DesVar = []
    Load = []
    Disp = []
    Properties = []
    Initialized = False
    StaticAnalyzed = False
    ComputedDisplacement = False
    ComputedInternalForces = False
    ComputedStress = False
    ModeledPartialDerivatives = False
    SensitivityAnalyzed = False

    def Initialize(self):
        self.__init__()
        self.Initialized = True
        self.Nodes = np.array(self.Nodes, dtype=float)
        self.El = np.array(self.El, dtype=int)
        self.nEl = len(self.El)     # number of elements
        self.nN = len(self.Nodes[:, 0])       # number of nodes
        self.nx = np.size(self.DesVar)
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
        self.ϰ = np.zeros([self.nEl])
        # lengths and rotations
        self.ell = np.zeros([self.nEl])
        self.TX = np.zeros([self.nEl, self.nNPoC, self.nNPoC])
        self.T = np.zeros([self.nEl, 2*self.nNDoF, 2*self.nNDoF])
        # self.r = np.zeros([self.nEl, 3, self.nSeg+1])
        self.mass = 0
        self.idx = [[]]*self.nEl
        self.EMat = np.zeros([self.nEl, self.nSVal, self.nSVal])
        self.Sec = np.zeros([self.nEl, self.nSec, 3])
        self.B = np.zeros([self.nEl, self.nSeg+1, self.nSec, self.nSVal, 2*self.nNDoF])
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
                    if self.Properties[ii][4] in [1, "round"]:
                        r = self.Properties[ii][5]
                        self.A[i] = pi*r**2
                        self.Ix[i] = pi*r**4/2
                        self.Iy[i] = pi*r**4/4
                        self.Iz[i] = pi*r**4/4
                        if self.nNDoF == 3:
                            self.Sec[i, :, :] = np.array([[   0,  0, 0],
                                                          [   0,  r, 0],
                                                          [   0, -r, 0]])
                        if self.nNDoF == 6:
                            self.Sec[i, :, :] = np.array([[              0,               0, 0],
                                                          [              r,               0, r],
                                                          [ np.sqrt(2)/2*r,  np.sqrt(2)/2*r, r],
                                                          [              0,               r, r],
                                                          [-np.sqrt(2)/2*r,  np.sqrt(2)/2*r, r],
                                                          [             -r,               0, r],
                                                          [-np.sqrt(2)/2*r, -np.sqrt(2)/2*r, r],
                                                          [              0,              -r, r],
                                                          [ np.sqrt(2)/2*r, -np.sqrt(2)/2*r, r]])
                        self.ϰ[i] = 0.847
                    elif self.Properties[ii][4] in [2, "roundtube"]:
                        r = self.Properties[ii][5]
                        t = self.Properties[ii][6]
                        self.A[i] = pi*(r**2-(r-t)**2)
                        self.Ix[i] = pi*(r**4-(r-t)**4)/2
                        self.Iy[i] = pi*(r**4-(r-t)**4)/4
                        self.Iz[i] = pi*(r**4-(r-t)**4)/4
                        if self.nNDoF == 3:
                            self.Sec[i, :, :] = np.array([[   0,  0, 0],
                                                          [   0,  r, 0],
                                                          [   0, -r, 0]])
                        if self.nNDoF == 6:
                            self.Sec[i, :, :] = np.array([[              0,               0, 0],
                                                          [              r,               0, r],
                                                          [ np.sqrt(2)/2*r,  np.sqrt(2)/2*r, r],
                                                          [              0,               r, r],
                                                          [-np.sqrt(2)/2*r,  np.sqrt(2)/2*r, r],
                                                          [             -r,               0, r],
                                                          [-np.sqrt(2)/2*r, -np.sqrt(2)/2*r, r],
                                                          [              0,              -r, r],
                                                          [ np.sqrt(2)/2*r, -np.sqrt(2)/2*r, r]])
                        self.ϰ[i] = 0.847 # needs to be corrected!!!!
                    elif self.Properties[ii][4] in [3, "rect", "Rectangle"]:
                        h = self.Properties[ii][5]
                        b = self.Properties[ii][6]
                        self.A[i] = b*h
                        if h >= b:
                            c1 = 1/3*(1-0.63/(h/b)+0.052/(h/b)**5)
                            c2 = 1-0.65/(1+(h/b)**3)
                            c3 = 0.743+0.514/(1+(h/b)**3)
                            self.Ix[i] = c1*h*b**3
                            if self.nNDoF == 3:
                                self.Sec[i, :, :] = np.array([[   0,    0, 0],
                                                              [   0,  h/2, 0],
                                                              [   0, -h/2, 0]])
                            if self.nNDoF == 6:
                                self.Sec[i, :, :] = np.array([[   0,    0,       0],
                                                              [ b/2,    0,    c2*b],
                                                              [ b/2,  h/2,       0],
                                                              [   0,  h/2, c2*c3*b],
                                                              [-b/2,  h/2,       0],
                                                              [-b/2,    0,    c2*b],
                                                              [-b/2, -h/2,       0],
                                                              [   0, -h/2, c2*c3*b],
                                                              [ b/2, -h/2,       0]])
                        else:
                            c1 = 1/3*(1-0.63/(b/h)+0.052/(b/h)**5)
                            c2 = 1-0.65/(1+(b/h)**3)
                            c3 = 0.743+0.514/(1+(b/h)**3)
                            self.Ix[i] = c1*b*h**3
                            if self.nNDoF == 3:
                                self.Sec[i, :, :] = np.array([[   0,    0, 0],
                                                              [   0,  h/2, 0],
                                                              [   0, -h/2, 0]])
                            if self.nNDoF == 6:
                                self.Sec[i, :, :] = np.array([[   0,    0,       0],
                                                              [ b/2,    0, c2*c3*b],
                                                              [ b/2,  h/2,       0],
                                                              [   0,  h/2,    c2*b],
                                                              [-b/2,  h/2,       0],
                                                              [-b/2,    0, c2*c3*b],
                                                              [-b/2, -h/2,       0],
                                                              [   0, -h/2,    c2*b],
                                                              [ b/2, -h/2,       0]])
                        self.Iy[i] = b*h**3/12
                        self.Iz[i] = h*b**3/12
                        self.ϰ[i] = 10*(1+self.nu[i])/(12+11*self.nu[i])  #Solid rectangular cross-sectional geometry after Cowper (1966)
                    elif self.Properties[ii][4] in [4, "recttube"]:
                        h = self.Properties[ii][5]
                        b = self.Properties[ii][6]
                        t = self.Properties[ii][7]
                        self.A[i] = 2*t*(b+h-2*t)
                        self.Ix[i] = 2*t*(b-t)**2*(h-t)**2/(b+h-2*t)
                        self.Iy[i] = b*h**3/12-(b-2*t)*(h-2*t)**3/12
                        self.Iz[i] = h*b**3/12-(h-2*t)*(b-2*t)**3/12
                        if self.nNDoF == 3:
                            self.Sec[i, :, :] = np.array([[   0,    0, 0],
                                                          [   0,  h/2, 0],
                                                          [   0, -h/2, 0]])
                        if self.nNDoF == 6:
                            x = (b-t)*(h-t)/(b+h-2*t)
                            self.Sec[i, :, :] = np.array([[   0,    0, 0],
                                                          [ b/2,    0, x],
                                                          [ b/2,  h/2, x],
                                                          [   0,  h/2, x],
                                                          [-b/2,  h/2, x],
                                                          [-b/2,    0, x],
                                                          [-b/2, -h/2, x],
                                                          [   0, -h/2, x],
                                                          [ b/2, -h/2, x]])
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
            if self.nNDoF == 3:
                self.EMat[i] = np.diag([self.E[i], self.E[i]])
            elif self.nNDoF == 6:
                self.EMat[i] = np.diag([self.E[i], self.E[i], self.E[i], self.G[i]])
            self.ell[i] = np.linalg.norm(self.Nodes[self.El[i, 1]-1, :] -
                                       self.Nodes[self.El[i, 0]-1, :])
            self.mass += (self.A[i]*self.ell[i]*self.rho[i])
            self.TX[i] = self.TransXMat(i)
            self.T[i] = self.TransMat(i)
            self.idx[i] = np.r_[self.nNDoF*(self.El[i, 0]-1):self.nNDoF*(self.El[i, 0]-1)+self.nNDoF,
                                self.nNDoF*(self.El[i, 1]-1):self.nNDoF*(self.El[i, 1]-1)+self.nNDoF].tolist()
            for j in range(self.nSeg+1):
                ξ = j/(self.nSeg)
                for ii in range(self.nSec):
                    self.B[i, j, ii] = self.StrainDispMat(ξ, self.ell[i], self.Sec[i, ii, 0], self.Sec[i, ii, 1], self.Sec[i, ii, 2])

        if self.plotting:
            self.r0S = np.zeros([self.nEl, self.nNPoC, self.nSeg+1])
            for i in range(self.nEl):
                for j in range(self.nSeg+1):
                    ξ = j/(self.nSeg)
                    self.r0S[i, :, j] = self.TX[i]@self.ShapeMat(ξ, self.ell[i])@self.T[i]@self.r0[self.idx[i]]

    def Assemble(self, MatElem):
        Matrix = np.zeros([self.nNDoF*self.nN, self.nNDoF*self.nN])
        for i in range(self.nEl):
            Matrix[np.ix_(self.idx[i], self.idx[i])] += self.T[i].T@MatElem(i)@self.T[i]
        return Matrix

    def NMat(self, i, ξ):
        NMat = np.zeros([self.nNPoC, self.nNDoF*self.nN])
        NMat[:, self.idx[i]] += self.TX[i]@self.ShapeMat(ξ, self.ell[i])@self.T[i]
        return NMat

    def AssembleOneDirection(self, MatElem):  # needed for FFRF
        Matrix = np.zeros([self.nNPoC, self.nNDoF*self.nN])
        for i in range(self.nEl):
            Matrix[:, self.idx[i]] += self.TX[i]@MatElem(i)@self.T[i]
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
        if not self.ModeledPartialDerivatives:
            self.ModelPartialDerivatives(xDelta)
        self.SensitivityAnalyzed = True
        self.uNabla = np.zeros((len(self.u), np.size(self.DesVar)))
        FPseudo = np.zeros((len(self.F), self.nx))
        for i in range(self.nx):
            FPseudo[:, i] = self.FNabla[:, i]-self.kNabla[:, :, i]@self.u
        self.uNabla[self.DoF_DL] = np.linalg.solve(self.k[self.DoF_DL, :][:, self.DoF_DL],
                                                   FPseudo[self.DoF_DL, :])

    def ModelPartialDerivatives(self, xDelta):
        self.ModeledPartialDerivatives = True
        if not self.SensitivityAnalyzed:
            self.kNabla = np.zeros([self.nNDoF*self.nN, self.nNDoF*self.nN, self.nx])
            self.FNabla = np.zeros([self.nNDoF*self.nN, self.nx])
        self.TNabla = np.zeros([self.nEl, 2*self.nNDoF, 2*self.nNDoF, self.nx])
        self.BNabla = np.zeros([self.nEl, self.nSeg+1, self.nSec, self.nSVal, 2*self.nNDoF, self.nx])
        self.massNabla = np.zeros([self.nx])
        self.EMatNabla = np.zeros((self.nEl, self.nSVal, self.nSVal, self.nx))
        for i in range(self.nx):
            new = deepcopy(self)
            xPert = xDelta*(1+getattr(new, new.DesVar[i]))
            setattr(new, new.DesVar[i],
                    getattr(new, new.DesVar[i])+xPert)
            new.Initialize()
            if not self.SensitivityAnalyzed:
                new.k = new.Assemble(new.StiffMatElem)
                self.kNabla[:, :, i] = (new.k-self.k)/xPert
                self.FNabla[:, i] = (new.F-self.F)/xPert
            self.TNabla[:, :, :, i] = (new.T-self.T)/xPert
            self.BNabla[:, :, :, :, :, i] = (new.B-self.B)/xPert
            self.EMatNabla[:, :, :, i] = (new.EMat-self.EMat)/xPert
            self.massNabla[i] = (new.mass-self.mass)/xPert

    def ComputeDisplacement(self):
        self.ComputedDisplacement = True
        if not self.StaticAnalyzed:
            self.StaticAnalysis()
        self.uE = np.zeros([self.nEl, 2*self.nNDoF])
        self.uS = np.zeros([self.nEl, self.nNPoC, self.nSeg+1])
        for iEl in range(self.nEl):
            self.uE[iEl]  = self.T[iEl]@self.u[self.idx[iEl]]
            for j in range(self.nSeg+1):
                ξ = j/(self.nSeg)
                self.uS[iEl, :, j] = self.TX[iEl]@self.ShapeMat(ξ, self.ell[iEl])@self.uE[iEl]
        self.uSmag = np.sqrt(np.sum(self.uS**2, axis=1))

    def ComputeInternalForces(self):
        self.CoputedInternalForces = True
        self.QE = np.zeros([self.nEl, 2*self.nNDoF])
        self.QS = np.zeros([self.nEl, self.nSeg+1, self.nNDoF])
        for iEl in range(self.nEl):
            self.QE[iEl]  = self.StiffMatElem(iEl)@self.uE[iEl]
            for j in range(self.nSeg+1):
                ξ = j/(self.nSeg)
                self.QS[iEl, j] = -self.QE[iEl, 0:self.nNDoF]-ξ*(-self.QE[iEl, self.nNDoF:2*self.nNDoF]-self.QE[iEl, 0:self.nNDoF])

    def ComputeStress(self):
        self.ComputedStress = True
        if not self.ComputedDisplacement:
            self.ComputeDisplacement()
        self.epsilon = np.zeros([self.nEl, self.nSeg+1, self.nSec, self.nSVal])
        self.sigma = np.zeros([self.nEl, self.nSeg+1, self.nSec, self.nSVal])
        self.sigmaEqv = np.zeros([self.nEl, self.nSeg+1, self.nSec])
        self.sigmaEqvMax = np.zeros([self.nEl, self.nSeg+1])
        for iEl in range(self.nEl):
            for j in range(self.nSeg+1):
                for ii in range(self.nSec):
                    self.epsilon[iEl, j, ii] = self.B[iEl, j, ii]@self.uE[iEl]
                    self.sigma[iEl, j, ii] = self.EMat[iEl]@self.epsilon[iEl, j, ii]
                    if self.nNDoF == 3:
                        self.sigmaEqv[iEl, j, ii] = np.sqrt(np.sum(self.sigma[iEl, j, ii, :])**2)
                    elif self.nNDoF == 6:
                        self.sigmaEqv[iEl, j, ii] = np.sqrt(np.sum(self.sigma[iEl, j, ii, :3])**2+3*self.sigma[iEl, j, ii, 3]**2)
                self.sigmaEqvMax[iEl, j] = np.max(self.sigmaEqv[iEl, j, :])

    def ComputeStressSensitivity(self, xDelta=1e-9):
        if not self.SensitivityAnalyzed:
            self.SensitivityAnalysis()
        if not self.ModeledPartialDerivatives:
            self.ModelPartialDerivatives(xDelta)
        self.uENabla = np.zeros([self.nEl, 2*self.nNDoF, self.nx])
        self.epsilonNabla = np.zeros((self.nEl, self.nSeg+1, self.nSec, self.nSVal, self.nx))
        self.sigmaNabla = np.zeros((self.nEl, self.nSeg+1, self.nSec, self.nSVal, self.nx))
        self.sigmaEqvNabla = np.zeros([self.nEl, self.nSeg+1, self.nSec, self.nx])
        self.sigmaEqvMaxNabla = np.zeros([self.nEl, self.nSeg+1, self.nx])
        for i in range(self.nx):
            for iEl in range(self.nEl):
                self.uENabla[iEl, :, i]  = self.T[iEl]@self.uNabla[self.idx[iEl], i]+self.TNabla[iEl, :, :, i]@self.u[self.idx[iEl]]
                for j in range(self.nSeg+1):
                    for ii in range(self.nSec):
                        self.epsilonNabla[iEl, j, ii, :, i] = self.BNabla[iEl, j, ii, :, :, i]@self.uE[iEl] + self.B[iEl, j, ii]@self.uENabla[iEl, :, i]
                        self.sigmaNabla[iEl, j, ii, :, i] = self.EMat[iEl]@self.epsilonNabla[iEl, j, ii, :, i] + self.EMatNabla[iEl, :, :, i]@self.epsilon[iEl, j, ii]
                        if self.sigmaEqv[iEl, j, ii] == 0:
                            self.sigmaEqvNabla[iEl, j, ii, i] = 0
                        else:
                            if self.nNDoF == 3:
                                self.sigmaEqvNabla[iEl, j, ii, i] = np.sum(self.sigmaNabla[iEl, j, ii, :, i]) * np.sum(self.sigma[iEl, j, ii, :]) / self.sigmaEqv[iEl, j, ii]
                            elif self.nNDoF == 6:
                                self.sigmaEqvNabla[iEl, j, ii, i] = (np.sum(self.sigma[iEl, j, ii, :3]) * np.sum(self.sigmaNabla[iEl, j, ii, :3, i]) +
                                                                     3*self.sigma[iEl, j, ii, 3] * self.sigmaNabla[iEl, j, ii, 3, i]) / self.sigmaEqv[iEl, j, ii]
                    self.sigmaEqvMaxNabla[iEl, j, i] = self.sigmaEqvNabla[iEl, j, np.argmax(self.sigmaEqv[iEl, j, :]), i]

    def EigenvalueAnalysis(self, nEig=2):
        if not self.Initialized:
            self.Initialize()
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
    from EasyBeam.BeamPlotting import _plotting2D as _plotting
    from EasyBeam.BeamPlotting import PlotMesh2D as PlotMesh
    nNDoF = 3   # number of nodal degrees of freedom
    nNPoC = 2   # number of nodal position coordinates
    nSVal = 2   # number of strain/stress values
    nSec = 3   # number of section points (integration points in the section)

class Beam3D(Beam):
    """
    I need Ix, Iy, Iz...
    """
    from EasyBeam.Beam3D import (ShapeMat, TransXMat, TransMat, StrainDispMat,
                                 StiffMatElem, MassMatElem)
    from EasyBeam.BeamPlotting import _plotting3D as _plotting
    from EasyBeam.BeamPlotting import PlotMesh3D as PlotMesh
    nNDoF = 6   # number of nodal degrees of freedom
    nNPoC = 3   # number of nodal position coordinates
    nSVal = 4   # number of strain/stress values
    nSec = 9   # number of section points (integration points in the section)

class BeamFFRF2D(Beam2D):
    from EasyBeam.BeamFFRF2D import (StfElem, SrfElem, FFRF_Output,
                                     FFRF_OutputSensitivities)

class BeamFFRF3D(Beam3D):
    from EasyBeam.BeamFFRF3D import (S__Elem, S11Elem, S22Elem, S33Elem,
                                     S12Elem, S13Elem,S23Elem,FFRF_Output,
                                     FFRF_OutputSensitivities)

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
            self.Properties = [['Steel', 7.85e-9, 210000, 0.3, "rect", self.h1, self.b1],
                               [  'Alu', 2.70e-9,  70000, 0.3, "rect", self.h2, self.b2]]
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
        Test.ComputeStress()
        return(Test.u, Test.sigma, Test.sigmaEqvMax)

    u0, sigma0, sigmaEqvMax0 = Eval(x0)
    uNabla = np.zeros([len(u0), len(x0)])
    sigmaNabla = np.zeros([sigma0.shape[0], sigma0.shape[1], sigma0.shape[2], sigma0.shape[3], len(x0)])
    sigmaEqvMaxNabla = np.zeros([sigmaEqvMax0.shape[0], sigmaEqvMax0.shape[1], len(x0)])
    for i in range(len(x0)):
        e = np.zeros_like(x0)
        e[i] = 1
        u1, sigma1, sigmaEqvMax1 = Eval(x0+e*xDelta)
        uNabla[:, i] = (u1-u0)/xDelta
        sigmaNabla[:, :, :, :, i] = (sigma1-sigma0)/xDelta
        sigmaNabla[:, :, :, :, i] = (sigma1-sigma0)/xDelta
        sigmaEqvMaxNabla[:, :, i] = (sigmaEqvMax1-sigmaEqvMax0)/xDelta

    t2 = time.time()

    # np.set_printoptions(precision=6, suppress=True)
    for i in range(len(x0)):
        print("\ndisplacement sensitivity "+str(Test.DesVar[i]))
        print(np.linalg.norm(uNabla[:, i]-Test.uNabla[:, i]))
        print("FD:\n", uNabla[:, i])
        print("Analytical:\n", Test.uNabla[:, i])
    for i in range(len(x0)):
        print("\nstress sensitivity "+str(Test.DesVar[i]))
        print(np.linalg.norm(sigmaNabla[:, :, :, :, i]-Test.sigmaNabla[:, :, :, :, i]))
        print("FD:\n", sigmaNabla[:, :, :, :, i])
        print("Analytical:\n", Test.sigmaNabla[:, :, :, :, i])
    for i in range(len(x0)):
        print("\nequivalent tress sensitivity "+str(Test.DesVar[i]))
        print(np.linalg.norm(sigmaEqvMaxNabla[:, :, i]-Test.sigmaEqvMaxNabla[:, :, i]))
        print("FD:\n", sigmaEqvMaxNabla[:, :, i])
        print("Analytical:\n", Test.sigmaEqvMaxNabla[:, :, i])
    print()
    print("summary:")
    rtol = 1e-6
    for i in range(len(x0)):
        print("du/" + Test.DesVar[i]  + ": " + str(np.isclose(uNabla[:, i], Test.uNabla[:, i], rtol=rtol)))
    for i in range(len(x0)):
        print("dsigma/" + Test.DesVar[i]  + ": ")
        print(str(np.isclose(sigmaNabla[:, :, :, :, i], Test.sigmaNabla[:, :, :, :, i], rtol=rtol)))
    for i in range(len(x0)):
        print("dsigmaEqv/" + Test.DesVar[i]  + ": ")
        print(str(np.isclose(sigmaEqvMaxNabla[:, :, i], Test.sigmaEqvMaxNabla[:, :, i], rtol=rtol)))

    print("\ncomputation time analytical:", t1-t0)
    print("\ncomputation time numerical:", t2-t1)

    print("u")
    print(np.isclose(Test.uNabla, uNabla, rtol=rtol))
    print("equivalent sigma")
    print(np.isclose(Test.sigmaEqvMaxNabla, sigmaEqvMaxNabla))
    V = Test.NMat(0, 0)
