import numpy as np
import scipy.linalg as spla
import numpy.linalg as npla
from scipy.constants import pi
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.collections as mplcollect
from copy import deepcopy

class Beam2D:
    nSeg = 100
    Scale = 1
    ScalePhi = 1
    massMatType = "consistent"
    stiffMatType = "Euler-Bernoulli"
    lineStyleUndeformed = "-"
    colormap = "RdBu" #"coolwarm_r" #"Blues"
    SizingVariables = []
    plotting = True

    def Initialize(self):
        self.Nodes = np.array(self.Nodes, dtype=float)
        self.El = np.array(self.El, dtype=int)
        self.nEl = len(self.El)     # number of elements
        self.nN = len(self.Nodes[:, 0])       # number of nodes
        self.BC = []        # boundary conditions
        self.BC_DL = []
        self.DL = []        # displacement load
        for i in range(len(self.Disp)):
            for ii in range(3):
                entry = self.Disp[i][1][ii]
                if isinstance(entry, int) or isinstance(entry, float):
                    self.BC_DL.append(3*self.Disp[i][0]+ii)
                    if entry == 0:
                        self.BC.append(3*self.Disp[i][0]+ii)
                    else:
                        self.DL.append(3*self.Disp[i][0]+ii)

        self.DoF = []       # degrees-of-freedom
        self.DoF_DL = []
        for i in range(3*self.nN):
            if i not in self.BC:
                self.DoF.append(i)
            if i not in self.BC_DL:
                self.DoF_DL.append(i)

        # initial displacements
        self.u = np.empty([3*self.nN])
        self.u[:] = np.nan
        for i in range(len(self.Disp)):
            for ii in range(3):
                if (isinstance(self.Disp[i][1][ii], int) or
                    isinstance(self.Disp[i][1][ii], float)):
                    self.u[3*self.Disp[i][0]+ii] = self.Disp[i][1][ii]

        # initial forces
        self.F = np.empty([3*self.nN])
        self.F[:] = np.nan
        self.F[self.DoF] = 0
        for i in range(len(self.Load)):
            for ii in range(3):
                if (isinstance(self.Load[i][1][ii], int) or
                    isinstance(self.Load[i][1][ii], float)):
                    self.F[3*self.Load[i][0]+ii] = self.Load[i][1][ii]

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
        self.r0S = np.zeros([self.nEl, 2, self.nSeg+1])
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
                    else:
                        print("oops nothing more programmed!!!")
            self.ell[i] = np.linalg.norm(self.Nodes[self.El[i, 1], :] -
                                       self.Nodes[self.El[i, 0], :])
            self.mass += (self.A[i]*self.ell[i]*self.rho[i])
            if self.Nodes[self.El[i, 1], 0] >= self.Nodes[self.El[i, 0], 0]:
                """
                HERE is a division by zero...needs to be checked
                """
                self.β[i] = np.arctan((self.Nodes[self.El[i, 1], 1] -
                                       self.Nodes[self.El[i, 0], 1])/
                                      (self.Nodes[self.El[i, 1], 0] -
                                       self.Nodes[self.El[i, 0], 0]))
            else:
                self.β[i] = pi + np.arctan((self.Nodes[self.El[i, 1], 1] -
                                            self.Nodes[self.El[i, 0], 1])/
                                           (self.Nodes[self.El[i, 1], 0] -
                                            self.Nodes[self.El[i, 0], 0]))
            self.T2[i] = np.array([[np.cos(self.β[i]), -np.sin(self.β[i])],
                                  [np.sin(self.β[i]),  np.cos(self.β[i])]],
                                 dtype=float)
            self.T[i] = np.block([[    self.T2[i].T, np.zeros([2, 4])],
                                  [0, 0, 1, 0, 0, 0],
                                  [np.zeros([2, 3]), self.T2[i].T, np.zeros([2, 1])],
                                  [0, 0, 0, 0, 0, 1]])
            self.L[i, 0:3, 3*self.El[i, 0]:3*self.El[i, 0]+3] = np.eye(3)
            self.L[i, 3:6, 3*self.El[i, 1]:3*self.El[i, 1]+3] = np.eye(3)
            if self.plotting:
                for j in range(self.nSeg+1):
                    ξ = j/(self.nSeg)
                    self.r0S[i, :, j] = self.T2[i]@self.ShapeMat(ξ, self.ell[i])@self.T[i]@self.L[i]@self.r0

    def ShapeMat(self, ξ, ell):
        return(np.array([[1-ξ,               0,              0, ξ,            0,              0],
                         [  0, 1-3*ξ**2+2*ξ**3, ξ*ell*(1-ξ)**2, 0, ξ**2*(3-2*ξ), ξ**2*ell*(ξ-1)]]))

    def NMat(self, i, ξ):
        NMat = self.T2[i]@self.ShapeMat(ξ, self.ell[i])@self.T[i]@self.L[i]
        return NMat

    def StrainDispMat(self, ξ, ell, zU, zL):
        BL = np.array([[-1/ell,                  0,               0, 1/ell,                   0,               0],
                       [     0, zL*(6-12*ξ)/ell**2,  zL*(4-6*ξ)/ell,     0, zL*(-6+12*ξ)/ell**2, zL*(-6*ξ+2)/ell]])
        BU = np.array([[-1/ell,                  0,               0, 1/ell,                   0,               0],
                       [     0, zU*(6-12*ξ)/ell**2,  zU*(4-6*ξ)/ell,     0, zU*(-6+12*ξ)/ell**2, zU*(-6*ξ+2)/ell]])
        return(BL, BU)

    def StrainDispNablah(self, ξ, ell):
        BLNablah = np.array([[0,                0,            0, 0,                 0,             0],
                             [0, -1/2*(6-12*ξ)/ell**2, -1/2*(4-6*ξ)/ell, 0, -1/2*(-6+12*ξ)/ell**2, -1/2*(-6*ξ+2)/ell]])
        BUNablah = np.array([[0,               0,           0, 0,                0,            0],
                             [0, 1/2*(6-12*ξ)/ell**2, 1/2*(4-6*ξ)/ell, 0, 1/2*(-6+12*ξ)/ell**2, 1/2*(-6*ξ+2)/ell]])
        return(BLNablah, BUNablah)

    def StiffMatElem(self, i):
        A = self.A[i]
        E = self.E[i]
        ell = self.ell[i]
        I = self.I[i]
        nu = self.nu[i]
        ϰ = self.ϰ[i]
        # bar (column) terms of stiffness matrix
        k = E*A/ell*np.array([[ 1, 0, 0, -1, 0, 0],
                              [ 0, 0, 0,  0, 0, 0],
                              [ 0, 0, 0,  0, 0, 0],
                              [-1, 0, 0,  1, 0, 0],
                              [ 0, 0, 0,  0, 0, 0],
                              [ 0, 0, 0,  0, 0, 0]], dtype=float)

        # Bending terms after Euler-Bernoulli
        if self.stiffMatType[0].lower() in ["e", "b"]:
            phi = 0
        # Bending terms after Timoshenko-Ehrenfest
        elif self.stiffMatType[0].lower() == "t":
            G = E/(2*(1+nu))
            AS = A * ϰ
            phi = 12*E*I/(ϰ*A*G*l**2)
        c = E*I/(ell**3*(1+phi))
        k += c*np.array([[0,     0,              0, 0,      0,                0],
                         [0,    12,          6*ell, 0,    -12,            6*ell],
                         [0, 6*ell, ell**2*(4+phi), 0, -6*ell,   ell**2*(2-phi)],
                         [0,     0,              0, 0,      0,                0],
                         [0,   -12,         -6*ell, 0,     12,           -6*ell],
                         [0, 6*ell, ell**2*(2-phi), 0, -6*ell,   ell**2*(4+phi)]],
                        dtype=float)
        return k

    def MatMat(self, i):
        return(np.array([[self.E[i]*self.A[i],                   0],
                         [                  0, self.E[i]*self.I[i]]]))

    def MassMatElem(self, i):
        ell = self.ell[i]
        rho = self.rho[i]
        A = self.A[i]
        if self.stiffMatType[0].lower() in ["e", "b"]:
            if self.massMatType[0].lower() == "c":
                c = A*rho*ell/420
                m = c*np.array([[140,       0,         0,  70,       0,         0],
                                [  0,     156,    22*ell,   0,      54,   -13*ell],
                                [  0,  22*ell,  4*ell**2,   0,  13*ell, -3*ell**2],
                                [ 70,       0,         0, 140,       0,         0],
                                [  0,      54,    13*ell,   0,     156,   -22*ell],
                                [  0, -13*ell, -3*ell**2,   0, -22*ell,  4*ell**2]],
                               dtype=float)
            elif self.massMatType[0].lower() == "l":
                alpha = 0
                c = A*rho*ell/2
                m = c*np.array([[ 1, 0,              0, 1, 0,                0],
                                [ 0, 1,              0, 0, 0,                0],
                                [ 0, 0, 2*alpha*ell**2, 0, 0,                0],
                                [ 1, 0,              0, 1, 0,                0],
                                [ 0, 0,              0, 0, 1,                0],
                                [ 0, 0,              0, 0, 0, 2*alpha*ell**2.]],
                               dtype=float)
        elif self.stiffMatType[0].lower() == "t":
            IR = self.I[i]
            nu = 0.3
            G = self.E[i]/(2*(1+nu))
            AS = A * ϰ
            phi = 12*self.E[i]*self.I[i]/(ϰ*A*G*ell**2)
            m = A*rho*ell/420*np.array([[140, 0, 0,  70, 0, 0],
                                        [  0, 0, 0,   0, 0, 0],
                                        [  0, 0, 0,   0, 0, 0],
                                        [ 70, 0, 0, 140, 0, 0],
                                        [  0, 0, 0,   0, 0, 0],
                                        [  0, 0, 0,   0, 0, 0]], dtype=float)
            # tranlational inertia
            cT = A*rho*ell/(1+phi)**2
            m += cT*np.array([[0,                                   0,                                     0, 0,                                   0,                                     0],
                              [0,           13/35+7/10*phi+1/3*phi**2,   (11/210+11/120*phi+1/24*phi**2)*ell, 0,            9/70+3/10*phi+1/6*phi**2,    -(13/420+3/40*phi+1/24*phi**2)*ell],
                              [0, (11/210+11/120*phi+1/24*phi**2)*ell,  (1/105+1/60*phi+1/120*phi**2)*ell**2, 0,   (13/420+3/40*phi+1/24*phi**2)*ell, -(1/140+1/60*phi+1/120*phi**2)*ell**2],
                              [0,                                   0,                                     0, 0,                                   0,                                     0],
                              [0,            9/70+3/10*phi+1/6*phi**2,     (13/420+3/40*phi+1/24*phi**2)*ell, 0,           13/35+7/10*phi+1/3*phi**2,   (11/210+11/120*phi+1/24*phi**2)*ell],
                              [0,  -(13/420+3/40*phi+1/24*phi**2)*ell, -(1/140+1/60*phi+1/120*phi**2)*ell**2, 0, (11/210+11/120*phi+1/24*phi**2)*ell,  (1/105+1/60*phi+1/120*phi**2)*ell**2]],
                             dtype=float)
            # rotary inertia
            cR = rho*IR/(ell*(1+phi)**2)
            m += cR*np.array([[0,                  0,                                0, 0,                   0,                                0],
                              [0,                6/5,               (1/10-1/2*phi)*ell, 0,                -6/5,               (1/10-1/2*phi)*ell],
                              [0, (1/10-1/2*phi)*ell, (2/15+1/6*phi+1/3*phi**2)*ell**2, 0, (-1/10+1/2*phi)*ell, (1/30+1/6*phi-1/6*phi**2)*ell**2],
                              [0,                  0,                                0, 0,                   0,                                0],
                              [0,               -6/5,              (-1/10+1/2*phi)*ell, 0,                 6/5,              (-1/10+1/2*phi)*ell],
                              [0, (1/10-1/2*phi)*ell, (1/30+1/6*phi-1/6*phi**2)*ell**2, 0, (-1/10+1/2*phi)*ell, (2/15+1/6*phi+1/3*phi**2)*ell**2]],
                             dtype=float)
        return m

    def Assemble(self, MatElem):
        Matrix = np.zeros([3*self.nN, 3*self.nN])
        for i in range(self.nEl):
            Matrix += self.L[i].T@self.T[i].T@MatElem(i)@self.T[i]@self.L[i]
        return Matrix


    def StaticAnalysis(self):
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

    def ComputeStress(self):
        self.uE = np.zeros([self.nEl, 6])
        self.uS = np.zeros([self.nEl, 2, self.nSeg+1])
        self.sigmaU = np.zeros([self.nEl, self.nSeg+1, 2])
        self.sigmaL = np.zeros([self.nEl, self.nSeg+1, 2])
        self.epsilonU = np.zeros([self.nEl, self.nSeg+1, 2])
        self.epsilonL = np.zeros([self.nEl, self.nSeg+1, 2])
        self.sigmaMax = np.zeros([self.nEl, self.nSeg+1])
        self.epsilon = np.zeros([self.nEl, self.nSeg+1, 2])
        self.sigma = np.zeros([self.nEl, self.nSeg+1, 2])
        for iEl in range(self.nEl):
            self.uE[iEl]  = self.T[iEl]@self.L[iEl]@self.u
            for j in range(self.nSeg+1):
                ξ = j/(self.nSeg)
                self.uS[iEl, :, j] = self.T2[iEl]@self.ShapeMat(ξ, self.ell[iEl])@self.uE[iEl]
                BL, BU = self.StrainDispMat(ξ, self.ell[iEl], self.zU[iEl],
                                            self.zL[iEl])
                self.epsilonL[iEl, j] = BL@self.uE[iEl]
                self.epsilonU[iEl, j] = BU@self.uE[iEl]
                self.sigmaL[iEl, j] = self.epsilonL[iEl, j]*self.E[iEl]
                self.sigmaU[iEl, j] = self.epsilonU[iEl, j]*self.E[iEl]
                self.sigmaMax[iEl, j] = np.max((np.abs(np.sum(self.sigmaL[iEl, j])),
                                                np.abs(np.sum(self.sigmaU[iEl, j]))))
        self.rS = self.r0S+self.uS*self.Scale

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
                ix = 0
                for ii in range(len(self.SizingVariables)):
                    for iVar in self.SizingVariables[ii]:
                        BLNabla = np.zeros((2, 6, nx))
                        BUNabla = np.zeros((2, 6, nx))
                        if iVar == "h":
                            BLNabla[:, :, ix], BUNabla[:, :, ix] = \
                                self.StrainDispNablah(ξ, self.ell[iEl])
                        ix += 1
                self.epsilonLNabla[iEl, j] = BLNabla.transpose(0, 2, 1)@self.uE[iEl] + BL@uENabla
                self.epsilonUNabla[iEl, j] = BUNabla.transpose(0, 2, 1)@self.uE[iEl] + BU@uENabla
                self.sigmaLNabla[iEl, j] = self.E[iEl]*self.epsilonLNabla[iEl, j]
                self.sigmaUNabla[iEl, j] = self.E[iEl]*self.epsilonUNabla[iEl, j]

    def EigenvalueAnalysis(self, nEig=2, massMatType="consistent"):
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

    # Functions for FFR
    def StfElem(self, i):
        ell = self.ell[i]
        rho = self.rho[i]
        A = self.A[i]
        return(A*rho*ell/12*np.array([[6, 0,   0, 6, 0,    0],
                                      [0, 6, ell, 0, 6, -ell]], dtype=float))

    def SrfElem(self, i):
        ell = self.ell[i]
        rho = self.rho[i]
        A = self.A[i]
        return(A*rho*ell/60*np.array([[     0, 21, 3*ell,      0,  9, -2*ell],
                                      [   -21,  0,     0,     -9,  0,      0],
                                      [-3*ell,  0,     0, -2*ell,  0,      0],
                                      [     0,  9, 2*ell,      0, 21, -3*ell],
                                      [    -9,  0,     0,    -21,  0,      0],
                                      [ 2*ell,  0,     0,  3*ell,  0,      0]],
                                     dtype=float))

    def Assemble2x6(self, MatElem):
        Matrix = np.zeros([2, 3*self.nN])
        for i in range(self.nEl):
            Matrix += self.T2[i]@MatElem(i)@self.T[i]@self.L[i]
        return Matrix

    def FFRF_Output(self):
        if (self.stiffMatType[0].lower() == "e" and
            self.massMatType[0].lower() == "c"):
            self.kff = self.Assemble(self.StiffMatElem)
            self.Stf = self.Assemble2x6(self.StfElem)
            self.Srf = self.Assemble(self.SrfElem)
            self.Sff = self.Assemble(self.MassMatElem)
        else:
            print('Use stiffMatType = "Euler-Bernoulli"\
                  \nand massMatType = "consistent"')

    def _plotting(self, val, disp, title, colormap):
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.set_aspect('equal')
        c = np.linspace(val.min(), val.max(), 5)
        norm = mpl.colors.Normalize(vmin=-val.max(), vmax=val.max())
        lcAll = colorline(disp[:, 0, :], disp[:, 1, :], val, cmap=colormap,
                          plot=False, norm=MidpointNormalize(midpoint=0.))
        for i in range(self.nEl):
            xEl = self.Nodes[self.El[i, 0], 0], self.Nodes[self.El[i, 1], 0]
            yEl = self.Nodes[self.El[i, 0], 1], self.Nodes[self.El[i, 1], 1]
            plt.plot(xEl, yEl, c='gray', lw=0.5, ls=self.lineStyleUndeformed)
        for i in range(self.nEl):
            lc = colorline(disp[i, 0, :], disp[i, 1, :], val[i, :],
                            cmap=colormap, norm=lcAll.norm)
        cb = plt.colorbar(lcAll, ticks=c, shrink=0.5, ax=[ax], location="left",
                          aspect=10)
        #cb = plt.colorbar(lcAll, ticks=c, shrink=0.4, orientation="horizontal")
        xmin = disp[:, 0, :].min()-1
        xmax = disp[:, 0, :].max()+1
        ymin = disp[:, 1, :].min()-1
        ymax = disp[:, 1, :].max()+1
        xdelta = xmax - xmin
        ydelta = ymax - ymin
        buff = 0.1
        plt.xlim(xmin-xdelta*buff, xmax+xdelta*buff)
        plt.ylim(ymin-ydelta*buff, ymax+ydelta*buff)
        #cb.ax.set_title(title)
        cb.set_label(title, labelpad=0, y=1.1, rotation=0, ha="left")
        plt.show()

    def PlotStress(self, stress="all"):
        if stress.lower() in ["all", "upper"]:
            self._plotting(np.sum(self.sigmaU, 2), self.rS,
                           "upper fiber stress $\\sigma_U$\n[MPa]",
                           self.colormap)

        if stress.lower() in ["all", "lower"]:
            self._plotting(np.sum(self.sigmaL, 2), self.rS,
                           "lower fiber stress $\\sigma_L$\n[MPa]",
                           self.colormap)

        if stress.lower() in ["all", "max"]:
            self._plotting(self.sigmaMax, self.rS,
                           "maximum stress $|\\sigma_{max}|$\n[MPa]",
                           self.colormap)

        if stress.lower() in ["all", "bending"]:
            self._plotting(self.sigmaU[:, :, 1], self.rS,
                           "bending stress\n(upper fiber) $\\sigma_{bending}$\n[MPa]",
                           self.colormap)

        if stress.lower() in ["all", "axial"]:
            self._plotting(self.sigmaU[:, :, 0], self.rS,
                           "axial stress $\\sigma_{axial}$\n[MPa]",
                           self.colormap)

    def PlotDisplacement(self, component="all"):
        if component.lower() in ["mag", "all"]:
            self.dS = np.sqrt(self.uS[:, 0, :]**2+self.uS[:, 1, :]**2)
            self._plotting(self.dS, self.rS,
                           "deformation magnitude $|u|$\n[mm]", self.colormap)
        if component.lower() in ["x", "all"]:
            self._plotting(self.uS[:, 0, :], self.rS,
                           "$x$-deformation $u_x$\n[mm]", self.colormap)
        if component.lower() in ["y", "all"]:
            self._plotting(self.uS[:, 1, :], self.rS,
                           "$y$-deformation $u_y$\n[mm]", self.colormap)

    def PlotMode(self):
        Phii = np.zeros([3*self.nN])
        for ii in range(len(self.omega)):
            Phii[self.DoF] = self.Phi[:, ii]
            uE_Phi = np.zeros([self.nEl, 6])
            uS_Phi = np.zeros([self.nEl, 2, self.nSeg+1])
            for i in range(self.nEl):
                uE_Phi[i, :] = self.L[i]@Phii
                for j in range(self.nSeg+1):
                    ξ = j/(self.nSeg)
                    S = self.ShapeMat(ξ, self.ell[i])
                    uS_Phi[i, :, j] = self.T2[i]@S@self.T[i]@uE_Phi[i, :]
            # deformation
            rPhi = self.r0S+uS_Phi*self.ScalePhi
            dPhi = np.sqrt(uS_Phi[:, 0, :]**2+uS_Phi[:, 1, :]**2)
            self._plotting(dPhi, rPhi, ("mode " + str(ii+1) + "\n" +
                                        str(round(self.f0[ii], 4)) + " [Hz]"),
                                        self.colormap)

    def PlotMesh(self, NodeNumber=True, ElementNumber=True, FontMag=1):
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.set_aspect('equal')
        deltaMax = max(self.Nodes[:, 0].max()-self.Nodes[:, 0].min(),
                       self.Nodes[:, 1].max()-self.Nodes[:, 1].min())
        p = deltaMax*0.0075
        for i in range(self.nEl):
            xEl = self.Nodes[self.El[i, 0], 0], self.Nodes[self.El[i, 1], 0]
            yEl = self.Nodes[self.El[i, 0], 1], self.Nodes[self.El[i, 1], 1]
            plt.plot(xEl, yEl, c='gray', lw=self.A[i]/np.max(self.A), ls='-')
        plt.plot(self.Nodes[:, 0], self.Nodes[:, 1], ".k")
        if NodeNumber:
            for i in range(len(self.Nodes)):
                ax.annotate("N"+str(i+1), (self.Nodes[i, 0]+p,
                                           self.Nodes[i, 1]+p),
                            fontsize=5*FontMag, clip_on=False)
        if ElementNumber:
            for i in range(self.nEl):
                posx = (self.Nodes[self.El[i, 0], 0] +
                        self.Nodes[self.El[i, 1], 0])/2
                posy = (self.Nodes[self.El[i, 0], 1] +
                        self.Nodes[self.El[i, 1], 1])/2
                ax.annotate("E"+str(i+1), (posx+p, posy+p), fontsize=5*FontMag,
                            c="gray", clip_on=False)
        xmin = self.Nodes[:, 0].min()
        xmax = self.Nodes[:, 0].max()
        if self.Nodes[:,1].max()-self.Nodes[:,1].min() < 0.1:
            ymin = -10
            ymax = 10
        else:
            ymin = self.Nodes[:, 1].min()
            ymax = self.Nodes[:, 1].max()
        xdelta = xmax - xmin
        ydelta = ymax - ymin
        buff = 0.1
        plt.xlim(xmin-xdelta*buff, xmax+xdelta*buff)
        plt.ylim(ymin-ydelta*buff, ymax+ydelta*buff)
        plt.show()


import matplotlib.colors as colors
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def colorline(x, y, z, cmap='jet', linewidth=2, alpha=1.0,
              plot=True, norm=None):
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    segments = make_segments(x, y)
    lc = mplcollect.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                                   linewidth=linewidth, alpha=alpha)
    if plot:
        ax = plt.gca()
        ax.add_collection(lc)
    return lc

def make_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

class CrossSections():
    # rectangle
    def R(self, Output, dim):
        b = dim[0]
        h = dim[1]
        if Output == 'A':
            return b*h
        elif Output == 'I':
            return b*h**3/12
        elif Output == 'zU':
            return h/2
        elif Output == 'zL':
            return -h/2

    # C-Profile
    def C(self, Output, dim):
        b = dim[0]
        h = dim[1]
        t = dim[2]
        if Output == 'A':
            return b*h-(b-t)*(h-2*t)
        elif Output == 'I':
            return b*h**3/12-(b-t)*(h-2*t)**3/12
        elif Output == 'zU':
            return h/2
        elif Output == 'zL':
            return -h/2

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

    # Test.Nodes = [[  0,   0],
    #               [ 50,   0],
    #               [100,   0],
    #               [100,  50],
    #               [100, 100]]
    # Test.El = [[0, 1],
    #            [1, 2],
    #            [2, 3],
    #            [3, 4]]
    # Test.PropID = ["Alu", "Alu", "Steel", "Steel"]

    Test.Nodes = [[  0,   0],
                  [100,   0],
                  [100, 100]]
    Test.El = [[0, 1],
               [1, 2]]
    Test.PropID = ["Alu", "Steel"]

    Test.Disp = [[0, [  0, 0, 'f']],
                 [1, ['f', 0, 'f']]]
    Test.Load = [[2, [800, 0, 'f']]]

    Test.nSeg = 1
    Test.Initialize()
    Test.PlotMesh(FontMag=2)

    Test.Scale = 20
    Test.StaticAnalysis()

    Test.ComputeStress()

    Test.PlotDisplacement('all')
    Test.PlotStress('all')

    Test.ScalePhi = 0.1

    Test.EigenvalueAnalysis(nEig=len(Test.DoF))
    Test.PlotMode()

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

