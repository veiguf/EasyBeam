import numpy as np
import scipy.linalg as spla
import numpy.linalg as npla
from scipy.constants import pi
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.collections as mplcollect

class Beam2D:

    nStep = 100
    Scale = 1
    ScalePhi = 1
    massMatType = "consistent"
    stiffMatType = "Euler-Bernoulli"
    lineStyleUndeformed = "--"

    def Initialize(self):
        self.N = np.array(self.N, dtype=float)
        El = np.array(self.El)
        self.PropID = El[:, 2]
        self.El = np.array(El[:, 0:2], dtype=int)
        self.nEl = len(self.El[:, 0])     # number of elements
        self.nN = len(self.N[:, 0])       # number of nodes

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
                if isinstance(self.Disp[i][1][ii], int) or isinstance(self.Disp[i][1][ii], float):
                    self.u[3*self.Disp[i][0]+ii] = self.Disp[i][1][ii]

        # initial forces
        self.F = np.empty([3*self.nN])
        self.F[:] = np.nan
        self.F[self.DoF] = 0
        for i in range(len(self.Load)):
            for ii in range(3):
                if isinstance(self.Load[i][1][ii], int) or isinstance(self.Load[i][1][ii], float):
                    self.F[3*self.Load[i][0]+ii] = self.Load[i][1][ii]

        self.rho = np.zeros([self.nEl])
        self.E = np.zeros([self.nEl])
        self.A = np.zeros([self.nEl])
        self.I = np.zeros([self.nEl])
        self.eU = np.zeros([self.nEl])
        self.eL = np.zeros([self.nEl])
        # lengths and rotations
        self.l = np.zeros([self.nEl])
        self.β = np.zeros([self.nEl])
        self.T2 = np.zeros([self.nEl, 2, 2])
        self.T3 = np.zeros([self.nEl, 3, 3])
        self.T6 = np.zeros([self.nEl, 6, 6])
        # self.r = np.zeros([self.nEl, 3, self.nStep+1])
        self.mass = 0
        self.B1 = np.zeros([self.nEl, 6, 3*self.nN])
        self.r0 = np.insert(self.N, 2, 0, axis=1).flatten('C')
        self.r0S = np.zeros([self.nEl, 3, self.nStep+1])
        for i in range(self.nEl):
            for ii in range(len(self.Properties)):
                if self.PropID[i] == self.Properties[ii][0]:
                    self.rho[i] = self.Properties[ii][1]
                    self.E[i] = self.Properties[ii][2]
                    self.A[i] = self.Properties[ii][3]
                    self.I[i] = self.Properties[ii][4]
                    self.eU[i] = self.Properties[ii][5]
                    self.eL[i] = self.Properties[ii][6]
            self.l[i] = np.linalg.norm(self.N[self.El[i, 1], :] -
                                       self.N[self.El[i, 0], :])
            self.mass += (self.A[i]*self.l[i]*self.rho[i])
            if self.N[self.El[i, 1], 0] >= self.N[self.El[i, 0], 0]:
                self.β[i] = np.arctan((self.N[self.El[i, 1], 1]-self.N[self.El[i, 0], 1])/(self.N[self.El[i, 1], 0]-self.N[self.El[i, 0], 0]))
            else:
                self.β[i] = np.arctan((self.N[self.El[i, 1], 1]-self.N[self.El[i, 0], 1])/(self.N[self.El[i, 1], 0]-self.N[self.El[i, 0], 0]))+pi
            self.T3[i] = np.array([[np.cos(self.β[i]), -np.sin(self.β[i]), 0],
                                   [np.sin(self.β[i]),  np.cos(self.β[i]), 0],
                                   [                0,                  0, 1]],
                                        dtype=float)
            self.T2[i] = self.T3[i, 0:2, 0:2]
            self.T6[i] = np.block([[    self.T3[i].T, np.zeros([3, 3])],
                                   [np.zeros([3, 3]),     self.T3[i].T]])
            self.B1[i, 0:3, 3*self.El[i, 0]:3*self.El[i, 0]+3] = np.eye(3)
            self.B1[i, 3:6, 3*self.El[i, 1]:3*self.El[i, 1]+3] = np.eye(3)
            for j in range(self.nStep+1):
                ξ = j/(self.nStep)
                S = self.ShapeMat(ξ, self.l[i])
                self.r0S[i, :, j] = self.T3[i]@S@self.T6[i]@self.B1[i]@self.r0

    def ShapeMat(self, ξ, l):
        NL = np.array([[1-ξ,               0,            0, ξ,            0,            0],
                       [  0, 1-3*ξ**2+2*ξ**3, ξ*l*(1-ξ)**2, 0, ξ**2*(3-2*ξ), ξ**2*l*(ξ-1)],
                       [  0,     6*ξ/l*(ξ-1), 1-4*ξ+3*ξ**2, 0,  6*ξ/l*(1-ξ),    ξ*(3*ξ-2)]])
        return NL

    def StiffMatElem(self, i):
        A = self.A[i]
        E = self.E[i]
        l = self.l[i]
        I = self.I[i]
        # bar (column) terms of stiffness matrix
        k = E*A/l*np.array([[ 1, 0, 0, -1, 0, 0],
                            [ 0, 0, 0,  0, 0, 0],
                            [ 0, 0, 0,  0, 0, 0],
                            [-1, 0, 0,  1, 0, 0],
                            [ 0, 0, 0,  0, 0, 0],
                            [ 0, 0, 0,  0, 0, 0]],
                        dtype=float)
        # Bending terms after Euler-Bernoulli
        if self.stiffMatType[0].lower() == "e":
            phi = 0
        # Bending terms after Timoshenko-Ehrenfest
        elif self.stiffMatType[0].lower() == "t":
            nu = 0.3
            G = E/(2*(1+nu))
            AS = A * 10*(1+nu)/(12+11*nu)  #Solid rectangular cross-sectional geometry after Cowper (1966)
            phi = 12*E*I/(G*AS*l**2)
        c = E*I/(l**3*(1+phi))
        k += c*np.array([[0,   0,            0, 0,    0,            0],
                         [0,  12,          6*l, 0,  -12,          6*l],
                         [0, 6*l, l**2*(4+phi), 0, -6*l, l**2*(2-phi)],
                         [0,   0,            0, 0,    0,            0],
                         [0, -12,         -6*l, 0,   12,         -6*l],
                         [0, 6*l, l**2*(2-phi), 0, -6*l, l**2*(4+phi)]],
                        dtype=float)
        return k

    def MassMatElem(self, i):
        l = self.l[i]
        rho = self.rho[i]
        A = self.A[i]
        if self.stiffMatType[0].lower() == "e":
            if self.massMatType[0].lower() == "c":
                c = A*rho*l/420
                m = c*np.array([[140,     0,       0,  70,     0,       0],
                                [  0,   156,    22*l,   0,    54,   -13*l],
                                [  0,  22*l,  4*l**2,   0,  13*l, -3*l**2],
                                [ 70,     0,       0, 140,     0,       0],
                                [  0,    54,    13*l,   0,   156,   -22*l],
                                [  0, -13*l, -3*l**2,   0, -22*l,  4*l**2]],
                               dtype=float)
            elif self.massMatType[0].lower() == "l":
                alpha = 0
                c = A*rho*l/2
                m = c*np.array([[ 1, 0,            0, 1, 0,              0],
                                [ 0, 1,            0, 0, 0,              0],
                                [ 0, 0, 2*alpha*l**2, 0, 0,              0],
                                [ 1, 0,            0, 1, 0,              0],
                                [ 0, 0,            0, 0, 1,              0],
                                [ 0, 0,            0, 0, 0, 2*alpha*l**2.]],
                               dtype=float)
        elif self.stiffMatType[0].lower() == "t":
            IR = self.I[i]
            nu = 0.3
            G = self.E[i]/(2*(1+nu))
            AS = 5*A/6  # for think rechtangular cross-sectional geometry (needs to be calculated from geometry)
            phi = 12*self.E[i]*self.I[i]/(G*AS*l**2)
            m = A*rho*l/420*np.array([[140, 0, 0,  70, 0, 0],
                                      [  0, 0, 0,   0, 0, 0],
                                      [  0, 0, 0,   0, 0, 0],
                                      [ 70, 0, 0, 140, 0, 0],
                                      [  0, 0, 0,   0, 0, 0],
                                      [  0, 0, 0,   0, 0, 0]],
                                     dtype=float)
            # tranlational inertia
            cT = A*rho*l/(1+phi)**2
            m += cT*np.array([[0,                                 0,                                   0, 0,                                 0,                                   0],
                              [0,         13/35+7/10*phi+1/3*phi**2,   (11/210+11/120*phi+1/24*phi**2)*l, 0,          9/70+3/10*phi+1/6*phi**2,    -(13/420+3/40*phi+1/24*phi**2)*l],
                              [0, (11/210+11/120*phi+1/24*phi**2)*l,  (1/105+1/60*phi+1/120*phi**2)*l**2, 0,   (13/420+3/40*phi+1/24*phi**2)*l, -(1/140+1/60*phi+1/120*phi**2)*l**2],
                              [0,                                 0,                                   0, 0,                                 0,                                   0],
                              [0,          9/70+3/10*phi+1/6*phi**2,     (13/420+3/40*phi+1/24*phi**2)*l, 0,         13/35+7/10*phi+1/3*phi**2,   (11/210+11/120*phi+1/24*phi**2)*l],
                              [0,  -(13/420+3/40*phi+1/24*phi**2)*l, -(1/140+1/60*phi+1/120*phi**2)*l**2, 0, (11/210+11/120*phi+1/24*phi**2)*l,  (1/105+1/60*phi+1/120*phi**2)*l**2]],
                             dtype=float)
            # rotary inertia
            cR = rho*IR/(l*(1+phi)**2)
            m += cR*np.array([[0,                0,                              0, 0,                 0,                              0],
                              [0,              6/5,               (1/10-1/2*phi)*l, 0,              -6/5,               (1/10-1/2*phi)*l],
                              [0, (1/10-1/2*phi)*l, (2/15+1/6*phi+1/3*phi**2)*l**2, 0, (-1/10+1/2*phi)*l, (1/30+1/6*phi-1/6*phi**2)*l**2],
                              [0,                0,                              0, 0,                 0,                              0],
                              [0,             -6/5,              (-1/10+1/2*phi)*l, 0,               6/5,              (-1/10+1/2*phi)*l],
                              [0, (1/10-1/2*phi)*l, (1/30+1/6*phi-1/6*phi**2)*l**2, 0, (-1/10+1/2*phi)*l, (2/15+1/6*phi+1/3*phi**2)*l**2]],
                             dtype=float)
        return m

    def StfElem(self, i):
        l = self.l[i]
        rho = self.rho[i]
        A = self.A[i]
        S = A*rho*l/12*np.array([[6, 0, 0, 6, 0,  0],
                                 [0, 6, l, 0, 6, -l]],
                                 dtype=float)
        return S

    def SrfElem(self, i):
        l = self.l[i]
        rho = self.rho[i]
        A = self.A[i]
        S = A*rho*l/60*np.array([[   0, 21, 3*l,    0,  9, -2*l],
                                 [ -21,  0,   0,   -9,  0,    0],
                                 [-3*l,  0,   0, -2*l,  0,    0],
                                 [   0,  9, 2*l,    0, 21, -3*l],
                                 [  -9,  0,   0,  -21,  0,    0],
                                 [ 2*l,  0,   0,  3*l,  0,    0]],
                                dtype=float)
        return S

    def Assemble(self, MatElem):
        Matrix = np.zeros([3*self.nN, 3*self.nN])
        for i in range(self.nEl):
            Matrix += self.B1[i].T@self.T6[i].T@MatElem(i)@self.T6[i]@self.B1[i]
        return Matrix

    def Assemble2x6(self, MatElem):
        Matrix = np.zeros([2, 3*self.nN])
        for i in range(self.nEl):
            Matrix += self.T2[i]@MatElem(i)@self.T6[i]@self.B1[i]
        return Matrix

    def StaticAnalysis(self):
        self.k = self.Assemble(self.StiffMatElem)
        self.u[self.DoF_DL] = np.linalg.solve(self.k[self.DoF_DL, :][:, self.DoF_DL],
                                           self.F[self.DoF_DL]-
                                           self.k[self.DoF_DL, :][:, self.DL]@self.u[self.DL])
        self.F[self.BC_DL] = self.k[self.BC_DL, :][:, self.DoF_DL]@self.u[self.DoF_DL]
        self.r = self.r0+self.u

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

    def ComputeStress(self):
        # deformation
        self.uE = np.zeros([self.nEl, 6])
        self.uS = np.zeros([self.nEl, 3, self.nStep+1])
        for i in range(self.nEl):
            self.uE[i, :] = self.B1[i]@self.u
            for j in range(self.nStep+1):
                ξ = j/(self.nStep)
                S = self.ShapeMat(ξ, self.l[i])
                self.uS[i, :, j] = self.T3[i]@S@self.T6[i]@self.uE[i, :]
        self.rS = self.r0S+self.uS*self.Scale

        # stress
        BU = np.zeros([self.nEl, 6])
        BL = np.zeros([self.nEl, 6])
        self.sigmaU = np.zeros([self.nEl, self.nStep+1])
        self.sigmaL = np.zeros([self.nEl, self.nStep+1])
        self.sigmaMax = np.zeros([self.nEl, self.nStep+1])
        for i in range(self.nEl):
            for j in range(self.nStep+1):
                ξ = j/(self.nStep)
                # upper Fiber
                BU[i, :] = np.array([-1/self.l[i],
                                     1/self.l[i]**2*6*self.eU[i]*(1-2*ξ),
                                     1/self.l[i]*2*self.eU[i]*(2-3*ξ),
                                     1/self.l[i],
                                     1/self.l[i]**2*6*self.eU[i]*(2*ξ-1),
                                     1/self.l[i]*2*self.eU[i]*(1-3*ξ)])
                self.sigmaU[i, j] = self.E[i]*BU[i, :].T@(self.T6[i]@self.uE[i, :])
                # lower Fiber
                BL[i, :] = np.array([-1/self.l[i],
                                     1/self.l[i]**2*6*self.eL[i]*(1-2*ξ),
                                     1/self.l[i]*2*self.eL[i]*(2-3*ξ),
                                     1/self.l[i],
                                     1/self.l[i]**2*6*self.eL[i]*(2*ξ-1),
                                     1/self.l[i]*2*self.eL[i]*(1-3*ξ)])
                self.sigmaL[i, j] = self.E[i]*BL[i, :].T@(self.T6[i]@self.uE[i, :])
                self.sigmaMax[i, j] = max(abs(self.sigmaL[i, j]),
                                          abs(self.sigmaU[i, j]))

    def FFRF_Output(self):
        if self.stiffMatType[0].lower() == "e" and self.massMatType[0].lower() == "c":
            kff = self.Assemble(self.StiffMatElem)
            Stf = self.Assemble2x6(self.StfElem)
            Srf = self.Assemble(self.SrfElem)
            Sff = self.Assemble(self.MassMatElem)
        else:
            print('Use stiffMatType = "Euler-Bernoulli"\
                  \nand massMatType = "consistent"')
        return self.mass, kff, Stf, Srf, Sff, self.r0

    def _plotting(self, val, disp, title):
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.set_aspect('equal')
        c = np.linspace(val.min(), val.max(), 5)
        norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
        cmap.set_array([])
        lcAll = colorline(disp[:, 0, :], disp[:, 1, :], val, cmap="jet",
                          plot=False)
        for i in range(self.nEl):
            xEl = self.N[self.El[i, 0], 0], self.N[self.El[i, 1], 0]
            yEl = self.N[self.El[i, 0], 1], self.N[self.El[i, 1], 1]
            plt.plot(xEl, yEl, c='gray', lw=1, ls=self.lineStyleUndeformed)
        for i in range(self.nEl):
            lc = colorline(disp[i, 0, :], disp[i, 1, :], val[i, :],
                            cmap="jet", norm=lcAll.norm)
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
            self._plotting(self.sigmaU, self.rS,
                           "upper fiber stress $\\sigma_U$\n[MPa]")

        if stress.lower() in ["all", "lower"]:
            self._plotting(self.sigmaL, self.rS,
                           "lower fiber stress $\\sigma_L$\n[MPa]")

        if stress.lower() in ["all", "max"]:
            self._plotting(self.sigmaMax, self.rS,
                           "maximum stress $\\sigma_{max}$\n[MPa]")

    def PlotDisplacement(self, component="all"):
        if component.lower() in ["mag", "all"]:
            self.dS = np.sqrt(self.uS[:, 0, :]**2+self.uS[:, 1, :]**2)
            self._plotting(self.dS, self.rS,
                           "deformation\nmagnitude $|u|$\n[mm]")
        if component.lower() in ["x", "all"]:
            self._plotting(self.uS[:, 0, :], self.rS,
                           "$x$-deformation $u_x$\n[mm]")
        if component.lower() in ["y", "all"]:
            self._plotting(self.uS[:, 1, :], self.rS,
                           "$y$-deformation $u_y$\n[mm]")

    def PlotMode(self):
        Phii = np.zeros([3*self.nN])
        for ii in range(len(self.omega)):
            Phii[self.DoF] = self.Phi[:, ii]
            uE_Phi = np.zeros([self.nEl, 6])
            uS_Phi = np.zeros([self.nEl, 3, self.nStep+1])
            for i in range(self.nEl):
                uE_Phi[i, :] = self.B1[i]@Phii
                for j in range(self.nStep+1):
                    ξ = j/(self.nStep)
                    S = self.ShapeMat(ξ, self.l[i])
                    uS_Phi[i, :, j] = self.T3[i]@S@self.T6[i]@uE_Phi[i, :]
            # deformation
            rPhi = self.r0S+uS_Phi*self.ScalePhi
            dPhi = np.sqrt(uS_Phi[:, 0, :]**2+uS_Phi[:, 1, :]**2)
            self._plotting(dPhi, rPhi, ("mode " + str(ii+1) + "\n" +
                                        str(round(self.f0[ii], 4)) + " [Hz]"))

    def PlotMesh(self, NodeNumber=True, ElementNumber=True, FontMag=1):
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.set_aspect('equal')
        deltaMax = max(self.N[:, 0].max()-self.N[:, 0].min(),
                       self.N[:, 1].max()-self.N[:, 1].min())
        p = deltaMax*0.0075
        for i in range(self.nEl):
            xEl = self.N[self.El[i, 0], 0], self.N[self.El[i, 1], 0]
            yEl = self.N[self.El[i, 0], 1], self.N[self.El[i, 1], 1]
            plt.plot(xEl, yEl, c='gray', lw=1, ls='-')
        plt.plot(self.N[:, 0], self.N[:, 1], ".k")
        if NodeNumber:
            for i in range(len(self.N)):
                ax.annotate("N"+str(i), (self.N[i, 0]+p, self.N[i, 1]+p),
                            fontsize=5*FontMag, clip_on=False)
        if ElementNumber:
            for i in range(self.nEl):
                posx = (self.N[self.El[i, 0], 0]+self.N[self.El[i, 1], 0])/2
                posy = (self.N[self.El[i, 0], 1]+self.N[self.El[i, 1], 1])/2
                ax.annotate("E"+str(i), (posx+p, posy+p), fontsize=5*FontMag,
                            c="gray", clip_on=False)
        xmin = self.N[:, 0].min()
        xmax = self.N[:, 0].max()
        if self.N[:,1].max()-self.N[:,1].min() < 0.1:
            ymin = -10
            ymax = 10
        else:
            ymin = self.N[:, 1].min()
            ymax = self.N[:, 1].max()
        xdelta = xmax - xmin
        ydelta = ymax - ymin
        buff = 0.1
        plt.xlim(xmin-xdelta*buff, xmax+xdelta*buff)
        plt.ylim(ymin-ydelta*buff, ymax+ydelta*buff)
        plt.show()


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
        elif Output == 'eU':
            return h/2
        elif Output == 'eL':
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
        elif Output == 'eU':
            return h/2
        elif Output == 'eL':
            return -h/2

if __name__ == '__main__':

    Test = Beam2D()

    Test.stiffMatType = "Euler-Bernoulli"  # Euler-Bernoulli or Timoshenko-Ehrenfest
    Test.massMatType = "consistent"        # lumped or consistent

    b = 10      # mm
    h = 20      # mm
    Test.Properties = [['Steel', 7.85e-9, 210000, b*h, b*h**3/12, h/2, -h/2],
                       ['Alu', 2.70e-9,  70000, b*h, b*h**3/12, h/2, -h/2]]

    Test.N = [[  0,   0],
              [100,   0],
              [100, 100]]

    Test.El = [[0, 1, 'Alu'],
               [1, 2, 'Steel']]

    Test.Disp = [[0, [  0, 0, 'f']],
                 [1, ['f', 0, 'f']]]
    Test.Load = [[2, [800, 0, 'f']]]

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
