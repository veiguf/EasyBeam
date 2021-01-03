import numpy as np
import scipy.linalg as spla
import numpy.linalg as npla
from scipy.constants import pi
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.collections as mplcollect


class Beam2D:
    # # define the nodes
    # N = np.array([[   0, 0],
    #               [ 100, 0],
    #               [ 200, 0]])
    # # define the elements: which nodes are connected?
    # El = np.array([[0, 1],
    #                [1, 2]])
    # # boundary conditions and loads
    # BC = []
    # Load = []
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
        self.u = np.empty([3*self.nN, 1])
        self.u[:] = np.nan
        for i in range(len(self.Disp)):
            for ii in range(3):
                if isinstance(self.Disp[i][1][ii], int) or isinstance(self.Disp[i][1][ii], float):
                    self.u[3*self.Disp[i][0]+ii] = self.Disp[i][1][ii]

        # initial forces
        self.F = np.empty([3*self.nN, 1])
        self.F[:] = np.nan
        self.F[self.DoF] = 0
        for i in range(len(self.Load)):
            for ii in range(3):
                if isinstance(self.Load[i][1][ii], int) or isinstance(self.Load[i][1][ii], float):
                    self.F[3*self.Load[i][0]+ii] = self.Load[i][1][ii]

        self.rho = np.zeros([self.nEl, 1])
        self.E = np.zeros([self.nEl, 1])
        self.A = np.zeros([self.nEl, 1])
        self.I = np.zeros([self.nEl, 1])
        self.eU = np.zeros([self.nEl, 1])
        self.eL = np.zeros([self.nEl, 1])
        # lengths and rotations
        self.l = np.zeros([self.nEl, 1])
        self.θ = np.zeros([self.nEl, 1])
        self.T = np.zeros([self.nEl, 6, 6])
        self.r = np.zeros([self.nEl, 3, self.nStep+1])
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
            if self.N[self.El[i, 1], 0] >= self.N[self.El[i, 0], 0]:
                self.θ[i] = np.arctan((self.N[self.El[i, 1], 1]-self.N[self.El[i, 0], 1])/(self.N[self.El[i, 1], 0]-self.N[self.El[i, 0], 0]))
            else:
                self.θ[i] = np.arctan((self.N[self.El[i, 1], 1]-self.N[self.El[i, 0], 1])/(self.N[self.El[i, 1], 0]-self.N[self.El[i, 0], 0]))+pi
            self.T[i, :, :] = np.array([[np.cos(self.θ[i]), -np.sin(self.θ[i]), 0,                 0,                  0, 0],
                                        [np.sin(self.θ[i]),  np.cos(self.θ[i]), 0,                 0,                  0, 0],
                                        [                0,                  0, 1,                 0,                  0, 0],
                                        [                0,                  0, 0, np.cos(self.θ[i]), -np.sin(self.θ[i]), 0],
                                        [                0,                  0, 0, np.sin(self.θ[i]),  np.cos(self.θ[i]), 0],
                                        [                0,                  0, 0,                 0,                  0, 1]],
                                       dtype=float)
            for j in range(self.nStep+1):
                ξ = j/(self.nStep)
                self.r[i, :, j] = np.concatenate((self.N[self.El[i, 0], :],
                                                  self.θ[i]), axis=0)+self.T[i, 0:3, 0:3]@np.array([ξ*self.l[i], 0, 0],dtype=float)

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
            AS = 5*A/6  # for think rechtangular cross-sectional geometry (needs to be calculated from geometry)
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
            if self.massMatrixType[0].lower() == "c":
                c = A*rho*l/420
                m = c*np.array([[140,     0,       0,  70,     0,        0],
                                [  0,   156,    22*l,   0,    54,    -13*l],
                                [  0,  22*l,  4*l**2,   0,  13*l,  -3*l**2],
                                [ 70,     0,       0, 140,     0,        0],
                                [  0,    54,    13*l,   0,   156, -22*l**2],
                                [  0, -13*l, -3*l**2,   0, -22*l,   4*l**2]],
                               dtype=float)
            elif self.massMatrixType[0].lower() == "l":
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

    def Assemble(self, MatElem):
        MatL = np.zeros([self.nEl, 6, 6])
        MatG = np.zeros([self.nEl, 6, 6])
        Mat = np.zeros([3*self.nN, 3*self.nN])
        for i in range(self.nEl):
            MatL[i, :, :] = MatElem(i)
            MatG[i, :, :] = self.T[i]@MatL[i]@self.T[i].T
            Mat[3*self.El[i, 0]:3*self.El[i, 0]+3, 3*self.El[i, 0]:3*self.El[i, 0]+3] += MatG[i, 0:3, 0:3]
            Mat[3*self.El[i, 0]:3*self.El[i, 0]+3, 3*self.El[i, 1]:3*self.El[i, 1]+3] += MatG[i, 0:3, 3:6]
            Mat[3*self.El[i, 1]:3*self.El[i, 1]+3, 3*self.El[i, 0]:3*self.El[i, 0]+3] += MatG[i, 3:6, 0:3]
            Mat[3*self.El[i, 1]:3*self.El[i, 1]+3, 3*self.El[i, 1]:3*self.El[i, 1]+3] += MatG[i, 3:6, 3:6]
        return Mat

    def StaticAnalysis(self):
        self.k = self.Assemble(self.StiffMatElem)
        self.u[self.DoF_DL] = np.linalg.solve(self.k[self.DoF_DL, :][:, self.DoF_DL],
                                           self.F[self.DoF_DL]-
                                           self.k[self.DoF_DL, :][:, self.DL]@self.u[self.DL])
        self.F[self.BC_DL] = self.k[self.BC_DL, :][:, self.DoF_DL]@self.u[self.DoF_DL]

    def EigenvalueAnalysis(self, nEig=2, massMatrixType="consistent"):
        self.massMatrixType = massMatrixType
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

    def ShapeMat(self, ξ, l):
        NL = np.array([[1-ξ,               0,            0, ξ,            0,            0],
                       [  0, 1-3*ξ**2+2*ξ**3, ξ*l*(1-ξ)**2, 0, ξ**2*(3-2*ξ), ξ**2*l*(ξ-1)],
                       [  0,     6*ξ/l*(ξ-1), 1-4*ξ+3*ξ**2, 0,  6*ξ/l*(1-ξ),    ξ*(3*ξ-2)]])
        return NL

    def ComputeStress(self):
        v = np.zeros([self.nEl, 6])
        self.w = np.zeros([self.nEl, 3, self.nStep+1])
        BU = np.zeros([self.nEl, 6])
        BL = np.zeros([self.nEl, 6])
        self.sigmaU = np.zeros([self.nEl, self.nStep+1])
        self.sigmaL = np.zeros([self.nEl, self.nStep+1])
        self.sigmaMax = np.zeros([self.nEl, self.nStep+1])

        for i in range(self.nEl):
            v[i, :] = np.concatenate((self.u[3*self.El[i, 0]:3*self.El[i, 0]+3],
                                      self.u[3*self.El[i, 1]:3*self.El[i, 1]+3]),
                                     axis=0)[:, 0]
            for j in range(self.nStep+1):
                ξ = j/(self.nStep)
                NL = self.ShapeMat(ξ, self.l[i,0])
                self.w[i, :, j] = self.T[i, 0:3, 0:3]@NL@self.T[i].T@v[i, :]
                # upper Fiber
                BU[i, :] = np.array([-1/self.l[i, 0],
                                     1/self.l[i, 0]**2*6*self.eU[i, 0]*(1-2*ξ),
                                     1/self.l[i, 0]*2*self.eU[i, 0]*(2-3*ξ),
                                     1/self.l[i, 0],
                                     1/self.l[i, 0]**2*6*self.eU[i, 0]*(2*ξ-1),
                                     1/self.l[i, 0]*2*self.eU[i, 0]*(1-3*ξ)])
                self.sigmaU[i, j] = self.E[i, 0]*BU[i, :].T@(self.T[i].T@v[i, :])
                # lower Fiber
                BL[i, :] = np.array([-1/self.l[i, 0],
                                     1/self.l[i, 0]**2*6*self.eL[i, 0]*(1-2*ξ),
                                     1/self.l[i, 0]*2*self.eL[i, 0]*(2-3*ξ),
                                     1/self.l[i, 0],
                                     1/self.l[i, 0]**2*6*self.eL[i, 0]*(2*ξ-1),
                                     1/self.l[i, 0]*2*self.eL[i, 0]*(1-3*ξ)])
                self.sigmaL[i, j] = self.E[i, 0]*BL[i, :].T@(self.T[i].T@v[i, :])
                self.sigmaMax[i, j] = max(abs(self.sigmaL[i, j]),
                                          abs(self.sigmaU[i, j]))

        # deformation
        self.q = self.r+self.w*self.Scale

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
            self._plotting(self.sigmaU, self.q,
                           "upper fiber stress $\\sigma_U$\n[MPa]")

        if stress.lower() in ["all", "lower"]:
            self._plotting(self.sigmaL, self.q,
                           "lower fiber stress $\\sigma_L$\n[MPa]")

        if stress.lower() in ["all", "max"]:
            self._plotting(self.sigmaMax, self.q,
                           "maximum stress $\\sigma_{max}$\n[MPa]")

    def PlotDisplacement(self, component="all"):
        if component.lower() in ["mag", "all"]:
            self.d = np.sqrt(self.w[:, 0, :]**2+self.w[:, 1, :]**2)
            self._plotting(self.d, self.q,
                           "deformation\nmagnitude $|u|$\n[mm]")
        if component.lower() in ["x", "all"]:
            self._plotting(self.w[:, 0, :], self.q,
                           "$x$-deformation $u_x$\n[mm]")
        if component.lower() in ["y", "all"]:
            self._plotting(self.w[:, 1, :], self.q,
                           "$y$-deformation $u_y$\n[mm]")

    def PlotMode(self):
        Phii = np.zeros([3*self.nN, 1])
        for ii in range(len(self.omega)):
            Phii[self.DoF, 0] = self.Phi[:, ii]
            vPhi = np.zeros([self.nEl, 6])
            wPhi = np.zeros([self.nEl, 3, self.nStep+1])
            for i in range(self.nEl):
                vPhi[i, :] = np.concatenate((Phii[3*self.El[i, 0]:3*self.El[i, 0]+3],
                                             Phii[3*self.El[i, 1]:3*self.El[i, 1]+3]),
                                            axis=0)[:,0]
                for j in range(self.nStep+1):
                    ξ = j/(self.nStep)
                    NL = self.ShapeMat(ξ, self.l[i, 0])
                    wPhi[i, :, j] = self.T[i, 0:3, 0:3]@NL@self.T[i].T@vPhi[i, :]
            # deformation
            qPhi = self.r+wPhi*self.ScalePhi
            dPhi = np.sqrt(wPhi[:, 0, :]**2+wPhi[:, 1, :]**2)
            self._plotting(dPhi, qPhi, ("mode " + str(ii+1) + "\n" +
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


if __name__ == '__main__':
    Test = Beam2D()
    Test.stiffMatType = "Euler-Bernoulli"  # Euler-Bernoulli or Timoshenko-Ehrenfest
    Test.massMatType = "consistent"        # lumped or consistent
    b = 10      # mm
    h = 20      # mm
    Test.Properties = [['Prop1', 7.85e-9, 210000, b*h, b*h**3/12, h/2, -h/2],
                       ['Prop2', 2.70e-9,  70000, b*h, b*h**3/12, h/2, -h/2]]
    Test.N = [[  0,   0],
              [100,   0],
              [100, 100]]
    Test.El = [[0, 1, 'Prop1'],
               [1, 2, 'Prop2']]
    Test.Disp = [[0, [  0, 0, 'f']],
                 [1, ['f', 0, 'f']]]
    Test.Load = [[2, [100, 0, 'f']]]
    Test.Initialize()
    Test.PlotMesh()
    Test.StaticAnalysis()
    Test.Scale = 200
    Test.ScalePhi = 0.1
    Test.ComputeStress()
    Test.EigenvalueAnalysis(nEig=len(Test.DoF))
    Test.PlotDisplacement()
    Test.PlotStress(stress="all")
    Test.PlotMode()
