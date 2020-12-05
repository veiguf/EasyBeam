import numpy as np
from scipy.constants import pi
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.collections as mplcollect
import scipy.linalg as linalg

class Beam2D:
    # define the nodes
    N = np.array([[   0, 0],
                  [ 100, 0],
                  [ 200, 0]])
    # define the elements: which nodes are connected?
    El = np.array([[0, 1],
                   [1, 2]])
    # boundary conditions and loads
    BC = []
    Load = []
    nStep = 150
    Scale = 1

    def Initialize(self):
        self.N = np.array(self.N, dtype=float)
        self.El = np.array(self.El, dtype=int)
        self.nEl = len(self.El[:, 0])     # number of elements
        self.nN = len(self.N[:, 0])       # number of nodes

        # initial displacements
        self.u = np.empty([3*self.nN, 1])
        self.u[:] = np.nan
        self.u[self.BC] = 0

        self.DoF = []   # degrees-of-freedom
        for i in range(3*self.nN):
            if i not in self.BC:
                self.DoF.append(i)

        # initial forces
        self.F = np.empty([3*self.nN, 1])
        self.F[:] = np.nan
        self.F[self.DoF] = 0
        for i in range(len(self.Load)):
            self.F[self.Load[i][0]] = self.Load[i][1]

    def StiffMatElem(self, i):
        A = self.A[i]
        E = self.E[i]
        l = self.l[i]
        I = self.I[i]
        k = np.array([[ A*E/l,            0,           0, -A*E/l,            0,           0],
                      [     0,  12*E*I/l**3,  6*E*I/l**2,      0, -12*E*I/l**3,  6*E*I/l**2],
                      [     0,   6*E*I/l**2,     4*E*I/l,      0,  -6*E*I/l**2,     2*E*I/l],
                      [-A*E/l,            0,           0,  A*E/l,            0,           0],
                      [     0, -12*E*I/l**3, -6*E*I/l**2,      0,  12*E*I/l**3, -6*E*I/l**2],
                      [     0,   6*E*I/l**2,     2*E*I/l,      0,  -6*E*I/l**2,     4*E*I/l]],
                     dtype='f')
        return k

    def MassMatElem(self, i):
        l = self.l[i]
        if self.massMatrixType[0].lower() == "c":
            m = np.array([[140,     0,       0,  70,     0,        0],
                          [  0,   156,    22*l,   0,    54,    -13*l],
                          [  0,  22*l,  4*l**2,   0,  13*l,  -3*l**2],
                          [ 70,     0,       0, 140,     0,        0],
                          [  0,    54,    13*l,   0,   156, -22*l**2],
                          [  0, -13*l, -3*l**2,   0, -22*l,   4*l**2]],
                         dtype=float)
            m *= self.A[i]*l*self.rho[i]/420
        elif self.massMatrixType[0].lower() == "l":
            alpha = 0
            m = np.array([[ 1, 0,            0, 1, 0,              0],
                          [ 0, 1,            0, 0, 0,              0],
                          [ 0, 0, 2*alpha*l**2, 0, 0,              0],
                          [ 1, 0,            0, 1, 0,              0],
                          [ 0, 0,            0, 0, 1,              0],
                          [ 0, 0,            0, 0, 0, 2*alpha*l**2.]],
                         dtype=float)
            m *= self.rho[i]*self.A[i]*l/2
        return m

    def Assemble(self, MatElem):
        MatL = np.zeros([self.nEl, 6, 6])
        self.l = np.zeros([self.nEl, 1])
        self.θ = np.zeros([self.nEl, 1])
        MatG = np.zeros([self.nEl, 6, 6])
        self.T = np.zeros([self.nEl, 6, 6])
        Mat = np.zeros([3*self.nN, 3*self.nN])

        for i in range(self.nEl):
            # self.El[i, 1]
            self.l[i] = np.linalg.norm(self.N[self.El[i, 1], :] -
                                       self.N[self.El[i, 0], :])
            MatL[i, :, :] = MatElem(i)
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
            MatG[i, :, :] = self.T[i]@MatL[i]@self.T[i].T
            Mat[3*self.El[i, 0]:3*self.El[i, 0]+3, 3*self.El[i, 0]:3*self.El[i, 0]+3] += MatG[i, 0:3, 0:3]
            Mat[3*self.El[i, 0]:3*self.El[i, 0]+3, 3*self.El[i, 1]:3*self.El[i, 1]+3] += MatG[i, 0:3, 3:6]
            Mat[3*self.El[i, 1]:3*self.El[i, 1]+3, 3*self.El[i, 0]:3*self.El[i, 0]+3] += MatG[i, 3:6, 0:3]
            Mat[3*self.El[i, 1]:3*self.El[i, 1]+3, 3*self.El[i, 1]:3*self.El[i, 1]+3] += MatG[i, 3:6, 3:6]
        return Mat

    def StaticAnalysis(self):
        self.k = self.Assemble(self.StiffMatElem)
        self.u[self.DoF] = np.linalg.solve(self.k[self.DoF, :][:, self.DoF],
                                           self.F[self.DoF])
        self.F[self.BC] = self.k[self.BC, :][:, self.DoF]@self.u[self.DoF]

    def EigenvalueAnalysis(self, nEig=2, massMatrixType="consistent"):
        self.massMatrixType = massMatrixType
        self.k = self.Assemble(self.StiffMatElem)
        self.m = self.Assemble(self.MassMatElem)
        #try:
        lambdaComplex, self.Phi = linalg.eigh(self.k[self.DoF, :][:, self.DoF],
                                              self.m[self.DoF, :][:, self.DoF],
                                              eigvals=(0, nEig-1))
        #except:
        #    lambdaComplex, self.Phi = linalg.eig(self.k[self.DoF, :][:, self.DoF],
        #                                         self.m[self.DoF, :][:, self.DoF])
        self.omega = np.sqrt(lambdaComplex.real)
        self.f0 = self.omega/2/np.pi

    def ComputeStress(self):
        NL = np.zeros([self.nEl, 3, 6])
        self.r = np.zeros([self.nEl, 3, self.nStep+1])
        self.w = np.zeros([self.nEl, 3, self.nStep+1])
        BU = np.zeros([self.nEl, 6])
        BL = np.zeros([self.nEl, 6])
        v = np.zeros([self.nEl, 6])
        self.sigmaU = np.zeros([self.nEl, self.nStep+1])
        self.sigmaL = np.zeros([self.nEl, self.nStep+1])
        self.sigmaMax = np.zeros([self.nEl, self.nStep+1])

        for i in range(self.nEl):
            v[i, :] = np.concatenate((self.u[3*self.El[i, 0]:3*self.El[i, 0]+3],
                                      self.u[3*self.El[i, 1]:3*self.El[i, 1]+3]),
                                     axis=0)[:,0]
            for j in range(self.nStep+1):
                ξ = j/(self.nStep)
                NL = np.array([[1-ξ,                     0,                      0, ξ,                     0,                      0],
                               [  0,       1-3*ξ**2+2*ξ**3, ξ*self.l[i,0]*(1-ξ)**2, 0,          ξ**2*(3-2*ξ), ξ**2*self.l[i,0]*(ξ-1)],
                               [  0, 6*ξ/self.l[i,0]*(ξ-1),           1-4*ξ+3*ξ**2, 0, 6*ξ/self.l[i,0]*(1-ξ),              ξ*(3*ξ-2)]])
                self.r[i, :, j] = np.concatenate((self.N[self.El[i, 0], :], self.θ[i]), axis=0)+self.T[i, 0:3, 0:3]@np.array([ξ*self.l[i], 0, 0])
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

    def _plotting(self, val, title):
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.set_aspect('equal')
        c = np.linspace(val.min(), val.max(), 5)
        norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
        cmap.set_array([])
        lcAll = colorline(self.q[:, 0, :], self.q[:, 1, :], val, cmap="jet",
                          plot=False)
        for i in range(self.nEl):
            xEl = self.N[self.El[i, 0], 0], self.N[self.El[i, 1], 0]
            yEl = self.N[self.El[i, 0], 1], self.N[self.El[i, 1], 1]
            plt.plot(xEl, yEl, c='gray', lw=1, ls='-')
            lc = colorline(self.q[i, 0, :], self.q[i, 1, :], val[i, :],
                           cmap="jet", norm=lcAll.norm)
        cb = plt.colorbar(lcAll, ticks=c, shrink=0.5, ax=[ax], location="left",
                          aspect=10)
        #cb = plt.colorbar(lcAll, ticks=c, shrink=0.4, orientation="horizontal")
        xmin = self.q[:, 0, :].min()
        xmax = self.q[:, 0, :].max()
        ymin = self.q[:, 1, :].min()
        ymax = self.q[:, 1, :].max()
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
            self._plotting(self.sigmaU,
                           "upper fiber stress $\\sigma_U$\n[MPa]")

        if stress.lower() in ["all", "lower"]:
            self._plotting(self.sigmaL,
                           "lower fiber stress $\\sigma_U$\n[MPa]")

        if stress.lower() in ["all", "max"]:
            self._plotting(self.sigmaMax,
                      "maximum stress $\\sigma_{max}$\n[MPa]")

    def PlotDisplacement(self, component="all"):
        if component.lower() in ["mag", "all"]:
            self.d = np.sqrt(self.w[:, 0, :]**2+self.w[:, 1, :]**2)
            self._plotting(self.d, "deformation\nmagnitude $|u|$\n[mm]")
        if component.lower() in ["x", "all"]:
            self._plotting(self.w[:, 0, :], "$x$-deformation $u_x$\n[mm]")
        if component.lower() in ["y", "all"]:
            self._plotting(self.w[:, 1, :], "$y$-deformation $u_y$\n[mm]")

    def PlotMode(self):
        Phii = np.empty([3*self.nN, 1])
        for i in range(len(self.omega)):
            Phii[self.DoF, 0] = self.Phi[:, i]
            # THIS NEXT LINE IS WRONG, NEEDS TO BE FUNCTION OF Phii!!!!!!!!
            self.d = np.sqrt(self.w[:, 0, :]**2+self.w[:, 1, :]**2)
            self._plotting(self.d, ("mode " + str(i+1) + "\n" +
                                    str(round(self.f0[i], 4)) + " [Hz]"))

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
                ax.annotate("N"+str(i+1), (self.N[i, 0]+p, self.N[i, 1]+p),
                            fontsize=5*FontMag, clip_on=False)
        if ElementNumber:
            for i in range(self.nEl):
                posx = (self.N[self.El[i, 0], 0]+self.N[self.El[i, 1], 0])/2
                posy = (self.N[self.El[i, 0], 1]+self.N[self.El[i, 1], 1])/2
                ax.annotate("E"+str(i+1), (posx+p, posy+p), fontsize=5*FontMag,
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
    Test.N = [[  0,   0],
              [100,   0],
              [100, 100]]
    Test.El = [[0, 1],
               [1, 2]]
    Test.BC = [0, 1, 2]
    Test.Load = [[6,  100],
                 [7, -100]]
    Test.Initialize()
    Test.PlotMesh()
    b = 10      # mm
    h = 10      # mm
    Test.eU = np.ones([Test.nEl, 1])*h/2
    Test.eL = np.ones([Test.nEl, 1])*-h/2
    Test.A = np.ones([Test.nEl, 1])*b*h         # mm^2
    Test.I = np.ones([Test.nEl, 1])*b*h**3/12   # mm^4
    Test.E = np.ones([Test.nEl, 1])*210000      # MPa
    Test.rho = np.ones([Test.nEl, 1])*7.85e-9   # t/mm^3
    Test.StaticAnalysis()
    Test.Scale = 5
    Test.ComputeStress()
    Test.EigenvalueAnalysis(nEig=5, massMatrixType="c")
    #Test.PlotStress(stress="all")
    #Test.PlotDisplacement()
    Test.PlotMode()
