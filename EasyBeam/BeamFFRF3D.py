import numpy as np
from copy import deepcopy
from EasyBeam import BeamFFRF3D

def S__Elem(self, i):
    ell = self.ell[i]
    rho = self.rho[i]
    A = self.A[i]
    return(np.array([[A*ell*rho/2,                0,                0],
                     [          0,      A*ell*rho/2,                0],
                     [          0,                0,      A*ell*rho/2],
                     [          0,                0,                0],
                     [          0,                0, -A*ell**2*rho/12],
                     [          0,  A*ell**2*rho/12,                0],
                     [A*ell*rho/2,                0,                0],
                     [          0,      A*ell*rho/2,                0],
                     [          0,                0,      A*ell*rho/2],
                     [          0,                0,                0],
                     [          0,                0,  A*ell**2*rho/12],
                     [          0, -A*ell**2*rho/12,                0]],
                    dtype=float).T)

def S11Elem(self, i):
    ell = self.ell[i]
    rho = self.rho[i]
    A = self.A[i]
    return(np.array([[A*ell*rho/3, 0, 0, 0, 0, 0, A*ell*rho/6, 0, 0, 0, 0, 0],
                     [          0, 0, 0, 0, 0, 0,           0, 0, 0, 0, 0, 0],
                     [          0, 0, 0, 0, 0, 0,           0, 0, 0, 0, 0, 0],
                     [          0, 0, 0, 0, 0, 0,           0, 0, 0, 0, 0, 0],
                     [          0, 0, 0, 0, 0, 0,           0, 0, 0, 0, 0, 0],
                     [          0, 0, 0, 0, 0, 0,           0, 0, 0, 0, 0, 0],
                     [A*ell*rho/6, 0, 0, 0, 0, 0, A*ell*rho/3, 0, 0, 0, 0, 0],
                     [          0, 0, 0, 0, 0, 0,           0, 0, 0, 0, 0, 0],
                     [          0, 0, 0, 0, 0, 0,           0, 0, 0, 0, 0, 0],
                     [          0, 0, 0, 0, 0, 0,           0, 0, 0, 0, 0, 0],
                     [          0, 0, 0, 0, 0, 0,           0, 0, 0, 0, 0, 0],
                     [          0, 0, 0, 0, 0, 0,           0, 0, 0, 0, 0, 0]],
                    dtype=float))

def S22Elem(self, i):
    ell = self.ell[i]
    rho = self.rho[i]
    A = self.A[i]
    return(np.array([[0,                    0, 0, 0, 0,                   0, 0,                    0, 0, 0, 0,                    0],
                     [0,      13*A*ell*rho/35, 0, 0, 0, 11*A*ell**2*rho/210, 0,       9*A*ell*rho/70, 0, 0, 0, -13*A*ell**2*rho/420],
                     [0,                    0, 0, 0, 0,                   0, 0,                    0, 0, 0, 0,                    0],
                     [0,                    0, 0, 0, 0,                   0, 0,                    0, 0, 0, 0,                    0],
                     [0,                    0, 0, 0, 0,                   0, 0,                    0, 0, 0, 0,                    0],
                     [0,  11*A*ell**2*rho/210, 0, 0, 0,    A*ell**3*rho/105, 0,  13*A*ell**2*rho/420, 0, 0, 0,    -A*ell**3*rho/140],
                     [0,                    0, 0, 0, 0,                   0, 0,                    0, 0, 0, 0,                    0],
                     [0,       9*A*ell*rho/70, 0, 0, 0, 13*A*ell**2*rho/420, 0,      13*A*ell*rho/35, 0, 0, 0, -11*A*ell**2*rho/210],
                     [0,                    0, 0, 0, 0,                   0, 0,                    0, 0, 0, 0,                    0],
                     [0,                    0, 0, 0, 0,                   0, 0,                    0, 0, 0, 0,                    0],
                     [0,                    0, 0, 0, 0,                   0, 0,                    0, 0, 0, 0,                    0],
                     [0, -13*A*ell**2*rho/420, 0, 0, 0,   -A*ell**3*rho/140, 0, -11*A*ell**2*rho/210, 0, 0, 0,     A*ell**3*rho/105]],
                    dtype=float))

def S33Elem(self, i):
    ell = self.ell[i]
    rho = self.rho[i]
    A = self.A[i]
    return(np.array([[0, 0,                    0, 0,                    0, 0, 0, 0,                    0, 0,                   0, 0],
                     [0, 0,                    0, 0,                    0, 0, 0, 0,                    0, 0,                   0, 0],
                     [0, 0,      13*A*ell*rho/35, 0, -11*A*ell**2*rho/210, 0, 0, 0,       9*A*ell*rho/70, 0, 13*A*ell**2*rho/420, 0],
                     [0, 0,                    0, 0,                    0, 0, 0, 0,                    0, 0,                   0, 0],
                     [0, 0, -11*A*ell**2*rho/210, 0,     A*ell**3*rho/105, 0, 0, 0, -13*A*ell**2*rho/420, 0,   -A*ell**3*rho/140, 0],
                     [0, 0,                    0, 0,                    0, 0, 0, 0,                    0, 0,                   0, 0],
                     [0, 0,                    0, 0,                    0, 0, 0, 0,                    0, 0,                   0, 0],
                     [0, 0,                    0, 0,                    0, 0, 0, 0,                    0, 0,                   0, 0],
                     [0, 0,       9*A*ell*rho/70, 0, -13*A*ell**2*rho/420, 0, 0, 0,      13*A*ell*rho/35, 0, 11*A*ell**2*rho/210, 0],
                     [0, 0,                    0, 0,                    0, 0, 0, 0,                    0, 0,                   0, 0],
                     [0, 0,  13*A*ell**2*rho/420, 0,    -A*ell**3*rho/140, 0, 0, 0,  11*A*ell**2*rho/210, 0,    A*ell**3*rho/105, 0],
                     [0, 0,                    0, 0,                    0, 0, 0, 0,                    0, 0,                   0, 0]],
                    dtype=float))

def S12Elem(self, i):
    ell = self.ell[i]
    rho = self.rho[i]
    A = self.A[i]
    return(np.array([[0, 7*A*ell*rho/20, 0, 0, 0, A*ell**2*rho/20, 0, 3*A*ell*rho/20, 0, 0, 0, -A*ell**2*rho/30],
                     [0,              0, 0, 0, 0,               0, 0,              0, 0, 0, 0,                0],
                     [0,              0, 0, 0, 0,               0, 0,              0, 0, 0, 0,                0],
                     [0,              0, 0, 0, 0,               0, 0,              0, 0, 0, 0,                0],
                     [0,              0, 0, 0, 0,               0, 0,              0, 0, 0, 0,                0],
                     [0,              0, 0, 0, 0,               0, 0,              0, 0, 0, 0,                0],
                     [0, 3*A*ell*rho/20, 0, 0, 0, A*ell**2*rho/30, 0, 7*A*ell*rho/20, 0, 0, 0, -A*ell**2*rho/20],
                     [0,              0, 0, 0, 0,               0, 0,              0, 0, 0, 0,                0],
                     [0,              0, 0, 0, 0,               0, 0,              0, 0, 0, 0,                0],
                     [0,              0, 0, 0, 0,               0, 0,              0, 0, 0, 0,                0],
                     [0,              0, 0, 0, 0,               0, 0,              0, 0, 0, 0,                0],
                     [0,              0, 0, 0, 0,               0, 0,              0, 0, 0, 0,                0]],
                    dtype=float))

def S13Elem(self, i):
    ell = self.ell[i]
    rho = self.rho[i]
    A = self.A[i]
    return(np.array([[0, 0, 7*A*ell*rho/20, 0, -A*ell**2*rho/20, 0, 0, 0, 3*A*ell*rho/20, 0, A*ell**2*rho/30, 0],
                     [0, 0,              0, 0,                0, 0, 0, 0,              0, 0,               0, 0],
                     [0, 0,              0, 0,                0, 0, 0, 0,              0, 0,               0, 0],
                     [0, 0,              0, 0,                0, 0, 0, 0,              0, 0,               0, 0],
                     [0, 0,              0, 0,                0, 0, 0, 0,              0, 0,               0, 0],
                     [0, 0,              0, 0,                0, 0, 0, 0,              0, 0,               0, 0],
                     [0, 0, 3*A*ell*rho/20, 0, -A*ell**2*rho/30, 0, 0, 0, 7*A*ell*rho/20, 0, A*ell**2*rho/20, 0],
                     [0, 0,              0, 0,                0, 0, 0, 0,              0, 0,               0, 0],
                     [0, 0,              0, 0,                0, 0, 0, 0,              0, 0,               0, 0],
                     [0, 0,              0, 0,                0, 0, 0, 0,              0, 0,               0, 0],
                     [0, 0,              0, 0,                0, 0, 0, 0,              0, 0,               0, 0],
                     [0, 0,              0, 0,                0, 0, 0, 0,              0, 0,               0, 0]],
                    dtype=float))

def S23Elem(self, i):
    ell = self.ell[i]
    rho = self.rho[i]
    A = self.A[i]
    return(np.array([[0, 0,                    0, 0,                    0, 0, 0, 0,                    0, 0,                   0, 0],
                     [0, 0,      13*A*ell*rho/35, 0, -11*A*ell**2*rho/210, 0, 0, 0,       9*A*ell*rho/70, 0, 13*A*ell**2*rho/420, 0],
                     [0, 0,                    0, 0,                    0, 0, 0, 0,                    0, 0,                   0, 0],
                     [0, 0,                    0, 0,                    0, 0, 0, 0,                    0, 0,                   0, 0],
                     [0, 0,                    0, 0,                    0, 0, 0, 0,                    0, 0,                   0, 0],
                     [0, 0,  11*A*ell**2*rho/210, 0,    -A*ell**3*rho/105, 0, 0, 0,  13*A*ell**2*rho/420, 0,    A*ell**3*rho/140, 0],
                     [0, 0,                    0, 0,                    0, 0, 0, 0,                    0, 0,                   0, 0],
                     [0, 0,       9*A*ell*rho/70, 0, -13*A*ell**2*rho/420, 0, 0, 0,      13*A*ell*rho/35, 0, 11*A*ell**2*rho/210, 0],
                     [0, 0,                    0, 0,                    0, 0, 0, 0,                    0, 0,                   0, 0],
                     [0, 0,                    0, 0,                    0, 0, 0, 0,                    0, 0,                   0, 0],
                     [0, 0,                    0, 0,                    0, 0, 0, 0,                    0, 0,                   0, 0],
                     [0, 0, -13*A*ell**2*rho/420, 0,     A*ell**3*rho/140, 0, 0, 0, -11*A*ell**2*rho/210, 0,   -A*ell**3*rho/105, 0]],
                    dtype=float))

def FFRF_Output(self):
    if not self.Initialized:
        self.Initialize()
    if (self.stiffMatType[0].lower() == "e" and
        self.massMatType[0].lower() == "c"):
        self.kff = self.Assemble(self.StiffMatElem)
        self.mff = self.Assemble(self.MassMatElem)
        self.Str = self.AssembleOneDirection(self.S__Elem)
        S11 = self.Assemble(self.S11Elem)
        S22 = self.Assemble(self.S22Elem)
        S33 = self.Assemble(self.S33Elem)
        S12 = self.Assemble(self.S12Elem)
        S13 = self.Assemble(self.S13Elem)
        S23 = self.Assemble(self.S23Elem)
        self.Srr = np.block([[S22+S33,  -S12.T,  -S13.T],
                             [   -S12, S33+S11,  -S23.T],
                             [   -S13,    -S23, S22+S11]])
        self.Srf = np.block([[S23-S23.T],
                             [S13.T-S13],
                             [S12-S12.T]])
        self.Sff = S11+S22+S33
        self.Stf = np.block([[ 2*(S22+S33), -(S12+S12.T), -(S13+S13.T)],
                             [-(S12+S12.T),  2*(S33+S11), -(S23+S23.T)],
                             [-(S13+S13.T), -(S23+S23.T),  2*(S22+S11)]])
    else:
        print('Use stiffMatType = "Euler-Bernoulli"\
              \nand massMatType = "consistent"')

def FFRF_OutputSensitivities(self, xDelta=1e-9):
    if (self.stiffMatType[0].lower() == "e" and
        self.massMatType[0].lower() == "c"):
        nx = np.size(self.DesVar)
        self.massNabla = np.zeros([np.size(self.DesVar,)])
        self.kffNabla = np.zeros([self.nNDoF*self.nN, self.nNDoF*self.nN, nx])
        self.mffNabla = np.zeros([self.nNDoF*self.nN, self.nNDoF*self.nN, nx])
        self.StrNabla = np.zeros([self.nNPoC, self.nNDoF*self.nN, nx])
        self.StfNabla = np.zeros([3*self.nNDoF*self.nN, 3*self.nNDoF*self.nN, nx])
        self.SrrNabla = np.zeros([3*self.nNDoF*self.nN, 3*self.nNDoF*self.nN, nx])
        self.SrfNabla = np.zeros([3*self.nNDoF*self.nN, self.nNDoF*self.nN, nx])
        self.SffNabla = np.zeros([self.nNDoF*self.nN, self.nNDoF*self.nN, nx])
        self.r0Nabla = np.zeros([self.nNDoF*self.nN, nx])
        for i in range(nx):
            new = deepcopy(self)
            xPert = xDelta*(1+getattr(new, new.DesVar[i]))
            setattr(new, new.DesVar[i],
                    getattr(new, new.DesVar[i])+xPert)
            new.Initialize()
            new.kff = new.Assemble(new.StiffMatElem)
            new.mff = new.Assemble(new.MassMatElem)
            new.Str = new.AssembleOneDirection(new.S__Elem)
            S11 = new.Assemble(new.S11Elem)
            S22 = new.Assemble(new.S22Elem)
            S33 = new.Assemble(new.S33Elem)
            S12 = new.Assemble(new.S12Elem)
            S13 = new.Assemble(new.S13Elem)
            S23 = new.Assemble(new.S23Elem)
            new.Srr = np.block([[S22+S33,  -S12.T,  -S13.T],
                                [   -S12, S33+S11,  -S23.T],
                                [   -S13,    -S23, S22+S11]])
            new.Srf = np.block([[S23-S23.T],
                                [S13.T-S13],
                                [S12-S12.T]])
            new.Sff = S11+S22+S33
            new.Stf = np.block([[ 2*(S22+S33), -(S12+S12.T), -(S13+S13.T)],
                                [-(S12+S12.T),  2*(S33+S11), -(S23+S23.T)],
                                [-(S13+S13.T), -(S23+S23.T),  2*(S22+S11)]])
            self.massNabla[i] = (new.mass-self.mass)/xPert
            self.kffNabla[:, :, i] = (new.kff-self.kff)/xPert
            self.mffNabla[:, :, i] = (new.mff-self.mff)/xPert
            self.StrNabla[:, :, i] = (new.Str-self.Str)/xPert
            self.StfNabla[:, :, i] = (new.Stf-self.Stf)/xPert
            self.SrrNabla[:, :, i] = (new.Srr-self.Srr)/xPert
            self.SrfNabla[:, :, i] = (new.Srf-self.Srf)/xPert
            self.SffNabla[:, :, i] = (new.Sff-self.Sff)/xPert  # should correspond to mFE: why do they differ?
            self.r0Nabla[:, i] = (new.r0-self.r0)/xPert
    else:
        print('Use stiffMatType = "Euler-Bernoulli"\
              \nand massMatType = "consistent"')


if __name__ == '__main__':

    class Crank(BeamFFRF3D):
        wC = 20     # mm
        hC = 30     # mm
        lC = 120
        nElC = 3
        rhoMod = 7.85e-9
        EMod = 210000
        nuMod = 0.3
        DesVar = ["wC", "hC", "lC", "rhoMod", "EMod"]
        def __init__(self):
            self.plotting = False
            self.Properties = [['Prop1', self.rhoMod, self.EMod, self.nuMod, 'rect', self.hC, self.wC]]
            self.Nodes = []
            self.El = []
            self.PropID = []
            for i in range(self.nElC+1):
                self.Nodes.append([i*self.lC/self.nElC,   0, 0])
            for i in range(self.nElC):
                self.El.append([i+1, i+2])
                self.PropID.append('Prop1')
            self.Disp = []
            self.Load = []

    def Eval(x):
        Model = Crank()
        Model.wC = x[0]
        Model.hC = x[1]
        Model.lC = x[2]
        Model.rhoMod = x[3]
        Model.EMod = x[4]
        Model.FFRF_Output()
        return(Model.mass, Model.kff, Model.mff, Model.Str, Model.Stf, Model.Srr, Model.Srf, Model.Sff, Model.r0)

    x0 = np.array([20, 30, 120, 7.85e-9, 210000])
    xDelta = 1e-8

    mass0, kff0, mff0, Str0, Stf0, Srr0, Srf0, Sff0, r0 = Eval(x0)
    massNabla = np.zeros([len(x0)])
    kffNabla = np.zeros([kff0.shape[0], kff0.shape[1], len(x0)])
    mffNabla = np.zeros([mff0.shape[0], mff0.shape[1], len(x0)])
    StrNabla = np.zeros([Str0.shape[0], Str0.shape[1], len(x0)])
    StfNabla = np.zeros([Stf0.shape[0], Stf0.shape[1], len(x0)])
    SrrNabla = np.zeros([Srr0.shape[0], Srr0.shape[1], len(x0)])
    SrfNabla = np.zeros([Srf0.shape[0], Srf0.shape[1], len(x0)])
    SffNabla = np.zeros([Sff0.shape[0], Sff0.shape[1], len(x0)])
    r0Nabla = np.zeros([r0.shape[0], len(x0)])
    for i in range(len(x0)):
        xPert = (x0[i]+1)*xDelta
        e = np.zeros_like(x0)
        e[i] = 1
        mass1, kff1, mff1, Str1, Stf1, Srr1, Srf1, Sff1, r1 = Eval(x0+e*xPert)
        massNabla[i] = (mass1-mass0)/xPert
        kffNabla[:, :, i] = (kff1-kff0)/xPert
        mffNabla[:, :, i] = (mff1-mff0)/xPert
        StrNabla[:, :, i] = (Str1-Str0)/xPert
        StfNabla[:, :, i] = (Stf1-Stf0)/xPert
        SrrNabla[:, :, i] = (Srr1-Srr0)/xPert
        SrfNabla[:, :, i] = (Srf1-Srf0)/xPert
        SffNabla[:, :, i] = (Sff1-Sff0)/xPert
        r0Nabla[:, i] = (r1-r0)/xPert

    Test = Crank()
    Test.FFRF_Output()
    Test.FFRF_OutputSensitivities()

    # np.set_printoptions(precision=2, suppress=True)
    print(np.max(massNabla-Test.massNabla))
    print(np.max(kffNabla-Test.kffNabla))
    print(np.max(mffNabla-Test.mffNabla))
    print(np.max(StrNabla-Test.StrNabla))
    print(np.max(StfNabla-Test.StfNabla))
    print(np.max(SrrNabla-Test.SrrNabla))
    print(np.max(SrfNabla-Test.SrfNabla))
    print(np.max(SffNabla-Test.SffNabla))
    print(np.max(r0Nabla-Test.r0Nabla))
