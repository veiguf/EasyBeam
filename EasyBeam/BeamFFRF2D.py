import numpy as np
from copy import deepcopy
from EasyBeam import BeamFFRF2D

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

def FFRF_Output(self):
    if not self.Initialized:
        self.Initialize()
    if (self.stiffMatType[0].lower() == "e" and
        self.massMatType[0].lower() == "c"):

        self.kff = self.Assemble(self.StiffMatElem)
        self.Stf = self.AssembleOneDirection(self.StfElem)
        self.Srf = self.Assemble(self.SrfElem)
        self.Sff = self.Assemble(self.MassMatElem)
    else:
        print('Use stiffMatType = "Euler-Bernoulli"\
              \nand massMatType = "consistent"')

def FFRF_OutputSensitivities(self, xDelta=1e-9):
    if (self.stiffMatType[0].lower() == "e" and
        self.massMatType[0].lower() == "c"):

        nx = np.size(self.DesVar)
        self.massNabla = np.zeros([np.size(self.DesVar,)])
        self.kffNabla = np.zeros([self.nNDoF*self.nN, self.nNDoF*self.nN, nx])
        self.StfNabla = np.zeros([self.nNPoC, self.nNDoF*self.nN, nx])
        self.SrfNabla = np.zeros([self.nNDoF*self.nN, self.nNDoF*self.nN, nx])
        self.SffNabla = np.zeros([self.nNDoF*self.nN, self.nNDoF*self.nN, nx])
        self.r0Nabla = np.zeros([self.nNDoF*self.nN, nx])
        for i in range(nx):
            new = deepcopy(self)
            xPert = xDelta*(1+getattr(new, new.DesVar[i]))
            setattr(new, new.DesVar[i],
                    getattr(new, new.DesVar[i])+xPert)
            new.Initialize()
            new.kff = new.Assemble(new.StiffMatElem)
            new.Stf = new.AssembleOneDirection(new.StfElem)
            new.Srf = new.Assemble(new.SrfElem)
            new.Sff = new.Assemble(new.MassMatElem)
            self.massNabla[i] = (new.mass-self.mass)/xPert
            self.kffNabla[:, :, i] = (new.kff-self.kff)/xPert
            self.StfNabla[:, :, i] = (new.Stf-self.Stf)/xPert
            self.SrfNabla[:, :, i] = (new.Srf-self.Srf)/xPert
            self.SffNabla[:, :, i] = (new.Sff-self.Sff)/xPert
            self.r0Nabla[:, i] = (new.r0-self.r0)/xPert
    else:
        print('Use stiffMatType = "Euler-Bernoulli"\
              \nand massMatType = "consistent"')

if __name__ == '__main__':

    class Crank(BeamFFRF2D):
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
                self.Nodes.append([i*self.lC/self.nElC,   0])
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
        return(Model.mass, Model.kff, Model.Stf, Model.Srf, Model.Sff, Model.r0)

    x0 = np.array([20, 30, 120, 7.85e-9, 210000])
    xDelta = 1e-8

    mass0, kff0, Stf0, Srf0, Sff0, r0 = Eval(x0)
    massNabla = np.zeros([len(x0)])
    kffNabla = np.zeros([kff0.shape[0], kff0.shape[1], len(x0)])
    StfNabla = np.zeros([Stf0.shape[0], Stf0.shape[1], len(x0)])
    SrfNabla = np.zeros([Srf0.shape[0], Srf0.shape[1], len(x0)])
    SffNabla = np.zeros([Sff0.shape[0], Sff0.shape[1], len(x0)])
    r0Nabla = np.zeros([r0.shape[0], len(x0)])
    for i in range(len(x0)):
        xPert = (x0[i]+1)*xDelta
        e = np.zeros_like(x0)
        e[i] = 1
        mass1, kff1, Stf1, Srf1, Sff1, r1 = Eval(x0+e*xPert)
        massNabla[i] = (mass1-mass0)/xPert
        kffNabla[:, :, i] = (kff1-kff0)/xPert
        StfNabla[:, :, i] = (Stf1-Stf0)/xPert
        SrfNabla[:, :, i] = (Srf1-Srf0)/xPert
        SffNabla[:, :, i] = (Sff1-Sff0)/xPert
        r0Nabla[:, i] = (r1-r0)/xPert

    Test = Crank()
    Test.FFRF_Output()
    Test.FFRF_OutputSensitivities()

    # np.set_printoptions(precision=2, suppress=True)
    print(np.max(massNabla-Test.massNabla))
    print(np.max(kffNabla-Test.kffNabla))
    print(np.max(StfNabla-Test.StfNabla))
    print(np.max(SrfNabla-Test.SrfNabla))
    print(np.max(SffNabla-Test.SffNabla))
    print(np.max(r0Nabla-Test.r0Nabla))
