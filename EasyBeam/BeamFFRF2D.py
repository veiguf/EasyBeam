import numpy as np
from copy import deepcopy

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

def FFRF_OutputSensitivities(self, xDelta=1e-6):
    if (self.stiffMatType[0].lower() == "e" and
        self.massMatType[0].lower() == "c"):

        nx = np.size(self.SizingVariables)
        self.massNabla = np.zeros([np.size(self.SizingVariables,)])
        self.kffNabla = np.zeros([self.nNDoF*self.nN, self.nNDoF*self.nN, nx])
        self.StfNabla = np.zeros([self.nNPoC, self.nNDoF*self.nN, nx])
        self.SrfNabla = np.zeros([self.nNDoF*self.nN, self.nNDoF*self.nN, nx])
        self.SffNabla = np.zeros([self.nNDoF*self.nN, self.nNDoF*self.nN, nx])
        for i in range(nx):
            new = deepcopy(self)
            xPert = xDelta*(1+getattr(new, new.SizingVariables[i]))
            setattr(new, new.SizingVariables[i],
                    getattr(new, new.SizingVariables[i])+xPert)
            new.__init__()
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
    else:
        print('Use stiffMatType = "Euler-Bernoulli"\
              \nand massMatType = "consistent"')
