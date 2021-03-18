import numpy as np

def NMat(self, i, ξ):
    NMat = self.T2[i]@self.ShapeMat(ξ, self.ell[i])@self.T[i]@self.L[i]
    return NMat

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
