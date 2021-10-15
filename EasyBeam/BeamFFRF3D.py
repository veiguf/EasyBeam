import numpy as np


def NMat(self, i, ξ):
    NMat = self.TX[i]@self.ShapeMat(ξ, self.ell[i])@self.T[i]@self.L[i]
    return NMat

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
    if (self.stiffMatType[0].lower() == "e" and
        self.massMatType[0].lower() == "c"):
        self.kff = self.Assemble(self.StiffMatElem)
        self.mff = self.Assemble(self.MassMatElem)
        self.S__ = self.AssembleOneDirection(self.S__Elem)
        self.S11 = self.Assemble(self.S11Elem)
        self.S22 = self.Assemble(self.S22Elem)
        self.S33 = self.Assemble(self.S33Elem)
        self.S12 = self.Assemble(self.S12Elem)
        self.S13 = self.Assemble(self.S13Elem)
        self.S23 = self.Assemble(self.S23Elem)
    else:
        print('Use stiffMatType = "Euler-Bernoulli"\
              \nand massMatType = "consistent"')
