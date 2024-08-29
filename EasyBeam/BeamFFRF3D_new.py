import numpy as np
from copy import deepcopy


def SkewMat(v):
    # s = np.zeros([3, 3])
    # s[0, 1] = -v[2]
    # s[1, 0] = v[2]
    # s[0, 2] = v[1]
    # s[2, 0] = -v[1]
    # s[1, 2] = -v[0]
    # s[2, 1] = v[0]
    # return(s)
    return np.array([[    0, -v[2],  v[1]],
                     [ v[2],     0, -v[0]],
                     [-v[1],  v[0],     0]])

def SkewMatList(A):
    # A is a matrix with shape 3xn
    n = A.shape[1]
    ASkew = np.zeros([3, 3*n])
    for i in range(n):
        ASkew[:, 3*i:3*(i+1)] = SkewMat(A[:, i])
    return ASkew

def IψElem(self, i):
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

def IψSElem(self, i):
    ell = self.ell[i]
    rho = self.rho[i]
    A = self.A[i]
    return(np.array([[               0,                0,                0],
                     [               0,                0,      A*ell*rho/2],
                     [               0,     -A*ell*rho/2,                0],
                     [               0,                0,     -A*ell*rho/2],
                     [               0,                0,                0],
                     [     A*ell*rho/2,                0,                0],
                     [               0,      A*ell*rho/2,                0],
                     [    -A*ell*rho/2,                0,                0],
                     [               0,                0,                0],
                     [               0,                0,                0],
                     [               0,                0,                0],
                     [               0,                0,                0],
                     [               0, -A*ell**2*rho/12,                0],
                     [ A*ell**2*rho/12,                0,                0],
                     [               0,                0,                0],
                     [               0,                0, -A*ell**2*rho/12],
                     [               0,                0,                0],
                     [ A*ell**2*rho/12,                0,                0],
                     [               0,                0,                0],
                     [               0,                0,      A*ell*rho/2],
                     [               0,     -A*ell*rho/2,                0],
                     [               0,                0,     -A*ell*rho/2],
                     [               0,                0,                0],
                     [     A*ell*rho/2,                0,                0],
                     [               0,      A*ell*rho/2,                0],
                     [    -A*ell*rho/2,                0,                0],
                     [               0,                0,                0],
                     [               0,                0,                0],
                     [               0,                0,                0],
                     [               0,                0,                0],
                     [               0,  A*ell**2*rho/12,                0],
                     [-A*ell**2*rho/12,                0,                0],
                     [               0,                0,                0],
                     [               0,                0,  A*ell**2*rho/12],
                     [               0,                0,                0],
                     [-A*ell**2*rho/12,                0,                0]],
                    dtype=float).T)

def IuSψElem(self, i):
    ell = self.ell[i]
    rho = self.rho[i]
    A = self.A[i]
    x1 = self.Nodes[self.El[i, 0]-1, 0]
    y1 = self.Nodes[self.El[i, 0]-1, 1]
    z1 = self.Nodes[self.El[i, 0]-1, 2]
    x2 = self.Nodes[self.El[i, 1]-1, 0]
    y2 = self.Nodes[self.El[i, 1]-1, 1]
    z2 = self.Nodes[self.El[i, 1]-1, 2]
    return(np.array([[                                       0,         -A*ell*rho*z1/3 - A*ell*rho*z2/6,          A*ell*rho*y1/3 + A*ell*rho*y2/6],
                     [   7*A*ell*rho*z1/20 + 3*A*ell*rho*z2/20,                                        0,   -7*A*ell*rho*x1/20 - 3*A*ell*rho*x2/20],
                     [  -7*A*ell*rho*y1/20 - 3*A*ell*rho*y2/20,    7*A*ell*rho*x1/20 + 3*A*ell*rho*x2/20,                                        0],
                     [                                       0,                                        0,                                        0],
                     [ A*ell**2*rho*y1/20 + A*ell**2*rho*y2/30, -A*ell**2*rho*x1/20 - A*ell**2*rho*x2/30,                                        0],
                     [ A*ell**2*rho*z1/20 + A*ell**2*rho*z2/30,                                        0, -A*ell**2*rho*x1/20 - A*ell**2*rho*x2/30],
                     [                                       0,         -A*ell*rho*z1/6 - A*ell*rho*z2/3,          A*ell*rho*y1/6 + A*ell*rho*y2/3],
                     [   3*A*ell*rho*z1/20 + 7*A*ell*rho*z2/20,                                        0,   -3*A*ell*rho*x1/20 - 7*A*ell*rho*x2/20],
                     [  -3*A*ell*rho*y1/20 - 7*A*ell*rho*y2/20,    3*A*ell*rho*x1/20 + 7*A*ell*rho*x2/20,                                        0],
                     [                                       0,                                        0,                                        0],
                     [-A*ell**2*rho*y1/30 - A*ell**2*rho*y2/20,  A*ell**2*rho*x1/30 + A*ell**2*rho*x2/20,                                        0],
                     [-A*ell**2*rho*z1/30 - A*ell**2*rho*z2/20,                                        0,  A*ell**2*rho*x1/30 + A*ell**2*rho*x2/20]],
                    dtype=float).T)

def IuSψSElem(self, i):
    ell = self.ell[i]
    rho = self.rho[i]
    A = self.A[i]
    x1 = self.Nodes[self.El[i, 0]-1, 0]
    y1 = self.Nodes[self.El[i, 0]-1, 1]
    z1 = self.Nodes[self.El[i, 0]-1, 2]
    x2 = self.Nodes[self.El[i, 1]-1, 0]
    y2 = self.Nodes[self.El[i, 1]-1, 1]
    z2 = self.Nodes[self.El[i, 1]-1, 2]

    return(np.array([[                                       0,                                       0,                                         0],
                     [        -A*ell*rho*y1/3 - A*ell*rho*y2/6,         A*ell*rho*x1/3 + A*ell*rho*x2/6,                                         0],
                     [        -A*ell*rho*z1/3 - A*ell*rho*z2/6,                                       0,           A*ell*rho*x1/3 + A*ell*rho*x2/6],
                     [   7*A*ell*rho*y1/20 + 3*A*ell*rho*y2/20,  -7*A*ell*rho*x1/20 - 3*A*ell*rho*x2/20,                                         0],
                     [                                       0,                                       0,                                         0],
                     [                                       0,  -7*A*ell*rho*z1/20 - 3*A*ell*rho*z2/20,     7*A*ell*rho*y1/20 + 3*A*ell*rho*y2/20],
                     [   7*A*ell*rho*z1/20 + 3*A*ell*rho*z2/20,                                       0,    -7*A*ell*rho*x1/20 - 3*A*ell*rho*x2/20],
                     [                                       0,   7*A*ell*rho*z1/20 + 3*A*ell*rho*z2/20,    -7*A*ell*rho*y1/20 - 3*A*ell*rho*y2/20],
                     [                                       0,                                        0,                                        0],
                     [                                       0,                                        0,                                        0],
                     [                                       0,                                        0,                                        0],
                     [                                       0,                                        0,                                        0],
                     [-A*ell**2*rho*z1/20 - A*ell**2*rho*z2/30,                                        0,  A*ell**2*rho*x1/20 + A*ell**2*rho*x2/30],
                     [                                       0, -A*ell**2*rho*z1/20 - A*ell**2*rho*z2/30,  A*ell**2*rho*y1/20 + A*ell**2*rho*y2/30],
                     [                                       0,                                        0,                                        0],
                     [ A*ell**2*rho*y1/20 + A*ell**2*rho*y2/30, -A*ell**2*rho*x1/20 - A*ell**2*rho*x2/30,                                        0],
                     [                                       0,                                        0,                                        0],
                     [                                       0, -A*ell**2*rho*z1/20 - A*ell**2*rho*z2/30,  A*ell**2*rho*y1/20 + A*ell**2*rho*y2/30],
                     [                                       0,                                        0,                                        0],
                     [        -A*ell*rho*y1/6 - A*ell*rho*y2/3,          A*ell*rho*x1/6 + A*ell*rho*x2/3,                                        0],
                     [        -A*ell*rho*z1/6 - A*ell*rho*z2/3,                                        0,          A*ell*rho*x1/6 + A*ell*rho*x2/3],
                     [   3*A*ell*rho*y1/20 + 7*A*ell*rho*y2/20,   -3*A*ell*rho*x1/20 - 7*A*ell*rho*x2/20,                                        0],
                     [                                       0,                                        0,                                        0],
                     [                                       0,   -3*A*ell*rho*z1/20 - 7*A*ell*rho*z2/20,    3*A*ell*rho*y1/20 + 7*A*ell*rho*y2/20],
                     [   3*A*ell*rho*z1/20 + 7*A*ell*rho*z2/20,                                        0,   -3*A*ell*rho*x1/20 - 7*A*ell*rho*x2/20],
                     [                                       0,    3*A*ell*rho*z1/20 + 7*A*ell*rho*z2/20,   -3*A*ell*rho*y1/20 - 7*A*ell*rho*y2/20],
                     [                                       0,                                        0,                                        0],
                     [                                       0,                                        0,                                        0],
                     [                                       0,                                        0,                                        0],
                     [                                       0,                                        0,                                        0],
                     [ A*ell**2*rho*z1/30 + A*ell**2*rho*z2/20,                                        0, -A*ell**2*rho*x1/30 - A*ell**2*rho*x2/20],
                     [                                       0,  A*ell**2*rho*z1/30 + A*ell**2*rho*z2/20, -A*ell**2*rho*y1/30 - A*ell**2*rho*y2/20],
                     [                                       0,                                        0,                                        0],
                     [-A*ell**2*rho*y1/30 - A*ell**2*rho*y2/20,  A*ell**2*rho*x1/30 + A*ell**2*rho*x2/20,                                        0],
                     [                                       0,                                        0,                                        0],
                     [                                       0,  A*ell**2*rho*z1/30 + A*ell**2*rho*z2/20, -A*ell**2*rho*y1/30 - A*ell**2*rho*y2/20]],
                    dtype=float).T)


def IψψElem(self, i):
    ell = self.ell[i]
    rho = self.rho[i]
    A = self.A[i]
    return(np.array([[A*ell*rho/3,                    0,                    0, 0,                    0,                   0, A*ell*rho/6,                    0,                    0, 0,                   0,                    0],
                     [          0,      13*A*ell*rho/35,                    0, 0,                    0, 11*A*ell**2*rho/210,           0,       9*A*ell*rho/70,                    0, 0,                   0, -13*A*ell**2*rho/420],
                     [          0,                    0,      13*A*ell*rho/35, 0, -11*A*ell**2*rho/210,                   0,           0,                    0,       9*A*ell*rho/70, 0, 13*A*ell**2*rho/420,                    0],
                     [          0,                    0,                    0, 0,                    0,                   0,           0,                    0,                    0, 0,                   0,                    0],
                     [          0,                    0, -11*A*ell**2*rho/210, 0,     A*ell**3*rho/105,                   0,           0,                    0, -13*A*ell**2*rho/420, 0,   -A*ell**3*rho/140,                    0],
                     [          0,  11*A*ell**2*rho/210,                    0, 0,                    0,    A*ell**3*rho/105,           0,  13*A*ell**2*rho/420,                    0, 0,                   0,    -A*ell**3*rho/140],
                     [A*ell*rho/6,                    0,                    0, 0,                    0,                   0, A*ell*rho/3,                    0,                    0, 0,                   0,                    0],
                     [          0,       9*A*ell*rho/70,                    0, 0,                    0, 13*A*ell**2*rho/420,           0,      13*A*ell*rho/35,                    0, 0,                   0, -11*A*ell**2*rho/210],
                     [          0,                    0,       9*A*ell*rho/70, 0, -13*A*ell**2*rho/420,                   0,           0,                    0,      13*A*ell*rho/35, 0, 11*A*ell**2*rho/210,                    0],
                     [          0,                    0,                    0, 0,                    0,                   0,           0,                    0,                    0, 0,                   0,                    0],
                     [          0,                    0,  13*A*ell**2*rho/420, 0,    -A*ell**3*rho/140,                   0,           0,                    0,  11*A*ell**2*rho/210, 0,    A*ell**3*rho/105,                    0],
                     [          0, -13*A*ell**2*rho/420,                    0, 0,                    0,   -A*ell**3*rho/140,           0, -11*A*ell**2*rho/210,                    0, 0,                   0,     A*ell**3*rho/105]],
                    dtype=float))


def IψSψElem(self, i):
    ell = self.ell[i]
    rho = self.rho[i]
    A = self.A[i]
    return(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 7*A*ell*rho/20, 0, -A*ell**2*rho/20, 0, 0, 0, 3*A*ell*rho/20, 0, A*ell**2*rho/30, 0],
                     [0, -7*A*ell*rho/20, 0, 0, 0, -A*ell**2*rho/20, 0, -3*A*ell*rho/20, 0, 0, 0, A*ell**2*rho/30],
                     [0, 0, -13*A*ell*rho/35, 0, 11*A*ell**2*rho/210, 0, 0, 0, -9*A*ell*rho/70, 0, -13*A*ell**2*rho/420, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [7*A*ell*rho/20, 0, 0, 0, 0, 0, 3*A*ell*rho/20, 0, 0, 0, 0, 0],
                     [0, 13*A*ell*rho/35, 0, 0, 0, 11*A*ell**2*rho/210, 0, 9*A*ell*rho/70, 0, 0, 0, -13*A*ell**2*rho/420],
                     [-7*A*ell*rho/20, 0, 0, 0, 0, 0, -3*A*ell*rho/20, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, -11*A*ell**2*rho/210, 0, 0, 0, -A*ell**3*rho/105, 0, -13*A*ell**2*rho/420, 0, 0, 0, A*ell**3*rho/140],
                     [A*ell**2*rho/20, 0, 0, 0, 0, 0, A*ell**2*rho/30, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, -11*A*ell**2*rho/210, 0, A*ell**3*rho/105, 0, 0, 0, -13*A*ell**2*rho/420, 0, -A*ell**3*rho/140, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [A*ell**2*rho/20, 0, 0, 0, 0, 0, A*ell**2*rho/30, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 3*A*ell*rho/20, 0, -A*ell**2*rho/30, 0, 0, 0, 7*A*ell*rho/20, 0, A*ell**2*rho/20, 0],
                     [0, -3*A*ell*rho/20, 0, 0, 0, -A*ell**2*rho/30, 0, -7*A*ell*rho/20, 0, 0, 0, A*ell**2*rho/20],
                     [0, 0, -9*A*ell*rho/70, 0, 13*A*ell**2*rho/420, 0, 0, 0, -13*A*ell*rho/35, 0, -11*A*ell**2*rho/210, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [3*A*ell*rho/20, 0, 0, 0, 0, 0, 7*A*ell*rho/20, 0, 0, 0, 0, 0],
                     [0, 9*A*ell*rho/70, 0, 0, 0, 13*A*ell**2*rho/420, 0, 13*A*ell*rho/35, 0, 0, 0, -11*A*ell**2*rho/210],
                     [-3*A*ell*rho/20, 0, 0, 0, 0, 0, -7*A*ell*rho/20, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 13*A*ell**2*rho/420, 0, 0, 0, A*ell**3*rho/140, 0, 11*A*ell**2*rho/210, 0, 0, 0, -A*ell**3*rho/105],
                     [-A*ell**2*rho/30, 0, 0, 0, 0, 0, -A*ell**2*rho/20, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 13*A*ell**2*rho/420, 0, -A*ell**3*rho/140, 0, 0, 0, 11*A*ell**2*rho/210, 0, A*ell**3*rho/105, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [-A*ell**2*rho/30, 0, 0, 0, 0, 0, -A*ell**2*rho/20, 0, 0, 0, 0, 0]],
                    dtype=float))

def IψSψSElem(self, i):
    ell = self.ell[i]
    rho = self.rho[i]
    A = self.A[i]
    return(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, A*ell*rho/3, 0, -7*A*ell*rho/20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -A*ell**2*rho/20, 0, 0, 0, A*ell*rho/6, 0, -3*A*ell*rho/20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, A*ell**2*rho/30, 0, 0],
                     [0, 0, A*ell*rho/3, 0, 0, 0, -7*A*ell*rho/20, 0, 0, 0, 0, 0, A*ell**2*rho/20, 0, 0, 0, 0, 0, 0, 0, A*ell*rho/6, 0, 0, 0, -3*A*ell*rho/20, 0, 0, 0, 0, 0, -A*ell**2*rho/30, 0, 0, 0, 0, 0],
                     [0, -7*A*ell*rho/20, 0, 13*A*ell*rho/35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11*A*ell**2*rho/210, 0, 0, 0, -3*A*ell*rho/20, 0, 9*A*ell*rho/70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -13*A*ell**2*rho/420, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 13*A*ell*rho/35, 0, -13*A*ell*rho/35, 0, 0, 0, 0, 0, 11*A*ell**2*rho/210, 0, 0, 0, 11*A*ell**2*rho/210, 0, 0, 0, 0, 0, 9*A*ell*rho/70, 0, -9*A*ell*rho/70, 0, 0, 0, 0, 0, -13*A*ell**2*rho/420, 0, 0, 0, -13*A*ell**2*rho/420],
                     [0, 0, -7*A*ell*rho/20, 0, 0, 0, 13*A*ell*rho/35, 0, 0, 0, 0, 0, -11*A*ell**2*rho/210, 0, 0, 0, 0, 0, 0, 0, -3*A*ell*rho/20, 0, 0, 0, 9*A*ell*rho/70, 0, 0, 0, 0, 0, 13*A*ell**2*rho/420, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, -13*A*ell*rho/35, 0, 13*A*ell*rho/35, 0, 0, 0, 0, 0, -11*A*ell**2*rho/210, 0, 0, 0, -11*A*ell**2*rho/210, 0, 0, 0, 0, 0, -9*A*ell*rho/70, 0, 9*A*ell*rho/70, 0, 0, 0, 0, 0, 13*A*ell**2*rho/420, 0, 0, 0, 13*A*ell**2*rho/420],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, A*ell**2*rho/20, 0, 0, 0, -11*A*ell**2*rho/210, 0, 0, 0, 0, 0, A*ell**3*rho/105, 0, 0, 0, 0, 0, 0, 0, A*ell**2*rho/30, 0, 0, 0, -13*A*ell**2*rho/420, 0, 0, 0, 0, 0, -A*ell**3*rho/140, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 11*A*ell**2*rho/210, 0, -11*A*ell**2*rho/210, 0, 0, 0, 0, 0, A*ell**3*rho/105, 0, 0, 0, A*ell**3*rho/105, 0, 0, 0, 0, 0, 13*A*ell**2*rho/420, 0, -13*A*ell**2*rho/420, 0, 0, 0, 0, 0, -A*ell**3*rho/140, 0, 0, 0, -A*ell**3*rho/140],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, -A*ell**2*rho/20, 0, 11*A*ell**2*rho/210, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, A*ell**3*rho/105, 0, 0, 0, -A*ell**2*rho/30, 0, 13*A*ell**2*rho/420, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -A*ell**3*rho/140, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 11*A*ell**2*rho/210, 0, -11*A*ell**2*rho/210, 0, 0, 0, 0, 0, A*ell**3*rho/105, 0, 0, 0, A*ell**3*rho/105, 0, 0, 0, 0, 0, 13*A*ell**2*rho/420, 0, -13*A*ell**2*rho/420, 0, 0, 0, 0, 0, -A*ell**3*rho/140, 0, 0, 0, -A*ell**3*rho/140],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, A*ell*rho/6, 0, -3*A*ell*rho/20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -A*ell**2*rho/30, 0, 0, 0, A*ell*rho/3, 0, -7*A*ell*rho/20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, A*ell**2*rho/20, 0, 0],
                     [0, 0, A*ell*rho/6, 0, 0, 0, -3*A*ell*rho/20, 0, 0, 0, 0, 0, A*ell**2*rho/30, 0, 0, 0, 0, 0, 0, 0, A*ell*rho/3, 0, 0, 0, -7*A*ell*rho/20, 0, 0, 0, 0, 0, -A*ell**2*rho/20, 0, 0, 0, 0, 0],
                     [0, -3*A*ell*rho/20, 0, 9*A*ell*rho/70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13*A*ell**2*rho/420, 0, 0, 0, -7*A*ell*rho/20, 0, 13*A*ell*rho/35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -11*A*ell**2*rho/210, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 9*A*ell*rho/70, 0, -9*A*ell*rho/70, 0, 0, 0, 0, 0, 13*A*ell**2*rho/420, 0, 0, 0, 13*A*ell**2*rho/420, 0, 0, 0, 0, 0, 13*A*ell*rho/35, 0, -13*A*ell*rho/35, 0, 0, 0, 0, 0, -11*A*ell**2*rho/210, 0, 0, 0, -11*A*ell**2*rho/210],
                     [0, 0, -3*A*ell*rho/20, 0, 0, 0, 9*A*ell*rho/70, 0, 0, 0, 0, 0, -13*A*ell**2*rho/420, 0, 0, 0, 0, 0, 0, 0, -7*A*ell*rho/20, 0, 0, 0, 13*A*ell*rho/35, 0, 0, 0, 0, 0, 11*A*ell**2*rho/210, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, -9*A*ell*rho/70, 0, 9*A*ell*rho/70, 0, 0, 0, 0, 0, -13*A*ell**2*rho/420, 0, 0, 0, -13*A*ell**2*rho/420, 0, 0, 0, 0, 0, -13*A*ell*rho/35, 0, 13*A*ell*rho/35, 0, 0, 0, 0, 0, 11*A*ell**2*rho/210, 0, 0, 0, 11*A*ell**2*rho/210],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, -A*ell**2*rho/30, 0, 0, 0, 13*A*ell**2*rho/420, 0, 0, 0, 0, 0, -A*ell**3*rho/140, 0, 0, 0, 0, 0, 0, 0, -A*ell**2*rho/20, 0, 0, 0, 11*A*ell**2*rho/210, 0, 0, 0, 0, 0, A*ell**3*rho/105, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, -13*A*ell**2*rho/420, 0, 13*A*ell**2*rho/420, 0, 0, 0, 0, 0, -A*ell**3*rho/140, 0, 0, 0, -A*ell**3*rho/140, 0, 0, 0, 0, 0, -11*A*ell**2*rho/210, 0, 11*A*ell**2*rho/210, 0, 0, 0, 0, 0, A*ell**3*rho/105, 0, 0, 0, A*ell**3*rho/105],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, A*ell**2*rho/30, 0, -13*A*ell**2*rho/420, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -A*ell**3*rho/140, 0, 0, 0, A*ell**2*rho/20, 0, -11*A*ell**2*rho/210, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, A*ell**3*rho/105, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, -13*A*ell**2*rho/420, 0, 13*A*ell**2*rho/420, 0, 0, 0, 0, 0, -A*ell**3*rho/140, 0, 0, 0, -A*ell**3*rho/140, 0, 0, 0, 0, 0, -11*A*ell**2*rho/210, 0, 11*A*ell**2*rho/210, 0, 0, 0, 0, 0, A*ell**3*rho/105, 0, 0, 0, A*ell**3*rho/105]],
                    dtype=float))

def Bt(self, j):
    B = np.zeros([self.nNPoC, self.nN*self.nNDoF])
    B[:, j*self.nNDoF: j*self.nNDoF+self.nNPoC] = np.eye(3)
    return(B)

def Br(self, j):
    B = np.zeros([self.nNPoC, self.nN*self.nNDoF])
    B[:, j*self.nNDoF+self.nNPoC:j*self.nNDoF+self.nNDoF] = np.eye(3)
    return(B)

def AssembleOneDirectionSkew(self, MatElem):  # needed for FFRF
    # attention: Check how the assembling must be done when the elements are not aligned with the body frame!
    Matrix = np.zeros([self.nNPoC, self.nNDoF*self.nN*self.nNPoC])
    for i in range(self.nEl):
        Matrix[:, self.idxSkew[i]] += self.TX[i]@MatElem(i)@np.kron(self.T[i], np.identity(3))
    return Matrix

def AssembleSkew(self, MatElem):
    Matrix = np.zeros([self.nNDoF*self.nN*self.nNPoC, self.nNDoF*self.nN])
    for i in range(self.nEl):
        Matrix[np.ix_(self.idxSkew[i], self.idx[i])] += np.kron(self.T[i], np.identity(3)).T@MatElem(i)@self.T[i]
    return Matrix

def AssembleSkewSkew(self, MatElem):
    Matrix = np.zeros([self.nNDoF*self.nN*self.nNPoC, self.nNDoF*self.nN*self.nNPoC])
    for i in range(self.nEl):
        Matrix[np.ix_(self.idxSkew[i], self.idxSkew[i])] += np.kron(self.T[i], np.identity(3)).T@MatElem(i)@np.kron(self.T[i], np.identity(3))
    return Matrix

def FFRF_Output(self):
    if not self.Initialized:
        self.Initialize()

    if (self.stiffMatType[0].lower() == "e" and
        self.massMatType[0].lower() == "c"):

        self.idxSkew = [[]]*self.nEl
        # self.mass = 0
        self.Xo = np.zeros(3)
        self.Θo = np.zeros([3, 3])
        for i in range(self.nEl):
            ell = self.ell[i]
            rho = self.rho[i]
            A = self.A[i]
            x1 = self.Nodes[self.El[i, 0]-1, 0]
            y1 = self.Nodes[self.El[i, 0]-1, 1]
            z1 = self.Nodes[self.El[i, 0]-1, 2]
            x2 = self.Nodes[self.El[i, 1]-1, 0]
            y2 = self.Nodes[self.El[i, 1]-1, 1]
            z2 = self.Nodes[self.El[i, 1]-1, 2]
            self.idxSkew[i] = np.r_[self.nNDoF*(self.El[i, 0]-1)*self.nNPoC:(self.nNDoF*(self.El[i, 0]-1)+self.nNDoF)*self.nNPoC,
                                    self.nNDoF*(self.El[i, 1]-1)*self.nNPoC:(self.nNDoF*(self.El[i, 1]-1)+self.nNDoF)*self.nNPoC].tolist()
            # self.mass += (A*ell*rho)
            self.Xo += A*ell*rho*np.array([(x1+x2)/2, (y1+y2)/2, (z1+z2)/2])
            self.Θo += A*ell*rho*np.array([[y1**2/3 + y1*y2/3 + y2**2/3 + z1**2/3 + z1*z2/3 + z2**2/3,
                                            -x1*y1/3 - x1*y2/6 - x2*y1/6 - x2*y2/3,
                                            -x1*z1/3 - x1*z2/6 - x2*z1/6 - x2*z2/3],
                                           [-x1*y1/3 - x1*y2/6 - x2*y1/6 - x2*y2/3,
                                            x1**2/3 + x1*x2/3 + x2**2/3 + z1**2/3 + z1*z2/3 + z2**2/3,
                                            -y1*z1/3 - y1*z2/6 - y2*z1/6 - y2*z2/3],
                                           [-x1*z1/3 - x1*z2/6 - x2*z1/6 - x2*z2/3,
                                            -y1*z1/3 - y1*z2/6 - y2*z1/6 - y2*z2/3,
                                            x1**2/3 + x1*x2/3 + x2**2/3 + y1**2/3 + y1*y2/3 + y2**2/3]])
        self.Xo /= self.mass
        self.kff = self.Assemble(self.StiffMatElem)
        self.mff = self.Assemble(self.MassMatElem)
        self.Iψ = self.AssembleOneDirection(self.IψElem)
        # self.IψS = SkewMatList(self.Iψ)
        self.IψS = self.AssembleOneDirectionSkew(self.IψSElem)
        self.IuSψ = self.AssembleOneDirection(self.IuSψElem)
        self.IuSψr = np.zeros([self.nNPoC, self.nN*self.nNDoF])
        self.IuSψS = self.AssembleOneDirectionSkew(self.IuSψSElem)
        self.Iψψ = self.Assemble(self.IψψElem)
        self.IψSψ = self.AssembleSkew(self.IψSψElem)
        self.IψSψS = self.AssembleSkewSkew(self.IψSψSElem)

    elif (self.stiffMatType[0].lower() == "e" and
          self.massMatType[0].lower() == "l"):

        self.kff = self.Assemble(self.StiffMatElem)
        self.mff = self.Assemble(self.MassMatElem)

        self.mN = [[]]*self.nN
        self.ΘoN = [[]]*self.nN
        self.uoN = [[]]*self.nN
        # self.ΨtN = [[]]*self.nN
        # self.ΨrN = [[]]*self.nN

        # self.mass = 0
        self.Xo = np.zeros(self.nNPoC)
        self.Θo = np.zeros([self.nNPoC, self.nNPoC])
        self.Iψ = np.zeros([self.nNPoC, self.nN*self.nNDoF])
        self.IψS = np.zeros([self.nNPoC, self.nN*self.nNDoF*self.nNPoC])
        self.IuSψ = np.zeros([self.nNPoC, self.nN*self.nNDoF])
        self.IuSψr = np.zeros([self.nNPoC, self.nN*self.nNDoF])
        self.IuSψS = np.zeros([self.nNPoC, self.nN*self.nNDoF*self.nNPoC])
        self.Iψψ = np.zeros([self.nN*self.nNDoF, self.nN*self.nNDoF])
        self.IψSψ = np.zeros([self.nN*self.nNDoF*self.nNPoC, self.nN*self.nNDoF])
        self.IψSψS = np.zeros([self.nN*self.nNDoF*self.nNPoC, self.nN*self.nNDoF*self.nNPoC])

        for j in range(self.nN):

            self.mN[j] = self.mff[j*self.nNDoF, j*self.nNDoF]
            self.ΘoN[j] = self.mff[j*self.nNDoF+self.nNPoC:j*self.nNDoF+self.nNDoF, j*self.nNDoF+self.nNPoC:j*self.nNDoF+self.nNDoF]
            # self.ΨtN[j] = self.Bt(j)
            # self.ΨrN[j] = self.Br(j)

            # self.mass += self.mN[j]
            self.Xo += self.mN[j]*self.Nodes[j, :]
            self.Θo += self.mN[j]*SkewMat(self.Nodes[j, :]).T@SkewMat(self.Nodes[j, :])+self.ΘoN[j]

            # self.Iψ += self.mN[j]*self.ΨtN[j]
            self.Iψ[:, j*self.nNDoF: j*self.nNDoF+self.nNPoC] += self.mN[j]*np.eye(3)
            # self.IψS += self.mN[j]*SkewMatList(self.ΨtN[j])
            self.IψS[:, j*self.nNDoF*self.nNPoC: j*self.nNDoF*self.nNPoC+self.nNPoC*self.nNPoC] += \
                self.mN[j]*SkewMatList(np.eye(3))

            # self.IuSψ += self.mN[j]*SkewMat(self.Nodes[j, :]).T@self.ΨtN[j]+self.ΘoN[j]@self.ΨrN[j]
            self.IuSψ[:, j*self.nNDoF: j*self.nNDoF+self.nNPoC] += self.mN[j]*SkewMat(self.Nodes[j, :]).T
            self.IuSψ[:, j*self.nNDoF+self.nNPoC:j*self.nNDoF+self.nNDoF] += self.ΘoN[j]
            self.IuSψr[:, j*self.nNDoF+self.nNPoC:j*self.nNDoF+self.nNDoF] += self.ΘoN[j]
            # self.IuSψS += self.mN[j]*SkewMat(self.Nodes[j, :]).T@SkewMatList(self.ΨtN[j])
            self.IuSψS[:, j*self.nNDoF*self.nNPoC: j*self.nNDoF*self.nNPoC+self.nNPoC*self.nNPoC] += \
                self.mN[j]*SkewMat(self.Nodes[j, :]).T@SkewMatList(np.eye(3))

            # self.Iψψ += self.mN[j]*self.ΨtN[j].T@self.ΨtN[j]+self.ΨrN[j].T@self.ΘoN[j]@self.ΨrN[j]
            self.Iψψ[j*self.nNDoF: j*self.nNDoF+self.nNPoC, j*self.nNDoF: j*self.nNDoF+self.nNPoC] += \
                self.mN[j]*np.eye(3).T@np.eye(3)
            self.Iψψ[j*self.nNDoF+self.nNPoC:j*self.nNDoF+self.nNDoF, j*self.nNDoF+self.nNPoC:j*self.nNDoF+self.nNDoF] += \
                np.eye(3).T@self.ΘoN[j]@np.eye(3)
            # self.IψSψ += self.mN[j]*SkewMatList(self.ΨtN[j]).T@self.ΨtN[j]
            self.IψSψ[j*self.nNDoF*self.nNPoC: (j*self.nNDoF+self.nNPoC)*self.nNPoC, j*self.nNDoF: j*self.nNDoF+self.nNPoC] += \
                self.mN[j]*SkewMatList(np.eye(3)).T@np.eye(3)
            # self.IψSψS += self.mN[j]*SkewMatList(self.ΨtN[j]).T@SkewMatList(self.ΨtN[j])
            self.IψSψS[j*self.nNDoF*self.nNPoC: (j*self.nNDoF+self.nNPoC)*self.nNPoC, j*self.nNDoF*self.nNPoC: (j*self.nNDoF+self.nNPoC)*self.nNPoC] += \
                self.mN[j]*SkewMatList(np.eye(3)).T@SkewMatList(np.eye(3))
        self.Xo /= self.mass

    else:
        print('Use stiffMatType = "Euler-Bernoulli"\
              \nand massMatType = "consistent" or "lumped"')

    # # the following is for testing in lumped mass paper:
    # h = self.Properties[0][5]
    # w = self.Properties[0][6]
    # l = np.sum(self.ell)
    # self.Θo = np.diag([1/12*self.mass*(w**2+h**2),
    #                    1/12*self.mass*(l**2+h**2)+self.mass*(l/2)**2,
    #                    1/12*self.mass*(l**2+w**2)+self.mass*(l/2)**2])

def FFRF_OutputSensitivities(self, xDelta=1e-9):
    if ((self.stiffMatType[0].lower() == "e" and self.massMatType[0].lower() == "c") or 
        (self.stiffMatType[0].lower() == "e" and self.massMatType[0].lower() == "l")):
        nx = np.size(self.DesVar)

        self.kffNabla = np.zeros([self.nNDoF*self.nN, self.nNDoF*self.nN, nx])
        self.mffNabla = np.zeros([self.nNDoF*self.nN, self.nNDoF*self.nN, nx])

        self.massNabla = np.zeros([nx])
        self.XoNabla = np.zeros([self.nNPoC, nx])
        self.ΘoNabla = np.zeros([self.nNPoC, self.nNPoC, nx])
        self.IψNabla = np.zeros([self.nNPoC, self.nN*self.nNDoF, nx])
        self.IψSNabla = np.zeros([self.nNPoC, self.nN*self.nNDoF*self.nNPoC, nx])
        self.IuSψNabla = np.zeros([self.nNPoC, self.nN*self.nNDoF, nx])
        self.IuSψrNabla = np.zeros([self.nNPoC, self.nN*self.nNDoF, nx])
        self.IuSψSNabla = np.zeros([self.nNPoC, self.nN*self.nNDoF*self.nNPoC, nx])
        self.IψψNabla = np.zeros([self.nN*self.nNDoF, self.nN*self.nNDoF, nx])
        self.IψSψNabla = np.zeros([self.nN*self.nNDoF*self.nNPoC, self.nN*self.nNDoF, nx])
        self.IψSψSNabla = np.zeros([self.nN*self.nNDoF*self.nNPoC, self.nN*self.nNDoF*self.nNPoC, nx])
        self.r0Nabla = np.zeros([self.nNDoF*self.nN, nx])

        for i in range(nx):
            new = deepcopy(self)
            xPert = xDelta*(1+getattr(new, new.DesVar[i]))
            setattr(new, new.DesVar[i],
                    getattr(new, new.DesVar[i])+xPert)
            new.Initialize()
            new.FFRF_Output()

            self.kffNabla[:, :, i] = (new.kff-self.kff)/xPert
            self.mffNabla[:, :, i] = (new.mff-self.mff)/xPert

            self.massNabla[i] = (new.mass-self.mass)/xPert
            self.XoNabla[:, i] = (new.Xo-self.Xo)/xPert
            self.ΘoNabla[:, :, i] = (new.Θo-self.Θo)/xPert

            self.IψNabla[:, :, i] = (new.Iψ-self.Iψ)/xPert
            self.IψSNabla[:, :, i] = (new.IψS-self.IψS)/xPert

            self.IuSψNabla[:, :, i] = (new.IuSψ-self.IuSψ)/xPert
            self.IuSψrNabla[:, :, i] = (new.IuSψr-self.IuSψr)/xPert
            self.IuSψSNabla[:, :, i] = (new.IuSψS-self.IuSψS)/xPert

            self.IψψNabla[:, :, i] = (new.Iψψ-self.Iψψ)/xPert
            self.IψSψNabla[:, :, i] = (new.IψSψ-self.IψSψ)/xPert
            self.IψSψSNabla[:, :, i] = (new.IψSψS-self.IψSψS)/xPert

            self.r0Nabla[:, i] = (new.r0-self.r0)/xPert

    else:
        print('Use stiffMatType = "Euler-Bernoulli"\
              \nand massMatType = "consistent" or "lumped"')

if __name__ == '__main__':
    from EasyBeam import BeamFFRF3D_NEW
    class Crank(BeamFFRF3D_NEW):
        wC = 20     # mm
        hC = 30     # mm
        lC = 120
        nElC = 3
        rhoMod = 7.85e-9
        EMod = 210000
        nuMod = 0.3
        DesVar = ["wC", "hC", "lC", "rhoMod", "EMod"]
        massMatType = "lumped"  # "consistent", "lumped"
        def __init__(self):
            self.plotting = False
            self.Properties = [['Prop1', self.rhoMod, self.EMod, self.nuMod, 'rect', self.hC, self.wC]]
            self.Nodes = []
            self.El = []
            self.PropID = []
            for i in range(self.nElC+1):
                self.Nodes.append([i*self.lC/self.nElC, 0, 0])
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
        return(Model.mass, Model.kff, Model.mff, Model.Xo, Model.Θo, Model.Iψ,
               Model.IψS, Model.IuSψ, Model.IuSψS, Model.Iψψ, Model.IψSψ, Model.IψSψS)


    x0 = np.array([20, 30, 120, 7.85e-9, 210000])
    xDelta = 1e-8

    mass0, kff0, mff0, Xo0, Θo0, Iψ0, IψS0, IuSψ0, IuSψS0, Iψψ0, IψSψ0, IψSψS0 = Eval(x0)

    kffNabla = np.zeros([kff0.shape[0], kff0.shape[1], len(x0)])
    mffNabla = np.zeros([mff0.shape[0], mff0.shape[1], len(x0)])

    massNabla = np.zeros([len(x0)])
    XoNabla = np.zeros([Xo0.shape[0], len(x0)])
    ΘoNabla = np.zeros([Θo0.shape[0], Θo0.shape[1], len(x0)])
    IψNabla = np.zeros([Iψ0.shape[0], Iψ0.shape[1], len(x0)])
    IψSNabla = np.zeros([IψS0.shape[0], IψS0.shape[1], len(x0)])
    IuSψNabla = np.zeros([IuSψ0.shape[0], IuSψ0.shape[1], len(x0)])
    IuSψSNabla = np.zeros([IuSψS0.shape[0], IuSψS0.shape[1], len(x0)])
    IψψNabla = np.zeros([Iψψ0.shape[0], Iψψ0.shape[1], len(x0)])
    IψSψNabla = np.zeros([IψSψ0.shape[0], IψSψ0.shape[1], len(x0)])
    IψSψSNabla = np.zeros([IψSψS0.shape[0], IψSψS0.shape[1], len(x0)])

    for i in range(len(x0)):
        xPert = (x0[i]+1)*xDelta
        e = np.zeros_like(x0)
        e[i] = 1
        mass1, kff1, mff1, Xo1, Θo1, Iψ1, IψS1, IuSψ1, IuSψS1, Iψψ1, IψSψ1, IψSψS1 = Eval(x0+e*xPert)
        massNabla[i] = (mass1-mass0)/xPert
        kffNabla[:, :, i] = (kff1-kff0)/xPert
        mffNabla[:, :, i] = (mff1-mff0)/xPert
        XoNabla[:, i] = (Xo1-Xo0)/xPert
        ΘoNabla[:, :, i] = (Θo1-Θo0)/xPert
        IψNabla[:, :, i] = (Iψ1-Iψ0)/xPert
        IψSNabla[:, :, i] = (IψS1-IψS0)/xPert
        IuSψNabla[:, :, i] = (IuSψ1-IuSψ0)/xPert
        IuSψSNabla[:, :, i] = (IuSψS1-IuSψS0)/xPert
        IψψNabla[:, :, i] = (Iψψ1-Iψψ0)/xPert
        IψSψNabla[:, :, i] = (IψSψ1-IψSψ0)/xPert
        IψSψSNabla[:, :, i] = (IψSψS1-IψSψS0)/xPert

    Test = Crank()
    Test.FFRF_Output()
    Test.FFRF_OutputSensitivities()

    # np.set_printoptions(precision=2, suppress=True)
    print(np.max(massNabla-Test.massNabla))
    print(np.max(kffNabla-Test.kffNabla))
    print(np.max(mffNabla-Test.mffNabla))
    print(np.max(XoNabla-Test.XoNabla))
    print(np.max(ΘoNabla-Test.ΘoNabla))
    print(np.max(IψNabla-Test.IψNabla))
    print(np.max(IψSNabla-Test.IψSNabla))
    print(np.max(IuSψNabla-Test.IuSψNabla))
    print(np.max(IuSψSNabla-Test.IuSψSNabla))
    print(np.max(IψψNabla-Test.IψψNabla))
    print(np.max(IψSψNabla-Test.IψSψNabla))
    print(np.max(IψSψSNabla-Test.IψSψSNabla))
