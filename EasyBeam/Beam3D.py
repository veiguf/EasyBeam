import numpy as np
from numpy import sqrt

# def ShapeMat(self, ξ, ell):
#     # from: https://github.com/airinnova/framat/blob/8ebbc85d048484c339703ae107df6247e43990b6/src/framat/_element.py
#     N1 = 1 - ξ
#     N2 = ξ
#     N3 = 1 - 3*ξ**2 + 2*ξ**3
#     N4 = 3*ξ**2 - 2*ξ**3
#     N5 = ell*(ξ - 2*ξ**2 + ξ**3)
#     N6 = ell*(-ξ**2 + ξ**3)
#     M1 = 1 - ξ
#     M2 = ξ
#     M3 = -(6/ell)*(ξ - ξ**2)
#     M4 = (6/ell)*(ξ - ξ**2)
#     M5 = 1 - 4*ξ + 3*ξ**2
#     M6 = -2*ξ + 3*ξ**2
#     N = np.zeros((6, 12))
#     N[0, 0] = N1
#     N[0, 6] = N2
#     N[1, 1] = N3
#     N[1, 5] = N5
#     N[1, 7] = N4
#     N[1, 11] = N6
#     N[2, 2] = N3
#     N[2, 4] = -N5
#     N[2, 8] = N4
#     N[2, 10] = -N6
#     N[3, 3] = M1
#     N[3, 9] = M2
#     N[4, 2] = M3
#     N[4, 4] = M5
#     N[4, 8] = M4
#     N[4, 10] = M6
#     N[5, 1] = -M3
#     N[5, 5] = M5
#     N[5, 7] = -M4
#     N[5, 11] = M6
#     return N

def ShapeMat(self, ξ, ell):
    # from Shabana (2020) Eq. (6.113) with ζ=0 and η=0
    N = np.array([[1-ξ,                   0,                    0],
                  [  0,     1-3*ξ**2+2*ξ**3,                    0],
                  [  0,                   0,      1-3*ξ**2+2*ξ**3],
                  [  0,                   0,                    0],
                  [  0,                   0, -ell*(ξ-2*ξ**2+ξ**3)],
                  [  0, ell*(ξ-2*ξ**2+ξ**3),                    0],
                  [  ξ,                   0,                    0],
                  [  0,       3*ξ**2-2*ξ**3,                    0],
                  [  0,                   0,        3*ξ**2-2*ξ**3],
                  [  0,                   0,                    0],
                  [  0,                   0,    -ell*(-ξ**2+ξ**3)],
                  [  0,    ell*(-ξ**2+ξ**3),                    0]]).T
    return N

def TransXMat(self, i):
    # from Shabana (2020) Eq. (6.114)
    # local x-axis goes from first element to the second element
    # local z-axis is perpendicular (cross product) to local x-axis and global y-axis
    a1 = self.Nodes[self.El[i, 0]-1, 0]  # x1
    a2 = self.Nodes[self.El[i, 0]-1, 1]  # y1
    a3 = self.Nodes[self.El[i, 0]-1, 2]  # z1
    b1 = self.Nodes[self.El[i, 1]-1, 0]  # x1
    b2 = self.Nodes[self.El[i, 1]-1, 1]  # y2
    b3 = self.Nodes[self.El[i, 1]-1, 2]  # z2
    c1 = (b1-a1)/self.ell[i]  # xDelta/ell
    c2 = (b2-a2)/self.ell[i]  # yDelta/ell
    c3 = (b3-a3)/self.ell[i]  # zDelta/ell
    if (c1**2+c3**2) == 0:
        T3 = np.array([[ 0, -c2, 0],
                       [c2,   0, 0],
                       [ 0,   0, 1]])
    else:
        T3 = np.array([[c1, -(c1*c2)/sqrt(c1**2+c3**2), -c3/sqrt(c1**2+c3**2)],
                       [c2,          sqrt(c1**2+c3**2),                     0],
                       [c3, -(c2*c3)/sqrt(c1**2+c3**2),  c1/sqrt(c1**2+c3**2)]])
    return T3

def TransMat(self, i):
    T = np.zeros([2*self.nNDoF, 2*self.nNDoF])
    T[0:3, 0:3] = T[3:6, 3:6] = T[6:9, 6:9] = T[9:12, 9:12] = self.TX[i].T
    return T

def StrainDispMat(self, ξ, ell, y, z, r):
    B = np.array([[-1/ell,                   0,                   0,      0,               0,                0, 1/ell,                  0,                  0,     0,               0,                0],
                  [     0, -6*(2*ξ - 1)/ell**2,                   0,      0,               0, -2*(3*ξ - 2)/ell,     0, 6*(2*ξ - 1)/ell**2,                  0,     0,               0, -2*(3*ξ - 1)/ell],
                  [     0,                   0, -6*(2*ξ - 1)/ell**2,      0, 2*(3*ξ - 2)/ell,                0,     0,                  0, 6*(2*ξ - 1)/ell**2,     0, 2*(3*ξ - 1)/ell,                0],
                  [     0,                   0,                   0, -1/ell,               0,                0,     0,                  0,                  0, 1/ell,               0,                0]])
    B[1, :] *= y
    B[2, :] *= z
    B[3, :] *= r
    return(B)

def StiffMatElem(self, i):
    # from: https://github.com/airinnova/framat/blob/8ebbc85d048484c339703ae107df6247e43990b6/src/framat/_element.py
    A = self.A[i]
    E = self.E[i]
    ell = self.ell[i]
    Ix = self.Ix[i]
    Iy = self.Iy[i]
    Iz = self.Iz[i]
    nu = self.nu[i]
    ϰ = self.ϰ[i]
    G = self.G[i]

    # Bending terms after Euler-Bernoulli
    k = np.zeros((12, 12))
    if self.stiffMatType[0].lower() in ["e", "b"]:
        k[0, 0] = k[6, 6] = E*A/ell
        k[1, 1] = k[7, 7] = E*12*Iz/ell**3
        k[2, 2] = k[8, 8] = E*12*Iy/ell**3
        k[3, 3] = k[9, 9] = G*Ix/ell
        k[4, 4] = k[10, 10] = 4*E*Iy/ell
        k[5, 5] = k[11, 11] = 4*E*Iz/ell
        k[0, 6] = -E*A/ell
        k[1, 5] = 6*E*Iz/ell**2
        k[1, 7] = -12*E*Iz/ell**3
        k[1, 11] = 6*E*Iz/ell**2
        k[2, 4] = -6*E*Iy/ell**2
        k[2, 8] = -12*E*Iy/ell**3
        k[2, 10] = -6*E*Iy/ell**2
        k[3, 9] = -G*Ix/ell
        k[4, 8] = 6*E*Iy/ell**2
        k[4, 10] = 2*E*Iy/ell
        k[5, 7] = -6*E*Iz/ell**2
        k[5, 11] = 2*E*Iz/ell
        k[7, 11] = -6*E*Iz/ell**2
        k[8, 10] = 6*E*Iy/ell**2
        k += np.triu(k, k=1).T
    # Bending terms after Timoshenko-Ehrenfest
    elif self.stiffMatType[0].lower() == "t":
        "print not yet avaiable for 3d!!!!"
    return k

def MatMat(self, i):
    A = self.A[i]
    E = self.E[i]
    Ix = self.Ix[i]
    Iy = self.Iy[i]
    Iz = self.Iz[i]
    G = self.G[i]
    return(np.diag([E*A, E*Iy, E*Iz, G*A, G*A, G*Ix]))

def MassMatElem(self, i):
    A = self.A[i]
    ell = self.ell[i]
    rho = self.rho[i]
    Ix = self.Ix[i]
    Iy = self.Iy[i]
    Iz = self.Iz[i]

    if self.stiffMatType[0].lower() in ["e", "b"]:
        if self.massMatType[0].lower() == "c":
            if self.nodalInertia == False:
                m = np.array([[1/3,           0,           0, 0,           0,           0, 1/6,           0,           0, 0,           0,           0],
                              [  0,       13/35,           0, 0,           0,  11*ell/210,   0,        9/70,           0, 0,           0, -13*ell/420],
                              [  0,           0,       13/35, 0, -11*ell/210,           0,   0,           0,        9/70, 0,  13*ell/420,           0],
                              [  0,           0,           0, 0,           0,           0,   0,           0,           0, 0,           0,           0],
                              [  0,           0, -11*ell/210, 0,  ell**2/105,           0,   0,           0, -13*ell/420, 0, -ell**2/140,           0],
                              [  0,  11*ell/210,           0, 0,           0,  ell**2/105,   0,  13*ell/420,           0, 0,           0, -ell**2/140],
                              [1/6,           0,           0, 0,           0,           0, 1/3,           0,           0, 0,           0,           0],
                              [  0,        9/70,           0, 0,           0,  13*ell/420,   0,       13/35,           0, 0,           0, -11*ell/210],
                              [  0,           0,        9/70, 0, -13*ell/420,           0,   0,           0,       13/35, 0,  11*ell/210,           0],
                              [  0,           0,           0, 0,           0,           0,   0,           0,           0, 0,           0,           0],
                              [  0,           0,  13*ell/420, 0, -ell**2/140,           0,   0,           0,  11*ell/210, 0,  ell**2/105,           0],
                              [  0, -13*ell/420,           0, 0,           0, -ell**2/140,   0, -11*ell/210,           0, 0,           0,  ell**2/105]])
                m *= A*rho*ell
            elif self.nodalInertia[0].lower() == "t":
                m = np.array([[1/3,           0,           0,      0,           0,           0, 1/6,           0,           0,      0,           0,           0],
                              [  0,       13/35,           0,      0,           0,  11*ell/210,   0,        9/70,           0,      0,           0, -13*ell/420],
                              [  0,           0,       13/35,      0, -11*ell/210,           0,   0,           0,        9/70,      0,  13*ell/420,           0],
                              [  0,           0,           0, Ix/A/3,           0,           0,   0,           0,           0, Ix/A/6,           0,           0],
                              [  0,           0, -11*ell/210,      0,  ell**2/105,           0,   0,           0, -13*ell/420,      0, -ell**2/140,           0],
                              [  0,  11*ell/210,           0,      0,           0,  ell**2/105,   0,  13*ell/420,           0,      0,           0, -ell**2/140],
                              [1/6,           0,           0,      0,           0,           0, 1/3,           0,           0,      0,           0,           0],
                              [  0,        9/70,           0,      0,           0,  13*ell/420,   0,       13/35,           0,      0,           0, -11*ell/210],
                              [  0,           0,        9/70,      0, -13*ell/420,           0,   0,           0,       13/35,      0,  11*ell/210,           0],
                              [  0,           0,           0, Ix/A/6,           0,           0,   0,           0,           0, Ix/A/3,           0,           0],
                              [  0,           0,  13*ell/420,      0, -ell**2/140,           0,   0,           0,  11*ell/210,      0,  ell**2/105,           0],
                              [  0, -13*ell/420,           0,      0,           0, -ell**2/140,   0, -11*ell/210,           0,      0,           0,  ell**2/105]])
                m *= A*rho*ell
            elif self.nodalInertia[0].lower() == "a":
                m = np.array([[1/3,                           0,                           0,        0,                           0,                          0, 1/6,                           0,                           0,        0,                          0,                           0],
                              [  0,   13/35 + 6*Iz/(5*A*ell**2),                           0,        0,                           0, 11*ell/210 + Iz/(10*A*ell),   0,    9/70 - 6*Iz/(5*A*ell**2),                           0,        0,                          0, -13*ell/420 + Iz/(10*A*ell)],
                              [  0,                           0,   13/35 + 6*Iy/(5*A*ell**2),        0, -11*ell/210 - Iy/(10*A*ell),                          0,   0,                           0,    9/70 - 6*Iy/(5*A*ell**2),        0, 13*ell/420 - Iy/(10*A*ell),                           0],
                              [  0,                           0,                           0, Ix/(3*A),                           0,                          0,   0,                           0,                           0, Ix/(6*A),                          0,                           0],
                              [  0,                           0, -11*ell/210 - Iy/(10*A*ell),        0,  (A*ell**2 + 14*Iy)/(105*A),                          0,   0,                           0, -13*ell/420 + Iy/(10*A*ell),        0,    -ell**2/140 - Iy/(30*A),                           0],
                              [  0,  11*ell/210 + Iz/(10*A*ell),                           0,        0,                           0, (A*ell**2 + 14*Iz)/(105*A),   0,  13*ell/420 - Iz/(10*A*ell),                           0,        0,                          0,     -ell**2/140 - Iz/(30*A)],
                              [1/6,                           0,                           0,        0,                           0,                          0, 1/3,                           0,                           0,        0,                          0,                           0],
                              [  0,    9/70 - 6*Iz/(5*A*ell**2),                           0,        0,                           0, 13*ell/420 - Iz/(10*A*ell),   0,   13/35 + 6*Iz/(5*A*ell**2),                           0,        0,                          0, -11*ell/210 - Iz/(10*A*ell)],
                              [  0,                           0,    9/70 - 6*Iy/(5*A*ell**2),        0, -13*ell/420 + Iy/(10*A*ell),                          0,   0,                           0,   13/35 + 6*Iy/(5*A*ell**2),        0, 11*ell/210 + Iy/(10*A*ell),                           0],
                              [  0,                           0,                           0, Ix/(6*A),                           0,                          0,   0,                           0,                           0, Ix/(3*A),                          0,                           0],
                              [  0,                           0,  13*ell/420 - Iy/(10*A*ell),        0,     -ell**2/140 - Iy/(30*A),                          0,   0,                           0,  11*ell/210 + Iy/(10*A*ell),        0, (A*ell**2 + 14*Iy)/(105*A),                           0],
                              [  0, -13*ell/420 + Iz/(10*A*ell),                           0,        0,                           0,    -ell**2/140 - Iz/(30*A),   0, -11*ell/210 - Iz/(10*A*ell),                           0,        0,                          0,  (A*ell**2 + 14*Iz)/(105*A)]])
                m *= A*rho*ell
        elif self.massMatType[0].lower() == "l":
            if self.nodalInertia == False:
                m = np.diag([1/2, 1/2, 1/2, 0, ell**2*self.alphaLump, ell**2*self.alphaLump,
                             1/2, 1/2, 1/2, 0, ell**2*self.alphaLump, ell**2*self.alphaLump])
                m *= A*rho*ell
            elif self.nodalInertia[0].lower() == "t":
                m = np.diag([1/2, 1/2, 1/2, Ix/(2*A), ell**2*self.alphaLump, ell**2*self.alphaLump,
                             1/2, 1/2, 1/2, Ix/(2*A), ell**2*self.alphaLump, ell**2*self.alphaLump])
                m *= A*rho*ell
            elif self.nodalInertia[0].lower() == "a":
                m = np.diag([1/2, 1/2, 1/2, Ix/(2*A), Iy/(2*A)+ell**2*self.alphaLump, Iz/(2*A)+ell**2*self.alphaLump,
                             1/2, 1/2, 1/2, Ix/(2*A), Iy/(2*A)+ell**2*self.alphaLump, Iz/(2*A)+ell**2*self.alphaLump])
                m *= A*rho*ell
    elif self.stiffMatType[0].lower() == "t":
        pass
    return m
