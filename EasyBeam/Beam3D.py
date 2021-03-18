import numpy as np


def ShapeMat(self, ξ, ell):
    # from: https://github.com/airinnova/framat/blob/8ebbc85d048484c339703ae107df6247e43990b6/src/framat/_element.py
    N1 = 1 - xi
    N2 = xi
    N3 = 1 - 3*xi**2 + 2*xi**3
    N4 = 3*xi**2 - 2*xi**3
    N5 = L*(xi - 2*xi**2 + xi**3)
    N6 = L*(-xi**2 + xi**3)
    M1 = 1 - xi
    M2 = xi
    M3 = -(6/L)*(xi - xi**2)
    M4 = (6/L)*(xi - xi**2)
    M5 = 1 - 4*xi + 3*xi**2
    M6 = -2*xi + 3*xi**2
    N = np.zeros((6, 12))
    N[0, 0] = N1
    N[0, 6] = N2
    N[1, 1] = N3
    N[1, 5] = N5
    N[1, 7] = N4
    N[1, 11] = N6
    N[2, 2] = N3
    N[2, 4] = -N5
    N[2, 8] = N4
    N[2, 10] = -N6
    N[3, 3] = M1
    N[3, 9] = M2
    N[4, 2] = M3
    N[4, 4] = M5
    N[4, 8] = M4
    N[4, 10] = M6
    N[5, 1] = -M3
    N[5, 5] = M5
    N[5, 7] = -M4
    N[5, 11] = M6
    return N

def StrainDispMat(self, ξ, ell, zU, zL):

    return(BL, BU)

def StrainDispNablah(self, ξ, ell):
    # still 2d
    BLNablah = np.array([[0,                0,            0, 0,                 0,             0],
                         [0, -1/2*(6-12*ξ)/ell**2, -1/2*(4-6*ξ)/ell, 0, -1/2*(-6+12*ξ)/ell**2, -1/2*(-6*ξ+2)/ell]])
    BUNablah = np.array([[0,               0,           0, 0,                0,            0],
                         [0, 1/2*(6-12*ξ)/ell**2, 1/2*(4-6*ξ)/ell, 0, 1/2*(-6+12*ξ)/ell**2, 1/2*(-6*ξ+2)/ell]])
    return(BLNablah, BUNablah)

def StiffMatElem(self, i):
    A = self.A[i]
    E = self.E[i]
    ell = self.ell[i]
    Ix = self.Ix[i]
    Iy = self.Iy[i]
    Iz = self.Iz[i]
    nu = self.nu[i]
    ϰ = self.ϰ[i]

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
        k[4, 10] = 2*E*Iy/ell
        k[5, 11] = 2*E*Iz/ell
        k[7, 11] = -6*E*Iz/ell**2
        k[8, 10] = 6*E*Iy/ell**2
        k += np.triu(k, k=1).T
    # Bending terms after Timoshenko-Ehrenfest
    elif self.stiffMatType[0].lower() == "t":
        "print not yet avaiable for 3d!!!!"
    return k

def MatMat(self, i):
    E = self.E[i]
    Ix = self.Ix[i]
    Iy = self.Iy[i]
    Iz = self.Iz[i]
    G = self.G[i]
    return(np.diag([E*A, E*Iy, E*Iz, G*A, G*A, G*Ix]))

def MassMatElem(self, i):
    A = self.A[i]
    E = self.E[i]
    ell = self.ell[i]
    Ix = self.Ix[i]
    rho = self.rho[i]

    if self.stiffMatType[0].lower() in ["e", "b"]:
        if self.massMatType[0].lower() == "c":
            c = A*rho*ell/420
            m = np.zeros((12, 12))
            m[0, 0] = m[6, 6] = 140
            m[1, 1] = m[7, 7] = 156
            m[2, 2] = m[8, 8] = 156
            m[3, 3] = m[9, 9] = 140*Ix/A
            m[4, 4] = m[10, 10] = 4*ell**2
            m[5, 5] = m[11, 11] = 4*ell**2
            m[0, 6] = 70
            m[1, 5] = 22*ell
            m[1, 7] = 54
            m[1, 11] = -13*ell
            m[2, 4] = -22*ell
            m[2, 8] = 54
            m[2, 10] = 13*ell
            m[3, 9] = 70*Ix/A
            m[4, 8] = -13*ell
            m[4, 10] = -3*ell**2
            m[5, 7] = 13*ell
            m[5, 11] = -3*ell**2
            m[7, 11] = -22*ell
            m[8, 10] = 22*ell
            m += np.triu(m_elem, k=1).T
            m *= c

        elif self.massMatType[0].lower() == "l":
            pass
    elif self.stiffMatType[0].lower() == "t":
        pass
    return m
