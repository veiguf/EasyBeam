import numpy as np
from numpy import pi

def ShapeMat(self, ξ, ell):
    return(np.array([[1-ξ,               0,              0, ξ,            0,              0],
                     [  0, 1-3*ξ**2+2*ξ**3, ξ*ell*(1-ξ)**2, 0, ξ**2*(3-2*ξ), ξ**2*ell*(ξ-1)]]))

def TransXMat(self, i):
    xDelta = self.Nodes[self.El[i, 1]-1, 0]-self.Nodes[self.El[i, 0]-1, 0]
    yDelta = self.Nodes[self.El[i, 1]-1, 1]-self.Nodes[self.El[i, 0]-1, 1]
    if xDelta > 0:
        β = np.arctan(yDelta/xDelta)
    elif xDelta < 0:
        β = pi + np.arctan(yDelta/xDelta)
    else:       # elif xDelta == 0:
        if yDelta > 0:
            β = pi/2
        elif yDelta < 0:
            β = -pi/2
        else:   # elif yDelta == 0:
            raise NameError('Nodes have same position')
    T2 = np.array([[np.cos(β), -np.sin(β)],
                   [np.sin(β),  np.cos(β)]], dtype=float)
    return T2

def TransMat(self, i):
    T = np.block([[    self.TX[i].T,                  np.zeros([2, 4])],
                  [            0, 0, 1,         0, 0,                0],
                  [   np.zeros([2, 3]), self.TX[i].T, np.zeros([2, 1])],
                  [            0, 0, 0,         0, 0,                1]])
    return T

def StrainDispMat(self, ξ, ell, y, z, r):
    B = np.array([[-1/ell,                              0,                0, 1/ell,                             0,                        0],
                  [     0,            -6*(2*ξ - 1)/ell**2, -2*(3*ξ - 2)/ell,     0,            2*(6*ξ - 3)/ell**2,         -2*(3*ξ - 1)/ell]])
    B[1, :] *= z
    return(B)

def StrainDispNablah(self, ξ, ell):
    BLNablah = np.array([[0,                0,            0, 0,                 0,             0],
                         [0, -1/2*(6-12*ξ)/ell**2, -1/2*(4-6*ξ)/ell, 0, -1/2*(-6+12*ξ)/ell**2, -1/2*(-6*ξ+2)/ell]])
    BUNablah = np.array([[0,               0,           0, 0,                0,            0],
                         [0, 1/2*(6-12*ξ)/ell**2, 1/2*(4-6*ξ)/ell, 0, 1/2*(-6+12*ξ)/ell**2, 1/2*(-6*ξ+2)/ell]])
    return(BLNablah, BUNablah)

def StiffMatElem(self, i):
    A = self.A[i]
    E = self.E[i]
    ell = self.ell[i]
    I = self.Iy[i]
    nu = self.nu[i]
    ϰ = self.ϰ[i]
    # bar (column) terms of stiffness matrix
    k = E*A/ell*np.array([[ 1, 0, 0, -1, 0, 0],
                          [ 0, 0, 0,  0, 0, 0],
                          [ 0, 0, 0,  0, 0, 0],
                          [-1, 0, 0,  1, 0, 0],
                          [ 0, 0, 0,  0, 0, 0],
                          [ 0, 0, 0,  0, 0, 0]], dtype=float)

    # Bending terms after Euler-Bernoulli
    if self.stiffMatType[0].lower() in ["e", "b"]:
        phi = 0
    # Bending terms after Timoshenko-Ehrenfest
    elif self.stiffMatType[0].lower() == "t":
        G = E/(2*(1+nu))
        AS = A * ϰ
        phi = 12*E*I/(ϰ*A*G*ell**2)
    c = E*I/(ell**3*(1+phi))
    k += c*np.array([[0,     0,              0, 0,      0,                0],
                     [0,    12,          6*ell, 0,    -12,            6*ell],
                     [0, 6*ell, ell**2*(4+phi), 0, -6*ell,   ell**2*(2-phi)],
                     [0,     0,              0, 0,      0,                0],
                     [0,   -12,         -6*ell, 0,     12,           -6*ell],
                     [0, 6*ell, ell**2*(2-phi), 0, -6*ell,   ell**2*(4+phi)]],
                    dtype=float)
    return k

def MatMat(self, i):
    return(np.array([[self.E[i]*self.A[i],                    0],
                     [                  0, self.E[i]*self.Iy[i]]]))

def MassMatElem(self, i):
    ell = self.ell[i]
    rho = self.rho[i]
    A = self.A[i]
    ϰ = self.ϰ[i]
    if self.stiffMatType[0].lower() in ["e", "b"]:
        if self.massMatType[0].lower() == "c":
            # Felippa Eq. (31.20)
            c = A*rho*ell/420
            m = c*np.array([[140,       0,         0,  70,       0,         0],
                            [  0,     156,    22*ell,   0,      54,   -13*ell],
                            [  0,  22*ell,  4*ell**2,   0,  13*ell, -3*ell**2],
                            [ 70,       0,         0, 140,       0,         0],
                            [  0,      54,    13*ell,   0,     156,   -22*ell],
                            [  0, -13*ell, -3*ell**2,   0, -22*ell,  4*ell**2]],
                           dtype=float)
        elif self.massMatType[0].lower() == "l":
            # Felippa Eqs. (31.1) & (31.19)
            alpha = 1/78  # HRZ lumping
            c = A*rho*ell/2
            m = c*np.array([[ 1, 0,              0, 0, 0,              0],
                            [ 0, 1,              0, 0, 0,              0],
                            [ 0, 0, 2*alpha*ell**2, 0, 0,              0],
                            [ 0, 0,              0, 1, 0,              0],
                            [ 0, 0,              0, 0, 1,              0],
                            [ 0, 0,              0, 0, 0, 2*alpha*ell**2]],
                           dtype=float)
    elif self.stiffMatType[0].lower() == "t":
        IR = self.Iy[i]
        nu = 0.3
        G = self.E[i]/(2*(1+nu))
        AS = A * ϰ
        phi = 12*self.E[i]*self.Iy[i]/(ϰ*A*G*ell**2)
        m = A*rho*ell/420*np.array([[140, 0, 0,  70, 0, 0],
                                    [  0, 0, 0,   0, 0, 0],
                                    [  0, 0, 0,   0, 0, 0],
                                    [ 70, 0, 0, 140, 0, 0],
                                    [  0, 0, 0,   0, 0, 0],
                                    [  0, 0, 0,   0, 0, 0]], dtype=float)
        # tranlational inertia
        cT = A*rho*ell/(1+phi)**2
        m += cT*np.array([[0,                                   0,                                     0, 0,                                   0,                                     0],
                          [0,           13/35+7/10*phi+1/3*phi**2,   (11/210+11/120*phi+1/24*phi**2)*ell, 0,            9/70+3/10*phi+1/6*phi**2,    -(13/420+3/40*phi+1/24*phi**2)*ell],
                          [0, (11/210+11/120*phi+1/24*phi**2)*ell,  (1/105+1/60*phi+1/120*phi**2)*ell**2, 0,   (13/420+3/40*phi+1/24*phi**2)*ell, -(1/140+1/60*phi+1/120*phi**2)*ell**2],
                          [0,                                   0,                                     0, 0,                                   0,                                     0],
                          [0,            9/70+3/10*phi+1/6*phi**2,     (13/420+3/40*phi+1/24*phi**2)*ell, 0,           13/35+7/10*phi+1/3*phi**2,   (11/210+11/120*phi+1/24*phi**2)*ell],
                          [0,  -(13/420+3/40*phi+1/24*phi**2)*ell, -(1/140+1/60*phi+1/120*phi**2)*ell**2, 0, (11/210+11/120*phi+1/24*phi**2)*ell,  (1/105+1/60*phi+1/120*phi**2)*ell**2]],
                         dtype=float)
        # rotary inertia
        cR = rho*IR/(ell*(1+phi)**2)
        m += cR*np.array([[0,                  0,                                0, 0,                   0,                                0],
                          [0,                6/5,               (1/10-1/2*phi)*ell, 0,                -6/5,               (1/10-1/2*phi)*ell],
                          [0, (1/10-1/2*phi)*ell, (2/15+1/6*phi+1/3*phi**2)*ell**2, 0, (-1/10+1/2*phi)*ell, (1/30+1/6*phi-1/6*phi**2)*ell**2],
                          [0,                  0,                                0, 0,                   0,                                0],
                          [0,               -6/5,              (-1/10+1/2*phi)*ell, 0,                 6/5,              (-1/10+1/2*phi)*ell],
                          [0, (1/10-1/2*phi)*ell, (1/30+1/6*phi-1/6*phi**2)*ell**2, 0, (-1/10+1/2*phi)*ell, (2/15+1/6*phi+1/3*phi**2)*ell**2]],
                         dtype=float)
    return m
