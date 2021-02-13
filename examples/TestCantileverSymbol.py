import sympy as sym

bVal = 10          # mm
hVal = 20          # mm
FyVal = -100        # N
ellVal = 1000        # mm
EVal = 210000      # MPa
rhoVal = 7.85e-9   # t/mm^3

sym.init_printing()
# Primary analysis
u = sym.Matrix([[0, 0, 0, 0, 0, 0]]).T
h, b, E, rho, A, I, ell, Fy = sym.symbols('h, b, E, rho, A, I, ell, F_y')
A = h*b
I = b*h**3/12
# kCon = sym.Matrix([[-ell/(A*E),              0,              0],
#                    [          0, -ell**3/(12*E*I), 6*E*I/(ell**2)],
#                    [          0,              0,    ell/(2*E*I)]])
kCon = sym.Matrix([[A*E/ell,             0,            0],
                   [      0, 12*E*I/ell**3, -6*E*I/ell**2],
                   [      0, -6*E*I/ell**2,     4*E*I/ell]])
FCon = sym.Matrix([[Fy, Fy, 0]]).T
kConinv = kCon.inv()
uCon = kConinv*FCon
u[3:6,0] = uCon
zL = -h/2
zU = h/2
xi = [0, 1]
epsilonL =[[]]*2
epsilonU =[[]]*2
sigmaL =[[]]*2
sigmaU =[[]]*2
for i in range(2):
    BL = sym.Matrix([[-1/ell,                   0,               0, 1/ell,                    0,               0],
                     [     0, zL*(6-12*xi[i])/ell**2, zL*(4-6*xi[i])/ell,     0, zL*(-6+12*xi[i])/ell**2, zL*(2-6*xi[i])/ell]])
    BU = sym.Matrix([[-1/ell,                   0,               0, 1/ell,                    0,               0],
                     [     0, zU*(6-12*xi[i])/ell**2, zU*(4-6*xi[i])/ell,     0, zU*(-6+12*xi[i])/ell**2, zU*(2-6*xi[i])/ell]])
    epsilonL[i] = BL*u
    epsilonU[i] = BU*u
    sigmaL[i] = E*epsilonL[i]
    sigmaU[i] = E*epsilonU[i]
print(kCon.subs([(h, hVal), (b, bVal), (E, EVal), (rho, rhoVal), (Fy, FyVal),
                 (ell, ellVal)]))
print("deformation")
print(u.T.subs([(h, hVal), (b, bVal), (E, EVal), (rho, rhoVal), (Fy, FyVal),
                (ell, ellVal)]))
print("strain")
for i in range(2):
    print(epsilonL[i].subs([(h, hVal), (b, bVal), (E, EVal), (rho, rhoVal),
                            (Fy, FyVal), (ell, ellVal)]))
    print(epsilonU[i].subs([(h, hVal), (b, bVal), (E, EVal), (rho, rhoVal),
                            (Fy, FyVal), (ell, ellVal)]))
print("stress")
for i in range(2):
    print(sigmaL[i].subs([(h, hVal), (b, bVal), (E, EVal), (rho, rhoVal),
                          (Fy, FyVal), (ell, ellVal)]))
    print(sigmaU[i].subs([(h, hVal), (b, bVal), (E, EVal), (rho, rhoVal),
                          (Fy, FyVal), (ell, ellVal)]))

# Sensitivity analysis
kConNablah = kCon.diff(h)
kConNablab = kCon.diff(b)
uNablah = sym.Matrix([[0, 0, 0, 0, 0, 0]]).T
uNablab = sym.Matrix([[0, 0, 0, 0, 0, 0]]).T
uConNablah = -kConinv*(kConNablah*uCon)
uConNablab = -kConinv*(kConNablab*uCon)
uNablah[3:6,0] = uConNablah
uNablab[3:6,0] = uConNablab
epsilonLNablah =[[]]*2
epsilonLNablab =[[]]*2
epsilonUNablah =[[]]*2
epsilonUNablab =[[]]*2
sigmaLNablah =[[]]*2
sigmaLNablab =[[]]*2
sigmaUNablah =[[]]*2
sigmaUNablab =[[]]*2
for i in range(2):
    BL = sym.Matrix([[-1/ell,                   0,               0, 1/ell,                    0,               0],
                     [     0, zL*(6-12*xi[i])/ell**2, zL*(4-6*xi[i])/ell,     0, zL*(-6+12*xi[i])/ell**2, zL*(2-6*xi[i])/ell]])
    BU = sym.Matrix([[-1/ell,                   0,               0, 1/ell,                    0,               0],
                     [     0, zU*(6-12*xi[i])/ell**2, zU*(4-6*xi[i])/ell,     0, zU*(-6+12*xi[i])/ell**2, zU*(2-6*xi[i])/ell]])
    BLNablah = BL.diff(h)
    BLNablab = BL.diff(b)
    BUNablah = BU.diff(h)
    BUNablab = BU.diff(b)
    epsilonLNablah[i] = BLNablah*u + BL*uNablah
    epsilonUNablah[i] = BUNablah*u + BU*uNablah
    epsilonLNablab[i] = BLNablab*u + BL*uNablab
    epsilonUNablab[i] = BUNablab*u + BU*uNablab
    sigmaLNablah[i] = E*epsilonLNablah[i]
    sigmaUNablah[i] = E*epsilonUNablah[i]
    sigmaLNablab[i] = E*epsilonLNablab[i]
    sigmaUNablab[i] = E*epsilonUNablab[i]
print("deformation sensitivity")
print(uNablah.T.subs([(h, hVal), (b, bVal), (E, EVal), (rho, rhoVal),
                    (Fy, FyVal), (ell, ellVal)]))
print(uNablab.T.subs([(h, hVal), (b, bVal), (E, EVal), (rho, rhoVal),
                    (Fy, FyVal), (ell, ellVal)]))

print("strain sensitivity")
for i in range(2):
    print(epsilonLNablah[i].subs([(h, hVal), (b, bVal), (E, EVal), (rho, rhoVal),
                        (Fy, FyVal), (ell, ellVal)]))
    print(epsilonLNablab[i].subs([(h, hVal), (b, bVal), (E, EVal), (rho, rhoVal),
                        (Fy, FyVal), (ell, ellVal)]))
    print(epsilonUNablah[i].subs([(h, hVal), (b, bVal), (E, EVal), (rho, rhoVal),
                        (Fy, FyVal), (ell, ellVal)]))
    print(epsilonUNablab[i].subs([(h, hVal), (b, bVal), (E, EVal), (rho, rhoVal),
                        (Fy, FyVal), (ell, ellVal)]))
print("stress sensitivity")
for i in range(2):
    print(sigmaLNablah[i].subs([(h, hVal), (b, bVal), (E, EVal), (rho, rhoVal),
                        (Fy, FyVal), (ell, ellVal)]))
    print(sigmaLNablab[i].subs([(h, hVal), (b, bVal), (E, EVal), (rho, rhoVal),
                        (Fy, FyVal), (ell, ellVal)]))
    print(sigmaUNablah[i].subs([(h, hVal), (b, bVal), (E, EVal), (rho, rhoVal),
                        (Fy, FyVal), (ell, ellVal)]))
    print(sigmaUNablab[i].subs([(h, hVal), (b, bVal), (E, EVal), (rho, rhoVal),
                        (Fy, FyVal), (ell, ellVal)]))

