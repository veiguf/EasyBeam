from EasyBeam import Beam2D
import numpy as np
from scipy import optimize

# Parameter
b = 1  # mm
h = 1  # mm
F = -100  # N
l = 1000  # mm
E = 210000  # MPa
rho = 7.85e-9  # t/mm^3
I = b * h ** 3 / 12  # mm^4
A = b * h  # mm^2
nEl = 50
nu = 0.3

# Target shape
xBeam = np.linspace(0, l, nEl + 1)
# sine curve
ySin = 10 * np.sin(4 * np.pi * xBeam / l)
# linearly increasing
yLin = xBeam / (l) * 10
# sine curve sloped
ySinLin = np.sin(xBeam / l) * xBeam / (l) * 10
# v-shape
yV = np.zeros((51,))
yV[:26] = xBeam[:26] / l * -20
yV[25:] = np.flip(yV[:26])
# w-shape
yW = np.zeros((51,))
yW[:13] = xBeam[:13] / l * -40
yW[12:25] = np.flip(yW[:13])
yW[25:38] = yW[:13]
yW[38:] = np.flip(yW[:13])
yTarget = yW
#yTarget = yV
#yTarget = ySin
# yTarget = yLin
#yTarget = ySinLin
ShapeTarget = np.zeros(((nEl + 1) * 2,))
ShapeTarget[1::2] = yTarget


def SetupFE():
    Cantilever = Beam2D()
    Cantilever.SizingVariables = [["h", "b"]]
    Cantilever.stiffMatType = "Euler-Bernoulli"

    Cantilever.Nodes = [[]] * (nEl + 1)
    for i in range(nEl + 1):
        Cantilever.Nodes[i] = [l * i / nEl, 0.0]

    Cantilever.El = [[]] * (nEl)
    for i in range(nEl):
        Cantilever.El[i] = [i + 1, i + 2]

    Cantilever.Properties = [["Prop1", rho, E, nu, "rect", h, b]]
    Cantilever.PropID = ["Prop1"] * nEl
    Cantilever.Disp = [[1, [0.0, 0.0, 0.0]]]
    return Cantilever


def CalcStroke(nodesAct):
    nAct = len(nodesAct)
    ii = 0
    uMat = np.zeros(((nEl + 1) * 3, nAct * 2))
    for i in range(nAct):
        for j in range(2):
            if j == 0:
                loading = [1.0, 0, ""]
            elif j == 1:
                loading = [0, 1.0, ""]
            x = [0] * (len(nodesAct) * 2)
            x[ii] = 1.0
            Cantilever = SetupFE()
            for jj in range(nAct):
                Cantilever.Disp.append([nodesAct[jj], [x[jj * 2], x[jj * 2 + 1], "f"]])
            Cantilever.StaticAnalysis()
            uMat[:, ii] = Cantilever.u
            # Cantilever.PlotDisplacement(component='y', scale=20)
            ii += 1

    dofControl = []
    nodesControl = range(Cantilever.nN)
    for i in range(Cantilever.nN):
        dofControl.append(nodesControl[i] * 3)
        dofControl.append(nodesControl[i] * 3 + 1)
    H = uMat[dofControl, :]

    res = optimize.lsq_linear(H, ShapeTarget)
    uStroke = res.x

    # Validate uStroke and calculate RMSE
    Cantilever = SetupFE()
    for jj in range(nAct):
        Cantilever.Disp.append(
            [nodesAct[jj], [uStroke[jj * 2], uStroke[jj * 2 + 1], "f"]]
        )
    Cantilever.StaticAnalysis()
    # only x and y (no rotation!)
    #Cantilever.PlotDisplacement(component="y", scale=20)
    Cantilever.PlotStress(stress='max', scale=20)
    eRMS = np.sqrt(
        np.sum((Cantilever.u[dofControl] - ShapeTarget) ** 2) / len(ShapeTarget)
    )
    return eRMS, uStroke, Cantilever.F

# definition of actuators
nodesAct = [[]] * 8
eRMS = [[]] * 8
uStroke = [[]] * 8
F = [[]] * 8
nodesAct[0] = np.array(range(11, 52, 10)).tolist()
nodesAct[1] = np.array(range(11, 52, 8)).tolist()
nodesAct[2] = np.array(range(9, 52, 6)).tolist()
nodesAct[3] = np.array(range(6, 52, 5)).tolist()
nodesAct[4] = np.array(range(5, 52, 4)).tolist()
nodesAct[5] = np.array(range(3, 52, 3)).tolist()
nodesAct[6] = np.array(range(2, 52, 2)).tolist()
nodesAct[7] = np.array(range(1, 52, 1)).tolist()

# Parameter study
for i in range(len(nodesAct)):
    eRMS[i], uStroke[i], F[i] = CalcStroke(nodesAct[i])
    print(eRMS[i])
