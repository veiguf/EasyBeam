import KratosMultiphysics as km
import KratosMultiphysics.StructuralMechanicsApplication as sma
import KratosMultiphysics.LinearSolversApplication as esa
import numpy as np
import time

# Parameter
b = 10      # mm
h = 20      # mm
F = -100
l = 1000
E = 210000
rho = 7.85e-9
nu = 0.3
I = b*h**3/12
A = b*h
nElement = 2
nMode = 3

# Initialize Kratos model
model = km.Model()
mp = model.CreateModelPart("Structure")

# Add solution variables
mp.AddNodalSolutionStepVariable(km.DISPLACEMENT)
mp.AddNodalSolutionStepVariable(km.REACTION)
mp.AddNodalSolutionStepVariable(km.REACTION_MOMENT)
mp.AddNodalSolutionStepVariable(km.ROTATION)
mp.AddNodalSolutionStepVariable(km.NODAL_MASS)
mp.AddNodalSolutionStepVariable(km.VOLUME_ACCELERATION)

# Material and element properties
mp.GetProperties()[0].SetValue(sma.CROSS_AREA, A)
mp.GetProperties()[0].SetValue(sma.I33, I)
mp.GetProperties()[0].SetValue(km.YOUNG_MODULUS, E)
mp.GetProperties()[0].SetValue(km.DENSITY, rho)
mp.GetProperties()[0].SetValue(km.POISSON_RATIO, nu)
mp.GetProperties()[0].SetValue(km.CONSTITUTIVE_LAW, sma.BeamConstitutiveLaw())

# Create mesh
xNode = np.linspace(0, l, nElement+1)
for i, xi in enumerate(xNode):
    mp.CreateNewNode(i+1, xi, 0.0, 0.0)
elementType = "CrLinearBeamElement2D2N"
for i in range(nElement):
    mp.CreateNewElement(elementType, i+1, [i+1, i+2], mp.GetProperties()[0])

# Add degrees of freedom and their reactions
km.VariableUtils().AddDof(km.DISPLACEMENT_X, km.REACTION_X, mp)
km.VariableUtils().AddDof(km.DISPLACEMENT_Y, km.REACTION_Y, mp)
km.VariableUtils().AddDof(km.ROTATION_Z, km.REACTION_MOMENT_Z, mp)

# Boundry conditions
boundaryGroup = mp.CreateSubModelPart("BoundaryCondtionsDirichlet")
boundaryGroup.AddNodes([1])
km.VariableUtils().ApplyFixity(km.DISPLACEMENT_X, True, boundaryGroup.Nodes)
km.VariableUtils().ApplyFixity(km.DISPLACEMENT_Y, True, boundaryGroup.Nodes)
km.VariableUtils().ApplyFixity(km.ROTATION_Z, True, boundaryGroup.Nodes)

# Solver settings (can be done in a more Python format???)
SolverPar = km.Parameters("""{"solver_type"           : "eigen_eigensystem",
                              "max_iteration"         : 2000,
                              "tolerance"             : 1e-6,
                              "number_of_eigenvalues" : """+ str(nMode) + """,
                              "echo_level"            : 0,
                              "normalize_eigenvectors": true}""")
Solver = esa.EigensystemSolver(SolverPar)
BuilderSolver = km.ResidualBasedBlockBuilderAndSolver(Solver)
Scheme = sma.EigensolverDynamicScheme()
ModalDecomposition = False
echo = 0
Strategy = sma.EigensolverStrategy(mp, Scheme, BuilderSolver,
                                   ModalDecomposition, 1, 1)
Strategy.SetEchoLevel(echo)

# Solve
Strategy.Solve()

# Postprocessing
eigenvalues = mp.ProcessInfo[sma.EIGENVALUE_VECTOR]
time.sleep(0.01)
print("done")
omegan = []
fn = []
for i in range(nMode):
    omegan.append(eigenvalues[i])
    fn.append((eigenvalues[i]**0.5)/2/np.pi)
time.sleep(0.01)
print(fn)
