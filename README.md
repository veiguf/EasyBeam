<p align=center><img width="100%" src="figures/EasyBeam2.svg"></p>

# EasyBeam

**Easy Application for Structural analYsis with BEAMs**

A Python package for the structural analysis of planar structures based on Euler-Bernoulli beam theory. EasyBeam analyzes both statics, giving deformation and stress reslts, as well as the eigenvalue problem to assess the free-vibration resonance behavior.  Ths package is geared education and therefore trimmed to code readability and expandability, while maintaining usability and easy to understand graphical dipiction of the results. 

## Installation
### Prerequisites
Python 3 with the packages SciPy, NumPy and MatPlotLib are needed.  After installation of Python, you can install the necessary libraries via PIP:
```
pip install scipy
pip install numpy
pip install matplotlib
```

### Install
```
python setup.py install
```

### PIP
You can also install EasyBeam via PIP
```
pip install EasyBeam
```

## How to use 
* Step 1: definition of nodes

The coordinates of the nodes must be defined. In the shown example, three nodes are defined. For each node, the first value defines the x-coordinate and the second value defines the y-coordinate of the node.
```
Example.Nodes = [[ 100,   0],  # node 0 at x=100 and y=0
                 [ 200,   0],  # node 1 at x=200 and y=0
                 [ 200, 100]]  # node 2 at x=200 and y=100
```
* Step 2: definition of elements

Here are defined the elements and which nodes they connect. In the shown example, two elements are defined. For each element, the two node numbers, which the element connects, are selected.
```
Example.El = [[ 0,   1],  # element 0 between node 0 and node 1
              [ 1,   2]]  # element 1 between node 1 and node 2
```
* Step 3: definition of boundary conditions

It is possible to define boundary conditions on each node. The first value defines the node number on which the boundary condition is applied. The following three values defines the displacement in x-direction, the displacement in y-direction and the rotation. In the following example, node 0 is fixed in all three degrees of freedom.
```
Example.Disp = [[0, [0, 0, 0]]]  # node 0 with 0 displacement in x, 0 displacement in y and 0 rotation
```
In the following example, for node 0 the displacement is fixed in x and y direction and for node 1 the displacement is fixed in y direction.
```
Example.Disp = [[0, [  0, 0, 'f']],  # node 0 with 0 displacement in x, 0 displacement in y and free rotation
                [1, ['f', 0, 'f']]]  # node 0 with free displacement in x, 0 displacement in y and free rotation
```
It is also possible to apply a known displacement on a node. In the following example, node 0 is fixed in all three degrees of freedom and on node 1 is applied a displacement of 1 in y direction and a rotation of 0.1.
```
Example.Disp = [[0, [  0, 0,   0]],  # node 0 with 0 displacement in x, 0 displacement in y and 0 rotation
                [1, ['f', 1, 0.1]]]  # node 1 with free displacement in x, displacement of 1 in y and rotation of 0.1
```
* Step 4: definition of loads

The definition of loads is done similar to the definition of boundary conditions. The first value defines the node number on which the load is applied. The following three values defines the force in x-direction, the force in y-direction and the torque. In the following example, on node 1 is applied a force in x-direction of 100 and on node 2 is applied a force in y-direction of 200 and a torque of 50.
```
Example.Load = [[1, [100, 'f', 'f']],  # node 1 with a force in x of 100, no force in y and no torque
                [2, ['f', 200,  50]]]  # node 2 with no force in x, a force of 200 in y and a torque of 50
```
* Step 5: definition of cross section and material properties

Firstly, it is necessary to define the properties for the cross sections and materials. In the following example, two different property sets are defined.
```
Example.Properties = [['Mat1', 7.85e-9, 206900, 0.29, 'rect', 20, 10],  # 'Mat1' with density 7.85e-9, elastic modulus 206900, Poisson ration 0.29, rectangular cross section, height of 20 and width of 10
                      ['Mat2', 2.70e-9,  71000, 0.34, 'rect', 25, 15]]  # 'Mat2' with density 2.70e-9, elastic modulus  71000, Poisson ration 0.34, rectangular cross section, height of 25 and width of 15
```
Secondly, the property sets must be assigned to the elements. Therefore, a list is created where for each element, the property set is chosen. In the following example, 'Mat1' is assigned to element 0 and 'Mat2' is assigned to element 1.
```
Example.PropID = ['Mat1', 'Mat2']  # 'Mat1' is assigned to element 0 and 'Mat2' is assigned to element 1
```
