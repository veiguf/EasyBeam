<p align=center><img height="50%" width="50%" src="figures/EasyBeam.png"></p>

# EasyBeam

**Easy Application for Structural analYsis with BEAMs**

## Installation
### Prerequisites
Python 3.xx along with the packages SciPy, NumPy and MatPlotLib.  After installation of Python, you can install the necessary libraries via PIP:
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
* Step 1: define coordinates of the nodes
* Step 2: define the elements: which nodes are connected
* Step 3: define the boundary conditions: which dofs are constrained
* Step 4: define the load: dof and value
* Step 5: define cross section and material properties
