# GOMORS_Python3.7_PYSOT0.2.0
This GOMORS algorithm is the modified python version of the originally developed code which had been uploaded in this repository: https://github.com/drkupi/GOMORS_pySOT. 


# What is GOMORS?
GOMORS is a surrogate-assisted Multi-Objective Optimization (MO) strategy, designed for computational expensive MO problems, e.g., expensive environmental simulation optimization problems, hyperparameter tuning of Deep Neural Networks etc. GOMORS is implemented in the pySOT lbirary and framework, and uses Radial Basis Functions (RBFs), as surrogates. Moreover, GOMORS uses a Multi Objective Evolutionary Strategy (MOEA) to optimize RBF surrogates in each iteration. Any MOEA methods can be connected with GOMORS for optimizing surrogates. We currently use the Platypus library to link GOMORS with epsilon-MOEA. GOMORS also supports modest parallelization on up to 4 cores, and hence, is suitable for deskptop and laptop machines.

# What are the changes in this GOMORS repository?
The originally developed GOMORS in the "drkupi/GOMORS_pySOT" is functional only in the Python 2.7 version. Since the pysot library is available for the latest python versions, the GOMORS has been imported from Python 2.7 to Python 3.7. Also, the pysot version 0.1.36 used in the original version has been replaced with the pysot version 0.2.0.

# Installation Instructions
The prerequisites for using the GOMORS code in this repository is installation of a python 3.7 environment, the pySOT library, the platypus library, numpy, scipy and matplotlib. We recommend using a virtual environment within Anaconda. Instructions for installation of pre-requisites is as follows:

```
conda create --name mo-surrogate python=3.7
conda activate mo-surrogate
pip install pysot==0.2.0
pip install matplotlib
pip install platypus-opt
pip install numpy==1.16.5
pip install scipy==1.6.0
```

# Running GOMORS
An example of how to run GOMORS is provided in the file simple_experiment.py. The setup for running the algorithm is synonymous to how optimization experiments are setup in pysot. To link GOMORS to a user-defined MO optimization problem, kindly look at how problems are defined in the test_problems.py python file. For further information please write to me at vijeysubramani@gmail.com

# References
1. Akhtar, T., Shoemaker, C.A. Multi objective optimization of computationally expensive multi-modal functions with RBF surrogates and multi-rule selection. J Glob Optim 64, 17–32 (2016). https://doi.org/10.1007/s10898-015-0270-y
2. https://github.com/drkupi/GOMORS_pySOT
