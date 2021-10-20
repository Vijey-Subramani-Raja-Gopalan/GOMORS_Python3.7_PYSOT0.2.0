#!/usr/bin/env python

import time
from mo_utils import ND_Front
import math
from pySOT.optimization_problems import *
import numpy as np



class Yannis_Problem(OptimizationProblem):

    def __init__(self, nobj):
        dim = nobj
        self.lb = np.array([-512.0, -5.12])
        self.ub = np.array([512.0, 5.12])
        self.dim = dim
        self.nobj = nobj
        self.info = str(dim)+"-dimensional F7 Schwefel's function multimodal, asymmetric, separable problem \n" +\
                             "Global optimum: f(420.968746,420.968746,...,420.968746) = 0 \n" + \
                             "& F1 Sphere model unimodal, symmetric, separable problem \n" + \
                             "Global optimum: f(0,0,...,0) = 0 \n"
        self.int_var = []
        self.cont_var = np.arange(0, dim)
        self.nnn = 0
        self.pf = None     

    def objfunction(self, solution):

        """solution = Design variables values for evaluation #tvh"""
        self.__check_input__(solution)

        solution = list(solution)


        fitness = [0] * self.nobj


        "F7 Schwefel's function"
        alpha = 418.982887
        fitness[0] += alpha * self.dim
        "----------------------"

        for i in range(self.nobj):                  #loop through objectives
            #for j in range(self.nobj):
            for j in range(self.dim):               #loop through design variables
                if i == 0:                          #objective 1
                    "F7 Schwefel's function multimodal, asymmetric, separable"
                    #fitness[i] += solution[j] ** 2
                    fitness[i] -= solution[j] * math.sin(math.sqrt(math.fabs(solution[j])))
                if i != 0:                          #objective 2
                    "F1 Sphere model unimodal, symmetric, separable"
                    fitness[i] += solution[j] ** 2
                    #fitness[i] -= solution[j] * math.sin(math.sqrt(math.fabs(solution[j])))

        f = np.asarray(fitness)
        self.nnn += 1
        print(self.nnn)
        return f

