

import math
import random
import operator
import functools
from platypus.core import Problem, Solution, EPSILON, Generator
from platypus.types import Real, Binary
from abc import ABCMeta
import numpy as np


class GlobalProblem(Problem):

    def __init__(self, nvars, nobjs, fhat, xlow, xup):
        super(GlobalProblem, self).__init__(nvars, nobjs)
        #self.types[:] = Real(0, 1)
        #--------------------------------- #tvh
        for i in range(nvars):
            self.types[i] = Real(xlow.item(i), xup.item(i))
        # ---------------------------------

        self.fhat = fhat

    def evaluate(self, solution):
        x = np.asarray(solution.variables[:])
        f = []
        for fhat in self.fhat:
            f.append(fhat.predict(x))
        #solution.objectives[:] = f
        solution.objectives[:] = [o.tolist()[0][0] if type(o) == np.ndarray else o for o in f]


class GapProblem(Problem):

    def __init__(self, nvars, nobjs, fhat, xgap, rgap, lb, ub):
        super(GapProblem, self).__init__(nvars, nobjs)
        self.fhat = fhat
        self.set_bounds(xgap, nvars, rgap, lb, ub)

    def set_bounds(self, xgap, nvars, rgap, lb, ub):
        for i in range(nvars):
            #minval = max(0,xgap[i] - rgap)
            #maxval = min(1,xgap[i] + rgap)
            # --------------------------------- #tvh
            minval = max(lb[i],xgap[i] - rgap)
            maxval = min(ub[i], xgap[i] + rgap)
            # ---------------------------------
            self.types[i] = Real(minval, maxval)

    def evaluate(self, solution):
        x = np.asarray(solution.variables[:])
        f = []
        for fhat in self.fhat:
            f.append(fhat.predict(x))
        #solution.objectives[:] = f
        solution.objectives[:] = [z.tolist()[0][0] if type(z) == np.ndarray else z for z in f]

class CustomGenerator(Generator):

    def __init__(self, popsize):
        super(CustomGenerator, self).__init__()
        self.popsize = popsize
        self.iter = 0
        self.solutions = []

    def create(self, problem, nd_solutions):
        # Case 1 - Number of Nd Solutions are more than PopSize
        (N, l) = nd_solutions.shape
        if N >= self.popsize:
            indices = np.random.choice(N, self.popsize, replace=False)
            for i in indices:
                solution = Solution(problem)
                solution.variables = list(nd_solutions[i,:])
                self.solutions.append(solution)
        # Case 2 - When number of nd sols are less that popsize
        else:
            for i in range(N):
                solution = Solution(problem)
                solution.variables = list(nd_solutions[i,:])
                self.solutions.append(solution)
            for i in range(N, self.popsize):
                solution = Solution(problem)
                solution.variables = [x.rand() for x in problem.types]
                self.solutions.append(solution)

    def generate(self, problem):
        solution = self.solutions[self.iter]
        self.iter+=1
        return solution
