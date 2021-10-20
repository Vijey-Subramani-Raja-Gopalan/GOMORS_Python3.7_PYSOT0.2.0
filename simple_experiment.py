import time
start = time.time()
from gomors_sync_strategies import MoSyncStrategyNoConstraints
from gomors_adaptive_sampling import EvolutionaryAlgorithm
from test_problem import *
#from pySOT import SymmetricLatinHypercube, RBFInterpolant, CubicKernel, LinearTail
from pySOT.experimental_design import SymmetricLatinHypercube           #VJ
from pySOT.surrogate import RBFInterpolant, CubicKernel, LinearTail     #VJ
from pySOT.surrogate import GPRegressor
from poap.controller import SerialController, ThreadController, BasicWorkerThread
from archiving_strategies import NonDominatedArchive, EpsilonArchive
import numpy as np
from pySOT.optimization_problems import OptimizationProblem
import os.path
import logging

#from Townbrook import *


def main():
    #start = time.clock()    #tvh

    print("Sampling method: Mixed")
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    if os.path.exists("./logfiles/test_simple.log"):
        os.remove("./logfiles/test_simple.log")
    logging.basicConfig(filename="./logfiles/test_simple.log",
                        level=logging.INFO)     # for very detailed output for diagnostic purposes;  level:Confirmation that things are working as expected

    nthreads = 4  #original function evaluations per iteration
    #maxeval = 500
    maxeval = 40  #max no. of original function evaluations
    nsamples = 4        #nthreads changed

    print(("\nNumber of threads: " + str(nthreads)))
    print(("Maximum number of evaluations: " + str(maxeval)))
    print("Sampling method: Mixed")
    print("Experimental design: Symmetric Latin Hypercube")
    print("Surrogate: Cubic RBF")

    data = Yannis_Problem(nobj=2)
    print(data.dim)
    num = 1
    epsilons = [0.05, 0.05]
    # Create a strategy and a controller
    #an event-driven framework for building and combining asynchronous optimization strategies using two components:
    # controllers and strategies
    #The controller is capable of asking workers to run function evaluations and the strategy decides where to evaluate next
    #a threaded controller: that dispatches work to a queue of workers where each worker is able
    #to handle evaluation and kill requests. The requests are asynchronous in the sense that the workers are not required
    #to complete the evaluation or termination requests. The worker is forced to respond to evaluation requests, but may
    #ignore kill requests. When receiving an evaluation request, the worker should either attempt the evaluation or mark
    #the record as killed. The worker sends status updates back to the controller by updating the relevant record.
    controller = ThreadController()
    #controller = SerialController(data.objfunction)
    #Strategy:  it is responsible for choosing new evaluations, killing evaluations, and terminating the optimization run when a stopping criteria is reached

    controller.strategy = \
        MoSyncStrategyNoConstraints(
            worker_id=0, data=data,
            maxeval=maxeval, nsamples=nsamples,
            # exp_design=SymmetricLatinHypercube(dim=data.dim, npts=2*(data.dim+1)),
            exp_design=SymmetricLatinHypercube(dim=data.dim, num_pts=32),
            response_surface=RBFInterpolant(dim=data.dim, kernel=CubicKernel(), tail=LinearTail(dim=data.dim)),
            # sampling_method=EvolutionaryAlgorithm(data,epsilons=epsilons, cand_flag=1), archiving_method=EpsilonArchive(size_max=200,epsilon=epsilons))
            sampling_method=EvolutionaryAlgorithm(data, epsilons=epsilons, cand_flag=None),
            archiving_method=EpsilonArchive(size_max=2000, epsilon=epsilons))



    
    # Launch the threads and give them access to the objective function
    for _ in range(nthreads):
        worker = BasicWorkerThread(controller, data.objfunction)
        controller.launch_worker(worker)  #run(worker)
    print(nthreads)
    # Run the optimization strategy
    def merit(r):
        return r.value[0]
    result = controller.run(merit=merit)

    controller.strategy.save_plot(num)

    # --------------------------------- #tvh
    end = time.time()



    # ---------------------------------

if __name__ == '__main__':
    main()
