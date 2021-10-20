#!/usr/bin/env python
import numpy as np
from pySOT.optimization_problems import *
#from mo_utils import ND_Front                          #see explanation for variable self.pf


from pandas import DataFrame as df
import pandas as pd
import sampling
from calculate import Calculation

class optimization_problem_class(OptimizationProblem):
    def __init__(self, sample, restart, content_table, doe_sampling_size, design_variables, dir_set, case, calculation_type, jobs_parallel, sampling_size, results, feasibility_assessment):

        domain = sample.initialize_design_variables(restart)
        lb, ub = sample.get_dv_min_max(domain)

        self.results = results
        self.sampling_length = sampling_size
        self.restart = restart
        self.calculation_type = calculation_type
        self.case = case
        self.design_variables = design_variables
        self.dir_set = dir_set
        self.content_table = content_table
        self.doe_sampling_size = doe_sampling_size
        self.jobs_parallel = jobs_parallel
        self.feasibility_assessment = feasibility_assessment

        self.dim = dim = len(domain)                        # number of design variables -- number of dimensions
        self.nobj = nobj = len(content_table.objectitves)   # number of objective functions
        self.dv_name = content_table.df_dvs.columns.values  # name of design variables
        self.lb = lb                                        # lower boundaries for each design variable
        self.ub = ub                                      # upper boundaries for each design variable
        self.info = str(dim) + "-dimensional " + " with " + str(nobj) + " objectives optimized by GOMORS"   # Problem information

        self.integer = []                                   # Integer variables
        self.int_var = []
        self.continuous = np.arange(0, dim)                 # Continuous variables
        self.cont_var = np.arange(0, dim)
        self.pf = None                                      # If it is possible to plot the paretofrontier in a 2D-view: See GOMORS-example e.g. test_problem.py -> class ZDT1 --> function paretofront()


        self.sample = 'sampling'
        self.sample_int = 0

    def objfunction(self, solution):

        solution_work = solution[:]

        if solution_work.shape[0] != self.dim:
            raise ValueError('Dimension mismatch')

        self.sample_int += 1
        # write
        if (self.doe_sampling_size - self.sample_int) >= 0:
            print('>>> GOMORS DOE-sample-nr.: ' + str(self.sample_int))
            self.sample_generation = 'gomors DOE-sampling {}'.format(self.sample_int)
        else:
            print('>>> GOMORS sample point-nr.: ' + str(self.sample_int))
            self.sample_generation = 'gomors sample {}'.format(self.sample_int)


        # put solution(sample point for evaluation) into DataFrame
        '''solution.shape = (1,solution.shape[0])                  #convert candidate (called solution) for DataFrame '''
        solution_work.shape = (1,solution_work.shape[0])                  #convert candidate (called solution) for DataFrame
        '''self.df_X = df(solution, columns=list(self.dv_name))    #DataFrame of candidate to evaluate '''
        self.df_X = df(solution_work, columns=list(self.dv_name))    #DataFrame of candidate to evaluate

        sampl = sampling.sampling_class(design_variables=self.design_variables, dir_set=self.dir_set, sampling_length=self.sampling_length, case=self.case)
        sampl.set_self_restart(False)       #ToDo tvh uerberpruefen ob das nicht doch self.restart sein sollte
        sampl.set_sample(self.df_X)

        if self.feasibility_assessment == True:
            df_full_sampling_infeasibility, infeasibility_factors = sampl.check_optimization_input(self, self.sample_int)   #check for infeasibility of candidate
        else:
            df_full_sampling_infeasibility = pd.concat(
                [pd.DataFrame(float(0.0), index=np.arange(len(sampl.df_sampling)), columns=["feasibility factor"]),
                 sampl.df_sampling], axis=1)

        # add new sample point for evaluation to content_table
        self.content_table.append_dvs(self.df_X)
        self.content_table.append_feasibility(df_full_sampling_infeasibility)
        self.content_table.append_generation(self.sample_generation, self.df_X)
        self.content_table.append_status(self.df_X)
        self.content_table.append_objectives(self.df_X)
        self.content_table.append_constraints(self.df_X)
        self.content_table.iteration = self.sample_generation
        self.content_table.create_full_sampling()

        # start evaluation
        calculation_doe = Calculation(self.calculation_type, self.jobs_parallel, self.case, self.design_variables,
                                      self.content_table, self.restart, gen=None)
        self.results, self.dirs, self.content_table = calculation_doe.start_doe(self.results, self.sample_generation)

        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter("00_full_results.xlsx")
        #writer = pd.ExcelWriter("00_full_results.xlsx", engine='xlsxwriter')

        # Convert the dataframe to an XlsxWriter Excel object.
        self.content_table.df_content_table.to_excel(writer, sheet_name='Sheet1')

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()
        writer.close()

        #write result to Excelsheet
        #self.content_table.df_content_table.to_excel("00_full_results.xlsx")

        # return evaluated objective values to gomors
        f_tmp = self.content_table.df_sampling_objectives.values.tolist()[-1]     # take objective value from content_table
        f = np.asarray(f_tmp)                                                   # convert to correct output-format
        solution = list(solution)

        if self.sample_int == 20:
            print('20. sample')

        return f
