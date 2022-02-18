
from typing import Tuple
import numpy as np

from MFEA_lib.model.mfea import AbstractModel
from ..operators import Crossover, Mutation, Selection
from ..tasks.function import AbstractFunc
from ..EA import *
import sys
import matplotlib.pyplot as plt
import random

class MFEA1(AbstractModel):
    def compile(self, tasks: list[AbstractFunc], crossover: Crossover.SBX_Crossover(nc = 2), mutation: Mutation.GaussMutation, selection: Selection.ElitismSelection, *args, **kwargs):
        
        return super().compile(tasks, crossover, mutation, selection, *args, **kwargs)
    
    def findParentSameSkill(self, subpop: SubPopulation, ind):
        ind2 = ind 
        while ind2 is ind: 
            ind2 = subpop.__getRandomItems__(size= 1)[0]
        
        return ind2 

    def fit(self,num_generations, num_inds_each_task: list, rmp = 0.3, bound = [0, 1], evaluate_initial_skillFactor = False,
     log_oneline = False, num_epochs_printed = 20, *args, **kwargs):
        

        population = Population(
            num_inds_each_task,
            dim = self.dim_uss, 
            bound = bound,
            list_tasks = self.tasks,
            evaluate_initial_skillFactor = evaluate_initial_skillFactor
        ) 
        
        self.history_cost.append([ind.fcost for ind in population.get_solves()])


        for epoch in range(num_generations):

            # init offspring pop
            offspring = Population(
                nb_inds_tasks= [0] * len(self.tasks),
                dim =  self.dim_uss, 
                bound = bound, 
                list_tasks= self.tasks,
                # evaluate_initial_skillFactor = evaluate_initial_skillFactor
            )

            # create offspring pop
            while len(offspring) < len(population): 
                # choose parent 
                pa, pb = population.__getRandomInds__(size= 2) 

                # crossover 
                if pa.skill_factor == pb.skill_factor or np.random.rand() < rmp: 
                    oa, ob = self.crossover(pa, pb) 
                    oa.skill_factor, ob.skill_factor = np.random.choice([pa.skill_factor, pb.skill_factor], size= 2, replace= True)
                else: 
                    pa1 = self.findParentSameSkill(population[pa.skill_factor], pa) 
                    oa, _ = self.crossover(pa, pa1) 
                    oa.skill_factor = pa.skill_factor

                    pb1 = self.findParentSameSkill(population[pb.skill_factor], pb) 
                    ob, _ = self.crossover(pb, pb1) 
                    ob.skill_factor = pb.skill_factor 
                
                # mutation 
                #NOTE: no mutation 
            

                # eval and append # addIndividual already has eval  
                offspring.__addIndividual__(oa) 
                offspring.__addIndividual__(ob) 


            
            # merge offspring and population
            
            population = population + offspring 

            # selection 
            self.selection(population, num_inds_each_task)

            # save history 
            self.history_cost.append([ind.fcost for ind in population.get_solves()])
            
            # print 
    
            if (epoch + 1) % ( num_epochs_printed) == 0:
                if log_oneline == True:
                    sys.stdout.write('\r')
                sys.stdout.write('Epoch [{}/{}], [%-20s] %3d%% ,func_val: {}'
                    .format(epoch + 1, num_generations,self.history_cost[-1])
                    % ('=' * ((epoch + 1) // (num_generations // 20)) + '>' , (epoch + 1) * 100 // num_generations)
                    )
                if log_oneline == False:
                    print("\n")
                sys.stdout.flush()
        
        print("End")

        # solve 
        self.last_pop = population 
        return self.last_pop.get_solves() 
