from typing import Tuple
import numpy as np

from MFEA_lib.model.mfea import AbstractModel
from ..operators import Crossover, Mutation, Selection
from ..tasks.function import AbstractFunc
from ..EA import *
import sys
import matplotlib.pyplot as plt
import random

class Memory:
    def __init__(self, H=5, sigma=0.1):
        self.H = H
        self.index = 0
        self.sigma = sigma
        self.M = np.zeros((H), dtype=float) + 0.5
    

    def random_Gauss(self):
        mean = np.random.choice(self.M)

        # rmp_sampled = np.random.normal(loc=mean, scale=self.varience)
        # mu + sigma * Math.sqrt(-2.0 * Math.log(rand.nextDouble())) * Math.sin(2.0 * Math.PI * rand.nextDouble());
        rmp_sampled = 0
        while rmp_sampled <= 0:
            rmp_sampled = mean + self.sigma * np.sqrt(-2.0 * np.log(np.random.uniform())) * np.sin(2.0 * np.pi * np.random.uniform())

        if rmp_sampled > 1:
            return 1
        return rmp_sampled
    
    def update_M(self, value):
        self.M[self.index] = value
        self.index = (self.index + 1) % self.H



class SA_LSA_MODEL(AbstractModel): 
    def compile(self, tasks: list[AbstractFunc], crossover: Crossover.AbstractCrossover, mutation: Mutation.AbstractMutation, selection: Selection.AbstractSelection, *args, **kwargs):
        return super().compile(tasks, crossover, mutation, selection, *args, **kwargs)
    
    def findParentSameSkill(self, subpop: SubPopulation, ind):
        ind2 = ind 
        while ind2 is ind: 
            ind2 = subpop.__getRandomItems__(size= 1)[0]
        
        return ind2 

    def Update_History_Memory(self, history_memories, S, sigma):
        for i in range((len(self.tasks))):
            j = i + 1
            while j < len(self.tasks):
                if len(S[i][j]) != 0:
                    history_memories[i][j].update_M(
                        np.sum(np.array(sigma[i][j]) * np.array(S[i][j]) ** 2)
                        / np.sum(np.array(sigma[i][j]) * (np.array(S[i][j])) + 1e-10)
                    )

                j += 1

        return history_memories


    def fit(self, max_inds_each_task: list, min_inds_each_task: list, max_eval_each_task: list, H = 30, bound = [0, 1], evaluate_initial_skillFactor = False,
        log_oneline = False, num_epochs_printed = 20, *args, **kwargs): 

            current_inds_each_task = np.copy(max_inds_each_task) 
            eval_each_task = np.zeros_like(max_eval_each_task)

            population = Population(
                nb_inds_tasks= current_inds_each_task, 
                dim = self.dim_uss, 
                bound = bound, 
                list_tasks= self.tasks, 
                evaluate_initial_skillFactor= evaluate_initial_skillFactor
            )

            memory_H = [[Memory(H) for i in range(len(self.tasks))] for j in range(len(self.tasks))]

            while np.sum(eval_each_task) < np.sum(max_eval_each_task):

                S = np.empty((len(self.tasks), len(self.tasks), 0)).tolist() 
                sigma = np.empty((len(self.tasks), len(self.tasks), 0)).tolist()

                offsprings = Population(
                    nb_inds_tasks= [0] * len(self.tasks),
                    dim = self.dim_uss, 
                    bound = bound,
                    list_tasks= self.tasks,
                )

                # create new offspring population 
                while len(offsprings) < len(population): 
                    pa, pb = population.__getRandomInds__(size =2) 

                    if pa.skill_factor > pb.skill_factor: 
                        pa, pb = pb, pa
                    
                    

                    # crossover 
                    if pa.skill_factor == pb.skill_factor: 
                        oa, ob = self.crossover(pa, pb)
                        oa.skill_factor, ob.skill_factor= np.random.choice([pa.skill_factor, pb.skill_factor], size= 2, replace= True) 

                    else: 
                        # create rmp 
                        rmp = memory_H[pa.skill_factor][pb.skill_factor].random_Gauss() 
                        r = np.random.uniform() 
                        if r < rmp: 
                            oa, ob = self.crossover(pa, pb)
                            oa.skill_factor, ob.skill_factor= np.random.choice([pa.skill_factor, pb.skill_factor], size= 2, replace= True) 
                        else: 
                            pa1 = self.findParentSameSkill(population[pa.skill_factor], pa) 
                            oa, _ = self.crossover(pa, pa1) 
                            oa.skill_factor = pa.skill_factor 

                            pb1 = self.findParentSameSkill(population[pb.skill_factor], pb)
                            ob, _ = self.crossover(pb, pb1) 
                            ob.skill_factor = pb.skill_factor 

                        
                    
                    # append and eval 
                    offsprings.__addIndividual__(oa)
                    offsprings.__addIndividual__(ob) 

                    # cal delta
                    if pa.skill_factor != pb.skill_factor:

                        delta = 0 

                        if oa.skill_factor == pa.skill_factor : 
                            if pa.fcost > 0: 
                                delta = np.max([delta, (pa.fcost - oa.fcost) / (pa.fcost)])
                        else: 
                            if pb.fcost > 0: 
                                delta = np.max([delta, (pb.fcost - oa.fcost) / (pb.fcost)]) 
                        
                        if ob.skill_factor == pa.skill_factor:
                            if pa.fcost > 0: 
                                delta = np.max([delta, (pa.fcost - ob.fcost) / (pa.fcost)])    
                        else: 
                            if pb.fcost > 0: 
                                delta = np.max([delta, (pb.fcost - ob.fcost) / (pb.fcost)]) 
                        

                        # update S and sigma 
                        if delta > 0: 
                            S[pa.skill_factor][pb.skill_factor].append(rmp)
                            sigma[pa.skill_factor][pb.skill_factor].append(delta) 
                    
                    eval_each_task[oa.skill_factor] += 1 
                    eval_each_task[ob.skill_factor] += 1 
                

                # update memory H 
                memory_H = self.Update_History_Memory(memory_H, S, sigma) 

                # linear size 

                # merge 
                population = population + offsprings 

                # selection 
                self.selection(population, current_inds_each_task)

                # save history 
                self.history_cost.append([ind.fcost for ind in population.get_solves()])
                
                # print 
        
                if (int(eval_each_task[0] / 100) + 1 - len(self.history_cost)) % (num_epochs_printed) == 0:
                    if log_oneline == True:
                        sys.stdout.write('\r')
                    sys.stdout.write('Epoch [{}/{}], [%-20s] %3d%% ,func_val: {}'
                        .format((int(eval_each_task[0]/100)) + 1, max_eval_each_task[0],self.history_cost[-1])
                        % ('=' * ((np.sum(eval_each_task)) // (np.sum(max_eval_each_task) // 20)) + '>' , (np.sum(eval_each_task) + 1) * 100 // np.sum(max_eval_each_task))
                        )
                    if log_oneline == False:
                        print("\n")
                    sys.stdout.flush()
        
            print("End")

            # solve 
            self.last_pop = population 

            return self.last_pop.get_solves() 



                    



        