from asyncio import tasks
from curses.ascii import CR
from typing import Tuple
import numpy as np
from ..operators import Crossover, Mutation, Selection
from ..tasks.function import AbstractFunc
from ..EA import *
import sys
import matplotlib.pyplot as plt
import random

class AbstractModel():
    def __init__(self, seed = None) -> None:
        # initial history of factorial cost
        self.history_cost: list = []
        self.solve: list[Individual]
        if seed is None:
            pass
        else:
            # FIXME not work
            np.random.seed(seed)
            random.seed(seed)

    def render_history(self, shape: Tuple[int, int], title = "", yscale = None, ylim: list[float, float] = None):
        fig = plt.figure(figsize= (shape[1]* 6, shape[0] * 5))
        fig.suptitle(title, size = 20)
        fig.set_facecolor("white")

        np_his_cost = np.array(self.history_cost)
        for i in range(np_his_cost.shape[1]):
            plt.subplot(shape[0], shape[1], i+1)

            plt.plot(np.arange(np_his_cost.shape[0]), np_his_cost[:, i])

            plt.title(self.tasks[i].name)
            plt.xlabel("Generations")
            plt.ylabel("Factorial Cost")
            
            if yscale is not None:
                plt.yscale(yscale)
            if ylim is not None:
                plt.ylim(bottom = ylim[0], top = ylim[1])
                
        plt.show()
        return fig
    
    def compile(self, 
        tasks: list[AbstractFunc], 
        crossover: Crossover.AbstractCrossover, 
        mutation: Mutation.AbstractMutation, 
        selection: Selection.AbstractSelection,
        *args, **kwargs):
        
        self.tasks = tasks
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection

        # get info from tasks
        self.dim_uss = max([t.dim for t in tasks])
        self.crossover.getInforTasks(tasks)
        self.mutation.getInforTasks(tasks)
        self.selection.getInforTasks(tasks)

    
    def fit(self, *args, **kwargs):
        pass

class MFEA_1(AbstractModel):
    def compile(self, rmp, tasks: list[AbstractFunc], crossover: Crossover.SBX_Crossover(nc = 2), mutation: Mutation.GaussMutation, selection: Selection.ElitismSelection, *args, **kwargs):
        self.rmp = rmp 
        return super().compile(tasks, crossover, mutation, selection, *args, **kwargs)
    
    def findParentSameSkill(self, subpop: SubPopulation, ind):
        ind2 = ind 
        while ind2 == ind: 
            ind2 = np.random.choice(np.arange(len(subpop)), size= (1))
            ind2 = int(ind2) 
        
        return ind2 

    def fit(self,num_generations, num_inds_each_task: list, bound = [0, 1], evaluate_initial_skillFactor = False, *args, **kwargs):
        
        dim_uss = np.max([t.dim for t in self.tasks])

        population = Population(num_inds_each_task, dim_uss, bound, self.tasks, evaluate_initial_skillFactor) 
        
        self.history_cost = np.append(self.history_cost, [[np.min(np.array([ind.fcost for ind in population[task].inds])) for task in range(len(self.tasks))]])

        for epoch in range(num_generations):
            # init offspring pop
            offspring = Population(num_inds_each_task= np.zeros(shape= (len(self.tasks,)), dtype= int),dim =  dim_uss, bound = bound, list_tasks= self.tasks)

            # create offspring pop
            while len(offspring) < len(population): 
                # choose parent 
                skf_pa, skf_pb = np.random.choice(np.arange(len(self.tasks)), size= (2,)) 
                if skf_pa == skf_pb:
                    index_pa, index_pb = np.random.choice(np.arange(len(population[skf_pa])), size= (2, ), replace= False) 
                else: 
                    index_pa = np.random.choice(np.arange(len(population[skf_pa])), size= (1,)) 
                    index_pb = np.random.choice(np.arange(len(population[skf_pb])), size= (1,)) 

                
                index_pa, index_pb = int(index_pa), int(index_pb) 
                
                # crossover 
                if skf_pa == skf_pb: 
                    oa, ob = self.crossover(population[skf_pa].inds[index_pa], population[skf_pb].inds[index_pb])
                    oa.skill_factor= ob.skill_factor = skf_pa 
                    
                elif np.random.uniform() < self.rmp: 
                    oa, ob = self.crossover(population[skf_pa].inds[index_pa], population[skf_pb].inds[index_pb])
                    oa.skill_factor, ob.skill_factor = np.random.choice([skf_pa, skf_pb], size= 2, replace= True) 
                else: 
                    index_pa1 = self.findParentSameSkill(population[skf_pa], index_pa) 
                    oa, _ = self.crossover(population[skf_pa].inds[index_pa], population[skf_pa].inds[index_pa1]) 
                    oa.skill_factor = skf_pa 

                    index_pb1 = self.findParentSameSkill(population[skf_pb], index_pb) 
                    ob,_ = self.crossover(population[skf_pb].inds[index_pb], population[skf_pb].inds[index_pb1])
                    ob.skill_factor = skf_pb 
                    

                # mutation 

                # eval 
                oa.eval(self.tasks[oa.skill_factor])
                ob.eval(self.tasks[ob.skill_factor])

                # append 
                offspring[oa.skill_factor].inds.append(oa) 
                offspring[ob.skill_factor].inds.append(ob) 

            
            # merge offspring and population
            
            # selection 

            # save history 

            # print 

        
        print("End")




class MFEA_base(AbstractModel):
    def compile(self, tasks: list[AbstractFunc], crossover: Crossover.SBX_Crossover, mutation: Mutation.Polynomial_Mutation, selection: Selection.ElitismSelection, *args, **kwargs):
        return super().compile(tasks, crossover, mutation, selection, *args, **kwargs)
    
    def fit(self, nb_generations, rmp = 0.3, nb_inds_each_task = 100, bound_pop = [0, 1], evaluate_initial_skillFactor = True,
            log_oneline = False, num_epochs_printed = 20, *args, **kwargs):
        super().fit(*args, **kwargs)

        # initialize population
        population = Population(
            nb_inds_tasks = [nb_inds_each_task] * len(tasks), 
            dim = self.dim_uss,
            bound= bound_pop,
            list_tasks= tasks,
            evaluate_initial_skillFactor = evaluate_initial_skillFactor
        )

        # save history
        self.history_cost.append([ind.fcost for ind in population.get_solves()])
        
        for epoch in range(nb_generations):
            
            # initial offspring_population of generation
            offsprings = Population(
                nb_inds_tasks = [0] * len(tasks), 
                dim = self.dim_uss,
                bound= bound_pop,
                list_tasks= tasks,
                evaluate_initial_skillFactor = evaluate_initial_skillFactor
            )

            while len(offsprings) < len(population):
                pa, pb = population.__getRandomInds__(2)
                
