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


class MFEA1(AbstractModel):
    pass

class SA_MFEA(AbstractModel):
    pass