
from os import popen
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
    
    def fit(self, max_inds_each_task: list, min_inds_each_task: list, max_eval_each_task: list, H = 30, bound = [0, 1], evaluate_initial_skillFactor = False,
        log_oneline = False, num_epochs_printed = 20, *args, **kwargs): 

            current_inds_each_task = np.copy(max_inds_each_task) 
            eval_each_task = np.zeros_like(max_eval_each_task)

            population = Population(
                nb_inds_tasks= max_eval_each_task, 
                dim = self.dim_uss, 
                bound = bound, 
                list_tasks= self.tasks, 
                evaluate_initial_skillFactor= evaluate_initial_skillFactor
            )

            memory_H = []
        