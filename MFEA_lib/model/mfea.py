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
        self.history_cost: np.ndarray
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

        for i in range(self.history_cost.shape[1]):
            plt.subplot(shape[0], shape[1], i+1)

            plt.plot(np.arange(self.history_cost.shape[0]), self.history_cost[:, i])

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

        # initial history of factorial cost
        self.history_cost = np.empty((0, len(tasks)), float)
    
    def fit(self, *args, **kwargs):
        pass