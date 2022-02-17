from typing import Any
import numpy as np
from ..EA import *

class AbstractSelection():
    def __init__(self) -> None:
        pass
    def __call__(self, population:Population, nb_inds_tasks:list, *args, **kwds) -> list:
        pass
    def getInforTasks(self, tasks: list[AbstractFunc]):
        self.dim_uss = max([t.dim for t in tasks])
        pass
class ElitismSelection(AbstractSelection):
    def __call__(self, population:Population, nb_inds_tasks: list, *args, **kwds) -> list:
        ls_idx_selected = []
        for idx_subpop, subpop in enumerate(population):
            N_i = min(nb_inds_tasks[idx_subpop], len(subpop))
        
            idx_selected_inds = np.where(subpop.scalar_fitness > 1/(N_i + 1))[0].tolist()
            subpop.select(idx_selected_inds)

            ls_idx_selected.append(idx_selected_inds)
        return ls_idx_selected
