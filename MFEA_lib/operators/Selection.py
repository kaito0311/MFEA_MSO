from typing import Any
import numpy as np
from ..EA import *

class AbstractSelection():
    def __init__(self) -> None:
        pass
    def __call__(self, population:Population, *args, **kwds) -> None:
        pass

class ElitismSelection(AbstractSelection):
    def __call__(self, population:Population, num_inds_each_task: list, *args, **kwds) -> list:
        list_selected = []
        for idx_subpop, subpop in enumerate(population):
            N_i = min(num_inds_each_task[idx_subpop], len(subpop))
        
            idx_selected_inds = np.where(subpop.scalar_fitness > 1/(N_i + 1))[0]
            subpop.select(idx_selected_inds)

            list_selected.append(idx_selected_inds)
        return list_selected
