from typing import Tuple
import numpy as np

from MFEA_lib.tasks.function import AbstractFunc
from ..EA import Individual

class AbstractCrossover():
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, pa: Individual, pb: Individual, *args, **kwargs) -> Tuple[Individual, Individual]:
        pass
    def getInforTasks(self, tasks: list[AbstractFunc]):
        self.dim_uss = max([t.dim for t in tasks])
        pass
    
    def update(self, *args, **kwargs) -> None:
        pass

class SBX_Crossover(AbstractCrossover):
    '''
    pa, pb in [0, 1]^n
    '''
    def __init__(self, nc = 15, *args, **kwargs):
        self.nc = nc
    def __call__(self, pa: Individual, pb: Individual, *args, **kwargs) -> Tuple[Individual, Individual]:
        u = np.random.rand(self.dim_uss)

        # ~1
        beta = np.where(u < 0.5, (2*u)**(1/(self.nc +1)), (2 * (1 - u))**(-1 / (1 + self.nc)))

        #like pa
        oa = Individual(np.clip(0.5*((1 + beta) * pa.genes + (1 - beta) * pb.genes), 0, 1))
        #like pb
        ob = Individual(np.clip(0.5*((1 - beta) * pa.genes + (1 + beta) * pb.genes), 0, 1))

        if pa.skill_factor == pb.skill_factor:
            idx_swap = np.where(np.random.rand(self.dim_uss) < 0.5)[0]
            oa.genes[idx_swap], ob.genes[idx_swap] = ob.genes[idx_swap], oa.genes[idx_swap]
        
        return oa, ob

class newSBX(AbstractCrossover):
    '''
    pa, pb in [0, 1]^n
    '''
    #TODO
    pass