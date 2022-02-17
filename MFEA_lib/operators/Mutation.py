import numpy as np

from MFEA_lib.tasks.function import AbstractFunc
from ..EA import Individual

class AbstractMutation():
    def __init__(self, *arg, **kwargs):
        pass
    def __call__(self, p: Individual, *arg, **kwargs) -> Individual:
        pass
    def getInforTasks(self, tasks: list[AbstractFunc]):
        self.dim_uss = max([t.dim for t in tasks])
        pass
class NoMutation(AbstractMutation):
    def __call__(self, p: Individual, *arg, **kwargs) -> Individual:
        return p
    
class Polynomial_Mutation(AbstractMutation):
    '''
    p in [0, 1]^n
    '''
    def __init__(self, nm = 15, pm = None, *arg, **kwargs):
        '''
        nm: parameters of Polynomial_mutation
        pm: prob mutate of Polynomial_mutation
        '''
        self.nm = nm
        self.pm = pm
    
    def __call__(self, ind: Individual, *arg, **kwargs) -> Individual:
        if self.pm is None:
            self.pm = 1/ind.__dim__

        idx_mutation = np.where(np.random.rand(ind.__dim__) <= self.pm)[0]

        #NOTE 
        u = np.zeros((ind.__dim__,)) + 0.5
        u[idx_mutation] = np.random.rand(len(idx_mutation))

        delta = np.where(u < 0.5,
            # delta_l
            (2*u)**(1/(self.nm + 1)) - 1,
            # delta_r
            1 - (2*(1-u))**(1/(self.nm + 1))
        )

        new_ind = Individual(
            genes = np.where(delta < 0,
                # delta_l: ind -> 0
                ind + delta * ind,
                # delta_r: ind -> 1
                ind + delta * (1 - ind)
            )
        )
        new_ind.skill_factor = ind.skill_factor
        return new_ind
    
class GaussMutation(AbstractMutation):
    '''
    p in [0, 1]^n
    '''
    def __init__(self, scale = 0.1, *arg, **kwargs):
        self.scale = scale
    
    def __call__(self, ind: Individual, *arg, **kwargs) -> Individual:
        if self.pm is None:
            self.pm = 1/ind.__dim__

        idx_mutation = np.where(np.random.rand(ind.__dim__) <= self.pm)[0]
        
        mutate_genes = ind[idx_mutation] + np.random.normal(0, self.scale, size = len(idx_mutation))

        mutate_genes = np.where(mutate_genes > 1, ind[idx_mutation] + np.random.rand(len(idx_mutation)) * (1 - ind[idx_mutation]), mutate_genes)
        mutate_genes = np.where(mutate_genes < 0, np.random.rand(len(idx_mutation)) * ind[idx_mutation], mutate_genes)
        
        new_genes = ind.genes
        new_genes[idx_mutation] = mutate_genes
        new_ind = Individual(new_genes)
        new_ind.skill_factor = ind.skill_factor
        return new_ind

class GMDScale(AbstractMutation):
    #TODO
    pass