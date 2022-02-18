import random
from re import I
from typing import Type
import numpy as np
from .tasks.function import AbstractFunc
from operator import itemgetter 
class Individual:
    def __init__(self, genes) -> None: 
        self.genes: np.ndarray = genes
        self.skill_factor: int = None
        self.fcost: float = None
        self.dim = len(self)

    def eval(self, task: AbstractFunc) -> None:
        self.fcost = task.func(self.genes)

    def __repr__(self) -> str:
        return 'Genes: {}\nSkill_factor: {}'.format(str(self.genes), str(self.skill_factor))
    def __str__(self) -> str:
        return str(self.genes)
    def __len__(self):
        return len(self.genes)
    def __getitem__(self, index):
        return self.genes[index]

    def __add__(self, other):
        ind = Individual(self[:] + other)
        ind.skill_factor = self.skill_factor
        return ind
    def __sub__(self, other):
        ind = Individual(self[:] - other)
        ind.skill_factor = self.skill_factor
        return ind
    def __mul__(self, other):
        ind = Individual(self[:] * other)
        ind.skill_factor = self.skill_factor
        return ind
    def __truediv__(self, other):
        ind = Individual(self[:] / other)
        ind.skill_factor = self.skill_factor
        return ind
    def __floordiv__(self, other):
        ind = Individual(self[:] // other)
        ind.skill_factor = self.skill_factor
        return ind
    def __mod__(self, other):
        ind = Individual(self[:] % other)
        ind.skill_factor = self.skill_factor
        return ind
    def __pow__(self, other):
        ind = Individual(self[:] ** other)
        ind.skill_factor = self.skill_factor
        return ind

    def __lt__(self, other) -> bool:
        if self.skill_factor == other.skill_factor:
            return self.fcost > other.fcost
        else:
            return False

    def __gt__(self, other) -> bool:
        if self.skill_factor == other.skill_factor:
            return self.fcost < other.fcost
        else:
            return False

    def __le__(self, other)-> bool:
        if self.skill_factor == other.skill_factor:
            return self.fcost >= other.fcost
        else:
            return False

    def __ge__(self, other)-> bool:
        if self.skill_factor == other.skill_factor:
            return self.fcost <= other.fcost
        else:
            return False

    def __eq__(self, other)-> bool:
        if self.skill_factor == other.skill_factor:
            return self.fcost == other.fcost
        else:
            return False

    def __ne__(self, other)-> bool:
        if self.skill_factor == other.skill_factor:
            return self.fcost != other.fcost
        else:
            return False
        
class SubPopulation:
    def __init__(self, skill_factor, num_inds, dim, bound = [0, 1], task: AbstractFunc = None) -> None:
        self.skill_factor = skill_factor
        self.task = task
        self.dim = dim
        self.ls_inds = [
            Individual(np.random.uniform(bound[0], bound[1], size= (dim, )))
            for i in range(num_inds)
        ]
        for i in range(num_inds):
            self.ls_inds[i].skill_factor = skill_factor
            self.ls_inds[i].fcost = self.task(self.ls_inds[i].genes)

        self.factorial_rank: np.ndarray = None
        self.scalar_fitness: np.ndarray = None
        self.update_rank()
        
    def __len__(self): 
        return len(self.ls_inds)

    def __getitem__(self, index):
        try:
            return self.ls_inds[index]
        except:
            if type(index) == list:
                return [self.ls_inds[i] for i in index]
            else:
                raise TypeError('Int, Slice or list[int], not ' + type(index))

    def __getRandomItems__(self, size:int = None, replace:bool = False):
        if size == 0:
            return []
        if size == None:
            return self.ls_inds[np.random.choice(len(self), size = None, replace= replace)]
        return [self.ls_inds[idx] for idx in np.random.choice(len(self), size= size, replace= replace).tolist()]


    def __addIndividual__(self, individual: Individual, update_rank = False):
        if individual.fcost is None:
            individual.fcost = self.task(individual.genes)
        self.ls_inds.append(individual)
        if update_rank:
            self.update_rank()

    def __add__(self, other):
        assert self.task == other.task, 'Cannot add 2 sub-population do not have the same task'
        assert self.dim == other.dim, 'Cannot add 2 sub-population do not have the same dimensions'
        UnionSubPop = SubPopulation(
            skill_factor = self.skill_factor,
            num_inds= 0,
            dim= self.dim,
            task= self.task
        )
        UnionSubPop.ls_inds = self.ls_inds + other.ls_inds
        UnionSubPop.update_rank()
        return UnionSubPop

    def update_rank(self):
        '''
        Update `factorial_rank` and `scalar_fitness`
        '''
        self.factorial_rank = np.argsort(np.argsort([ind.fcost for ind in self.ls_inds])) + 1
        self.scalar_fitness = 1/self.factorial_rank

    def select(self, index_selected_inds: list):
        self.ls_inds = [self.ls_inds[idx] for idx in index_selected_inds]

        self.factorial_rank = self.factorial_rank[index_selected_inds]
        self.scalar_fitness = self.scalar_fitness[index_selected_inds]
        
    def getSolveInd(self):
        return self.ls_inds[int(np.where(self.factorial_rank == 1)[0])]

class Population:
    def __init__(self, nb_inds_tasks: list, dim, bound = [0, 1], list_tasks:list[AbstractFunc] = [], 
        evaluate_initial_skillFactor = False) -> None:

        assert bound[0] < bound[1] and len(bound) == 2
        assert len(nb_inds_tasks) == len(list_tasks)

        if evaluate_initial_skillFactor:
            # empty population
            self.ls_subPop: list[SubPopulation] = [
                SubPopulation(skf, 0, dim, bound, list_tasks[skf]) for skf in range(len(nb_inds_tasks))
            ]

            # list individual (don't have skill factor)
            ls_inds = [
                Individual(np.random.uniform(bound[0], bound[1], size= (dim, )))
                for i in range(np.sum(nb_inds_tasks))
            ]
            # matrix factorial cost and matrix rank
            matrix_cost = np.array([np.apply_along_axis(t.func, 1, np.array([ind.genes for ind in ls_inds])) for t in list_tasks]).T
            matrix_rank_pop = np.argsort(np.argsort(matrix_cost, axis = 0), axis = 0) 

            count_inds = np.zeros((len(list_tasks),))
            
            #condition flag
            condition = False
            
            while not condition:
                # random task do not have enough individual
                idx_task = np.random.choice(np.where(count_inds < nb_inds_tasks)[0])
                # get best individual of task
                idx_ind = np.argsort(matrix_rank_pop[:, idx_task])[0]

                # set skill factor
                ls_inds[idx_ind].skill_factor = idx_task
                # add individual
                self.__addIndividual__(ls_inds[idx_ind])

                # set worst rank for ind
                matrix_rank_pop[idx_ind] = len(ls_inds) + 1
                count_inds[idx_task] += 1

                condition = np.all(count_inds == nb_inds_tasks)             

            for i in range(len(list_tasks)):
                self.ls_subPop[i].update_rank()
        else:
            self.ls_subPop: list[SubPopulation] = [
                SubPopulation(skf, nb_inds_tasks[skf], dim, bound, list_tasks[skf]) for skf in range(len(nb_inds_tasks))
            ]

        self.ls_tasks = list_tasks
        self.nb_tasks = len(list_tasks)
        self.dim_uss = dim
        self.bound = bound

    def __len__(self):
        return np.sum([len(subPop) for subPop in self.ls_subPop])

    def __getitem__(self, index):
        return self.ls_subPop[index]

    def __getRandomIndsTask__(self, idx_task, size, replace: bool = False):
        return self.ls_subPop[idx_task].__getRandomItems__(size, replace)

    def __getRandomInds__(self, size: int = None, replace: bool = False):
        if size == None:
            return self.ls_subPop[np.random.randint(0, self.nb_tasks)].__getRandomItems__(None, replace) 
        else:
            nb_randInds = [0] * self.nb_tasks
            for idx in np.random.choice(self.nb_tasks, size = size, replace= True).tolist():
                nb_randInds[idx] += 1

            res = []
            for idx, nb_inds in enumerate(nb_randInds):
                res += self.ls_subPop[idx].__getRandomItems__(size = nb_inds, replace= replace)

            return res
        
    def __addIndividual__(self, individual:Individual, update_rank = False):
        self.ls_subPop[individual.skill_factor].__addIndividual__(individual, update_rank)

    def get_solves(self):
        return [subPop.getSolveInd() for subPop in self.ls_subPop]

    def __add__(self, other):
        assert self.nb_tasks == other.nb_tasks and self.dim_uss == other.dim_uss and self.bound == other.bound
        newPop = Population(
            nb_inds_tasks= [0] * self.nb_tasks,
            dim = self.dim_uss,
            bound = self.bound,
            list_tasks= self.ls_tasks
        )
        newPop.ls_subPop = [
            self.ls_subPop[idx] + other.ls_subPop[idx]
            for idx in range(self.nb_tasks)
        ]
        return newPop