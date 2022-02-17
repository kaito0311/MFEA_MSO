import random
import numpy as np
from .tasks.function import AbstractFunc

class Individual:
    def __init__(self, genes) -> None: 
        self.genes: np.ndarray = genes
        self.skill_factor: int = None
        self.fcost: float = None

    def eval(self, task: AbstractFunc) -> None:
        self.fcost = task.func(self.genes)

    def __repr__(self) -> str:
        return 'Genes: {}\nSkill_factor: {}'.format(str(self.genes), str(self.skill_factor))
    def __str__(self) -> str:
        return str(self.genes)
    def __dim__(self):
        return len(self.genes)
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
        self.ls_inds = [
            Individual(np.random.uniform(bound[0], bound[1], size= (dim, )))
            for i in range(num_inds)
        ]
        for i in range(num_inds):
            self.ls_inds[i].skill_factor = skill_factor
            self.ls_inds[i].fcost = task(self.ls_inds[i].genes)

        self.factorial_rank: np.ndarray = None
        self.scalar_fitness: np.ndarray = None
        self.update_rank()
        
    def __len__(self):
        return len(self.ls_inds)

    def __getitem__(self, index):
        return self.ls_inds[index]

    def __getRandomItem__(self, size):
        return self.ls_inds[np.random.choice(len(self), size)]

    def __addIndividual__(self, individual: Individual, update_rank = False):
        self.ls_inds.append(individual)
        if update_rank:
            self.update_rank()

    def update_rank(self):
        '''
        Update `factorial_rank` and `scalar_fitness`
        '''
        self.factorial_rank = np.argsort(np.argsort([ind.fcost for ind in self.ls_inds])) + 1
        self.scalar_fitness = 1/self.factorial_rank

    def select(self, index_selected_inds):
        self.ls_inds = self.ls_inds[index_selected_inds]
        self.factorial_rank = self.factorial_rank[index_selected_inds]
        self.scalar_fitness = self.scalar_fitness[index_selected_inds]
        
    def getSolveInd(self):
        return self.ls_inds[np.where(self.factorial_rank == 1)[0]]

class Population:
    def __init__(self, num_inds_tasks: list, dim, bound = [0, 1], list_tasks:list[AbstractFunc] = [], 
        evaluate_initial_skillFactor = False) -> None:

        assert bound[0] < bound[1] and len(bound) == 2
        assert len(num_inds_tasks) == len(list_tasks)

        if evaluate_initial_skillFactor:
            self.ls_subPop: list[SubPopulation] = [
                SubPopulation(skf, 0, dim, bound, list_tasks[skf]) for skf in range(len(num_inds_tasks))
            ]

            # TODO 
            pass

        else:
            self.ls_subPop: list[SubPopulation] = [
                SubPopulation(skf, num_inds_tasks[skf], dim, bound, list_tasks[skf]) for skf in range(len(num_inds_tasks))
            ]

        self.nb_tasks = len(list_tasks)

    def __len__(self):
        return np.sum([len(subPop) for subPop in self.ls_subPop])

    def __getitem__(self, index):
        return self.ls_subPop[index]

    def __getRandomIndsTask__(self, idx_task, size):
        return self.ls_subPop[idx_task].__getRandomItem__(size)

    def __getRandomInds__(self, size):
        return [
            self.ls_subPop[np.random.randint(0, self.nb_tasks)].__getRandomItem__(1) 
            for i in range(size)
        ]

    def get_solves(self):
        return [subPop.getSolveInd() for subPop in self.ls_subPop]
