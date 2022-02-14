from asyncio import SubprocessProtocol
import numpy as np
from .tasks.function import AbstractFunc

class Population:
    def __init__(self, num_inds_each_task: list, dim, bound = [0, 1], list_tasks:list[AbstractFunc] = [], 
        evaluate_initial_skillFactor = False) -> None:

        assert bound[0] < bound[1] and len(bound) == 2
        assert len(num_inds_each_task) == len(list_tasks)

        if evaluate_initial_skillFactor:
            self.pop: list[SubPopulation] = [
                SubPopulation(skf, 0, dim, bound, list_tasks[skf]) for skf in range(len(num_inds_each_task))
            ]

            # TODO 
            pass

        else:
            self.pop: list[SubPopulation] = [
                SubPopulation(skf, num_inds_each_task[skf], dim, bound, list_tasks[skf]) for skf in range(len(num_inds_each_task))
            ]
    


class SubPopulation:
    def __init__(self, skill_factor, num_inds, dim, bound = [0, 1], task: AbstractFunc = None) -> None:
        self.skill_factor = skill_factor
        self.inds = [
            Individual(np.random.uniform(bound[0], bound[1], size= (dim, )))
            for i in range(num_inds)
        ]
        for i in range(num_inds):
            self.inds[i].skill_factor = skill_factor
            self.inds[i].fcost = task(self.inds[i].genes)

        self.factorial_rank: np.ndarray = None
        self.scalar_fitness: np.ndarray = None
        self.update_rank()
        
    def __len__(self):
        return len(self.inds)

    def __getitem__(self, index):
        return self.inds[index]

    def update_rank(self):
        '''
        Update `factorial_rank` and `scalar_fitness`
        '''
        self.factorial_rank = np.argsort(np.argsort([ind.fcost for ind in self.inds])) + 1
        self.scalar_fitness = 1/self.factorial_rank

class Individual:
    def __init__(self, genes) -> None: 
        self.genes: np.ndarray = genes
        self.skill_factor: int = None
        self.fcost: float = None

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






