from enum import Enum, auto

import numpy as np
from scipy.special import softmax


class Distribution(Enum):
    EVEN = auto()
    LINEAR = auto()
    SOFTMAX_LINEAR = auto()


def crossover(parents, n_offspring, distribution : Distribution = Distribution.EVEN):
    """Sample an offspring set via crossover from a mating pool

    Args:
        parents (numpy.ndarray): An array of shape `n x d_f` containing the mating pool, where `n`
            is the number of possible mates and `d_f` is the feature size, .
        n_offspring (int): The number of offspring.
        distribution (str): Key word to indicate how to sample the parent vectors.
            Choose from ['even', 'linear', 'softmax_linear']; 'even' means sample
            parents with a even probability; 'linear' means sample probability is
            linear to ranking, one scored high has better probability to be
            selected; 'softmax_linear' means the sample probability is exponential
            of linear ranking, steeper than the 'linear', for exploitation stages.
            Defaults to 'even'.

    Returns:
        offspring (numpy.ndarray): an array of shape `n_offspring x d_f` containing the offspring 
            pool.
    """
    d_f = parents.shape[1]
    possible_idxs = set(range(d_f))

    offspring = np.empty((n_offspring, d_f))

    n_inherit = np.ceil(np.random.normal(loc=d_f / 2, scale=d_f / 10, size=n_offspring))
    n_inherit = np.clip(n_inherit, d_f // 5, d_f * 4 // 5)

    for i in range(n_offspring):
        if distribution == Distribution.EVEN:
            p1, p2 = parents[np.random.choice(len(parents), 2, replace=False)]
        elif distribution == Distribution.LINEAR:
            p_ = np.arange(len(parents))[::-1] + 10
            p1, p2 = parents[np.random.choice(len(parents), 2, replace=False, p=p_/p_.sum())]
        elif distribution == Distribution.SOFTMAX_LINEAR:
            p_ = np.arange(len(parents))[::-1] + 10
            p1, p2 = parents[np.random.choice(len(parents), 2, replace=False, p=softmax(p_))]

        p1_idxs = np.random.choice(d_f, n_inherit[i], replace=False)
        p2_idxs = list(possible_idxs.difference(p1_idxs))

        offspring[i, p1_idxs] = p1[p1_idxs]
        offspring[i, p2_idxs] = p2[p2_idxs]

    return offspring

def fitness_sum(element):
    """
    Test fitness function.
    """
    return np.sum(element)

def mutation(offspring: np.ndarray, n_mutation: int = 1, p_mutation: float = 0.5):
    """Mutate offspring in the pool with a given probability

    Args:
        offspring_crossover (numpy.ndarray): the offspring pool before mutation.
        n_mutation (int): the number of bits to flip when mutating
        p_mutation (float): the probablity with which to mutate a child

    Returns:
        offspring_crossover (numpy.ndarray): An array represents the offspring
            pool after mutation.
    """
    offspring = offspring.astype(bool)
    fp_length = offspring.shape[1]
    probs = np.random.random(len(offspring))

    for child, prob in zip(offspring, probs):
        if prob <= p_mutation:
            mutation_idxs = np.random.choice(fp_length, n_mutation, replace=False)
            child[mutation_idxs] = ~child[mutation_idxs]

    return offspring.astype(int)
