import numpy as np


class EpsilonGreedyAgent:
    def __init__(self, n_arms, epsilon=0.1, seed=None):
        """
        n_arms: number of arms in the bandit
        epsilon: probability of random exploration
        seed: random seed for reproducibility
        """
        self.n_arms = n_arms
        self.epsilon = epsilon

        self.Q = np.zeros(n_arms)  # estimated value of each arm
        self.N = np.zeros(n_arms)  # number of times each arm was pulled

        self.rng = np.random.default_rng(seed)
