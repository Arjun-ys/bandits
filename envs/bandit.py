import numpy as np
class BernoulliBandit:
    def __init__(self,probs,seed=Npne):
        """
        Docstring for __init__
        :param probs: list or np array of success probabilities for each arm
        :param seed: random seed for reproducibility
        """
        self.probs = np.array(probs)
        self.n_arms=len(probs) #number of arms (slot machine arms that you can pull)

        self.rng=np.random.default_rng(seed)

    def pull(self, arm):
        """
        Pull an arm and return a reward (0 or 1).
        """
        if arm < 0 or arm >= self.n_arms:
            raise ValueError("Invalid arm index")

        reward = self.rng.random() < self.probs[arm]
        return int(reward)