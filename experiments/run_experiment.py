import numpy as np

from envs.bandit import BernoulliBandit
from algorithms.epsilon_greedy import EpsilonGreedyAgent


def run_experiment():
    # ----- Environment -----
    probs = [0.1, 0.5, 0.8]
    bandit = BernoulliBandit(probs, seed=42)

    # ----- Agent -----
    agent = EpsilonGreedyAgent(
        n_arms=len(probs),
        epsilon=0.1,
        seed=42
    )

    T = 1000
    total_reward = 0

    for t in range(T):

        # Agent decides which arm to pull:
        # explore with probability ε, otherwise exploit current knowledge
        arm = agent.select_action()

        # Environment returns a stochastic reward (0 or 1)
        reward = bandit.pull(arm)

        # Agent updates its estimate of this arm's value
        # using an incremental average of past rewards
        agent.update(arm, reward)
        

        total_reward += reward

    print("Total reward:", total_reward)
    print("Estimated values:", agent.Q)
    print("True probabilities:", probs)


if __name__ == "__main__":
    run_experiment()
