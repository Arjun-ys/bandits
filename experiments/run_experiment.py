import numpy as np
import matplotlib.pyplot as plt

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

    rewards = np.zeros(T)
    regrets = np.zeros(T)

    for t in range(T):
        # Agent decides which arm to pull:
        # explore with probability ε, otherwise exploit current knowledge
        arm = agent.select_action()

        # Get reward from environment(0 or 1)
        reward = bandit.pull(arm)

        # Update agent estimates
        agent.update(arm, reward)

        # Store reward
        rewards[t] = reward

        # Regret = best expected reward - obtained reward
        regrets[t] = bandit.optimal_reward - reward

    print("Final estimated values:", agent.Q)

    plot_results(rewards, regrets)


def plot_results(rewards, regrets):
    cumulative_rewards = np.cumsum(rewards)
    cumulative_regret = np.cumsum(regrets)

    plt.figure()
    plt.plot(cumulative_rewards)
    plt.xlabel("Time step")
    plt.ylabel("Cumulative reward")
    plt.title("Learning curve: cumulative reward")
    plt.show()

    plt.figure()
    plt.plot(cumulative_regret)
    plt.xlabel("Time step")
    plt.ylabel("Cumulative regret")
    plt.title("Learning curve: cumulative regret")
    plt.show()


if __name__ == "__main__":
    run_experiment()
