# Stateless Algorithms: Multi-Armed Bandits

This project is built to solidify my understanding of **stateless learning algorithms**, as covered in my Deep Learning class.

The focus is on the **Multi-Armed Bandits (MAB)** problem, a classic framework for studying the **exploration–exploitation trade-off**.

---

## Problem Overview

The Multi-Armed Bandits problem is often described using the analogy of a gambler choosing between multiple slot machines (arms).

- Each slot machine provides rewards with a different (unknown) probability
- The gambler wants to **maximize total reward over time**
- To do this, the gambler must balance:
  - **Exploration**: trying different machines to learn their rewards
  - **Exploitation**: choosing the machine that appears best so far

The challenge is to make good decisions **without knowing the true reward probabilities**.

---

## Environment

This project uses a **Bernoulli bandit environment**.

- The environment has **K arms**
- Each arm has a fixed but unknown probability of reward \( p \)
- When an arm is pulled:
  - Reward = `1` with probability \( p \)
  - Reward = `0` otherwise

Important distinction:
- The **agent does not know** the reward probabilities
- The **simulator/environment does know** them (used only for evaluation)

The environment is **stateless**:
- Each arm behaves the same way every time
- There are no state transitions or delayed rewards

---

## Agent

The agent implements the **ε-greedy algorithm**, a simple stateless learning strategy:

- With probability **ε**, the agent explores by selecting a random arm
- With probability **1 − ε**, the agent exploits by selecting the arm with the highest estimated reward
- The agent learns by updating **incremental averages** of observed rewards

---

## Key Concepts Demonstrated

- Stateless learning
- Trial-and-error optimization
- Exploration vs exploitation
- Incremental mean updates
- Regret as a performance metric

---

## Purpose of the Project

This project is intended for **learning and experimentation**, not performance optimization.

The goal is to build intuition for:
- How simple reward-driven agents learn
- Why randomness is essential for learning
- How intelligent behavior can emerge from simple rules
