## Markov Decision Process (MDP)

A Markov Decision Process (MDP) is a mathematical framework for modeling decision-making problems in stochastic environments. It is widely used in the field of reinforcement learning for modeling and solving sequential decision-making problems.

### Core Concepts

- **States (S):** A finite set of states representing the possible situations or configurations of the environment. Each state is a snapshot of the environment at a given moment and captures all relevant information needed to make decisions.

- **Actions (A):** A finite set of actions that an agent can perform in each state. Actions represent the choices available to the agent, which can change the state of the environment and influence the rewards it receives.

- **Transitions (P):** A set of transition probabilities, P(s'|s, a), representing the probability of transitioning from state s to state s' when taking action a. The transition probabilities must satisfy the Markov property, meaning that the future state depends only on the current state and action, not on the history of previous states and actions. This simplifies the problem by allowing the agent to make decisions based on the current state only.

- **Rewards (R):** A reward function, R(s, a, s'), representing the immediate reward received by the agent after taking action a in state s and transitioning to state s'. The rewards can be positive or negative, reflecting the desirability of the outcome. The agent's goal is to maximize the cumulative reward over time, which involves finding a balance between immediate and future rewards.

- **Discount Factor (γ):** A discount factor between 0 and 1, representing the importance of future rewards relative to immediate rewards. A value close to 0 means the agent prioritizes immediate rewards, while a value close to 1 means the agent takes future rewards into account. The discount factor affects the agent's decision-making process and can influence its behavior, such as being more short-sighted or more far-sighted.

## Required Math

MDPs are primarily based on probability theory, linear algebra, and dynamic programming. Key concepts include probability distributions, conditional probabilities, matrix operations, and algorithms such as value iteration and policy iteration.

## Explanation of Real-World Example

An MDP models the decision-making process of an agent in a stochastic environment. The agent takes actions based on its current state and receives rewards, aiming to maximize the expected cumulative reward over time. The agent's goal is to find the optimal policy, a mapping from states to actions that maximizes the expected cumulative reward.

### Example: Autonomous Taxi

Imagine an autonomous taxi operating in a city. The city can be represented as a grid, where each cell is a state. The taxi has a set of actions it can perform, such as moving north, south, east, west, picking up a passenger, or dropping off a passenger.

The transitions between states are stochastic, meaning the taxi might not always reach its intended destination due to traffic or other factors. For example, there is a 0.9 probability that the taxi moves in the intended direction and a 0.1 probability that it remains in the same state.

The taxi receives a reward for each action it takes. For instance, it might receive a positive reward for successfully picking up and dropping off a passenger and a negative reward for an illegal action (e.g., attempting to pick up a passenger when the taxi is already occupied). The discount factor γ determines the importance of future rewards. If the taxi has a high discount factor, it will prioritize actions that maximize long-term profits, such as minimizing the time spent traveling between locations.

The MDP can be used to find the optimal policy for the taxi, which specifies the best action to take in each state to maximize the expected cumulative reward. This policy can be used by the autonomous taxi to make decisions in real-time, optimizing its performance and profit.

## Markov Decision Process (MDP) Formulations

### Bellman Equation

The Bellman equation is a recursive relationship that defines the optimal value function $V^*(s)$, which represents the maximum expected cumulative reward the agent can obtain from state $s$ by following an optimal policy. The Bellman equation for an MDP is given by:

$$
V^*(s) = \max_{a \in A} \sum_{s' \in S} P(s' | s, a) [R(s, a, s') + \gamma V^*(s')]
$$

### Value Iteration Algorithm

Value Iteration is an iterative algorithm used to solve MDPs and find the optimal value function $V^*$. The algorithm updates the value function for each state until convergence. The update equation is:

$$
V_{k+1}(s) = \max_{a \in A} \sum_{s' \in S} P(s' | s, a) [R(s, a, s') + \gamma V_k(s')]
$$

### Policy Iteration Algorithm

Policy Iteration is another iterative algorithm used to solve MDPs and find the optimal policy. The algorithm alternates between two steps: policy evaluation and policy improvement. In the policy evaluation step, the value function for the current policy is computed, and in the policy improvement step, the policy is updated based on the new value function. The algorithm terminates when the policy converges and no longer changes. The policy evaluation equation is:

$$
V^{\pi}(s) = \sum_{s' \in S} P(s' | s, \pi(s)) [R(s, \pi(s), s') + \gamma V^{\pi}(s')]
$$

And the policy improvement step updates the policy as follows:

$$
\pi'(s) = \arg\max_{a \in A} \sum_{s' \in S} P(s' | s, a) [R(s, a, s') + \gamma V^{\pi}(s')]
$$
