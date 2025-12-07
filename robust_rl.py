import numpy as np


def generate_transition_tensor(S, A):
    """
    Generates a 3D tensor representing the transition probabilities.

    Parameters:
    S (int): Number of states.
    A (int): Number of actions.

    Returns:
    np.ndarray: A 3D tensor of shape (S, A, S) representing the transition probabilities.
    """
    # Randomly generate the probabilities
    transition_tensor = np.random.rand(S, A, S)

    # Normalize the probabilities so they sum to 1 across all s' for each (s, a)
    transition_tensor = transition_tensor / transition_tensor.sum(axis=0, keepdims=True)

    return transition_tensor


def generate_reward(S, A):
    """
    Generates a reward function for a given number of states and actions.

    Parameters:
    S (int): Number of states.
    A (int): Number of actions.

    Returns:
    function: A function reward(s, a) that returns the reward for taking action a in state s.
    """
    # Randomly generate the rewards for each state-action pair
    reward_matrix = np.random.rand(S, A)

    def reward(s, a):
        """
        Returns the reward for taking action a in state s.

        Parameters:
        s (int): Current state.
        a (int): Action taken.

        Returns:
        float: The reward for the state-action pair.
        """
        return reward_matrix[s, a]

    return reward


def td_learning(S, A, transition_tensor, reward_func, policy, gamma=0.9, alpha=0.1, episodes=10):
    """
    Implements the TD(0) learning algorithm.

    Parameters:
    S (int): Number of states.
    A (int): Number of actions.
    transition_tensor (np.ndarray): Transition probabilities of shape (S', A, S).
    reward_func (function): A function reward(s, a) that returns the reward for taking action a in state s.
    policy (np.ndarray): Policy array of shape (S, A), where policy[s, a] is the probability of taking action a in state s.
    gamma (float): Discount factor.
    alpha (float): Learning rate.
    episodes (int): Number of episodes to run.

    Returns:
    np.ndarray: Estimated value function array of shape (S,).
    """
    V = np.zeros(S)  # Initialize the value function to zero

    for episode in range(episodes):
        s = np.random.choice(S)  # Start at a random state

        for _ in range(10000):
            # Select an action according to the policy
            a = np.random.choice(A, p=policy[s])

            # Transition to the next state based on the transition probabilities
            s_prime = np.random.choice(S, p=transition_tensor[:, a, s])

            # Get the reward for the state-action pair
            reward = reward_func(s, a)

            # Update the value function using the TD(0) update rule
            td_error = reward + gamma * V[s_prime] - V[s]
            V[s] += alpha * td_error

            # Move to the next state
            s = s_prime
            if np.abs(td_error) < 10e-6:
                break

    return V


def robust_td_learning(S, A, transition_tensor, reward_func, policy, gamma=0.9, alpha=0.1, episodes=10, delta=0.1):
    """
    Implements the TD(0) learning algorithm.

    Parameters:
    S (int): Number of states.
    A (int): Number of actions.
    transition_tensor (np.ndarray): Transition probabilities of shape (S', A, S).
    reward_func (function): A function reward(s, a) that returns the reward for taking action a in state s.
    policy (np.ndarray): Policy array of shape (S, A), where policy[s, a] is the probability of taking action a in state s.
    gamma (float): Discount factor.
    alpha (float): Learning rate.
    episodes (int): Number of episodes to run.

    Returns:
    np.ndarray: Estimated value function array of shape (S,).
    """
    V = np.zeros(S)  # Initialize the value function to zero
    for s in range(S):
        for _ in range(100000):
            td_error = 0
            for a in range(A):
                for s_prime in range(S):
                    td_error += (reward_func(s, a) + gamma * V[s_prime] - V[s]  ) * transition_tensor[s_prime, a, s] * policy[s, a]
            omega = np.mean(V)
            kappa = np.sqrt(np.sum((V - omega) ** 2))
            sign = np.sign(V - omega)
            if not np.isclose(kappa, 0):
                u = sign * np.abs(V - omega) / kappa
                td_error = td_error - gamma * delta * u.dot(V)
            V[s] += alpha * td_error
            if np.abs(td_error) < 10e-7:
                break

    return V


# Estimated Value Function V(s) using TD-learning:
#  [5.01649605 4.54226774 4.46276859]
# Estimated Robust Value Function V(s) using TD-learning:
#  [4.65327279 4.14508549 4.1025244 ]
# (Constraint) Estimated Value Function V(s) using TD-learning:
#  [4.38580568 3.95291545 5.08586794]
# (Constraint) Estimated Robust Value Function V(s) using TD-learning:
#  [3.83753328 3.2049979  4.37899952]

# Estimated Value Function V(s) using TD-learning:
#  [5.14200432 4.60430812 4.60222289]
# Estimated Robust Value Function V(s) using TD-learning:
#  [4.65855027 4.13678462 4.10884336]
# (Constraint) Estimated Value Function V(s) using TD-learning:
#  [4.60677122 4.06681461 5.1699771 ]
# (Constraint) Estimated Robust Value Function V(s) using TD-learning:
#  [3.81519249 3.28447814 4.55981773]


pi0 = 0.1

# Example usage:
S = 3  # Number of states
A = 2  # Number of actions
gamma = 0.9  # Discount factor
alpha = 0.1  # Learning rate
episodes = 20  # Number of episodes

# Generate a random policy, transition tensor, and reward function
policy = np.ones((S, A))
policy[0, 0] = pi0
policy[0, 1] = 1 - pi0
policy = policy / policy.sum(axis=1, keepdims=True)  # Normalize to create a valid policy

transition_tensor = generate_transition_tensor(S, A)
transition_tensor[0, 0, 0] = 0
transition_tensor[1, 0, 0] = 1
transition_tensor[2, 0, 0] = 0

transition_tensor[0, 1, 0] = 0
transition_tensor[1, 1, 0] = 0
transition_tensor[2, 1, 0] = 1

transition_tensor[0, 0, 1] = 1
transition_tensor[1, 0, 1] = 0
transition_tensor[2, 0, 1] = 0

transition_tensor[0, 1, 1] = 1
transition_tensor[1, 1, 1] = 0
transition_tensor[2, 1, 1] = 0

transition_tensor[0, 0, 2] = 0.9
transition_tensor[1, 0, 2] = 0
transition_tensor[2, 0, 2] = 0.1

transition_tensor[0, 1, 2] = 0.9
transition_tensor[1, 1, 2] = 0
transition_tensor[2, 1, 2] = 0.1


# Reward function: deterministic rewards for state-action pairs
def reward_func(s, a):
    reward_matrix = np.array([
        [1, 1],  # Rewards for state 0
        [0, 0],  # Rewards for state 1
        [0, 0],  # Rewards for state 2
    ])
    return reward_matrix[s, a]


def reward_func_constraint(s, a):
    reward_matrix = np.array([
        [0, 0],  # Rewards for state 0
        [0, 0],  # Rewards for state 1
        [1, 1],  # Rewards for state 2
    ])
    return reward_matrix[s, a]


# Compute the value function using TD-learning
V_td = td_learning(S, A, transition_tensor, reward_func, policy, gamma, alpha, episodes)
rV_td = robust_td_learning(S, A, transition_tensor, reward_func, policy, gamma, alpha, episodes)

print("Estimated Value Function V(s) using TD-learning:\n", V_td)
print("Estimated Robust Value Function V(s) using TD-learning:\n", rV_td)

# Compute the value function using TD-learning
V_td = td_learning(S, A, transition_tensor, reward_func_constraint, policy, gamma, alpha, episodes)
rV_td = robust_td_learning(S, A, transition_tensor, reward_func_constraint, policy, gamma, alpha, episodes)

print("(Constraint) Estimated Value Function V(s) using TD-learning:\n", V_td)
print("(Constraint) Estimated Robust Value Function V(s) using TD-learning:\n", rV_td)


def solve_lagrangian(pi0, lambda_, c=0.1):
    # define policy
    policy = np.ones((S, A))
    policy[0, 0] = pi0
    policy[0, 1] = 1 - pi0
    policy = policy / policy.sum(axis=1, keepdims=True)  # Normalize to create a valid policy

    # solve primal value function
    rV_primal = robust_td_learning(S, A, transition_tensor, reward_func, policy, gamma, alpha, episodes)

    # solve constraint value function
    rV_constraint = robust_td_learning(S, A, transition_tensor, reward_func_constraint, policy, gamma, alpha, episodes)

    return rV_primal[0] - lambda_ * (c - rV_constraint[0])


if __name__ == "__main__":
    pass
    # import time
    #
    # # Define the ranges and grid for lambda_ and pi0
    # lambda_values = np.linspace(0, 1, 10)  # adjust the range as needed
    # pi0_values = np.linspace(0, 1, 10)
    #
    # # Create a meshgrid
    # lambda_grid, pi0_grid = np.meshgrid(lambda_values, pi0_values)
    #
    # # Initialize a matrix to store the Lagrangian function values
    # lagrangian_values = np.zeros_like(lambda_grid)
    # # Evaluate the Lagrangian function on the grid
    #
    # start_time = time.time()
    # for i in range(lambda_grid.shape[0]):
    #     for j in range(lambda_grid.shape[1]):
    #         lagrangian_values[i, j] = solve_lagrangian(pi0_grid[i, j], lambda_grid[i, j])
    #         end_time = time.time()
    #     print(f"Time taken: {end_time - start_time} seconds")
