import numpy as np

class FrozenLake:
    def __init__(self, size=4, p=0.8, hole_positions=None):
        self.size = size
        self.p = p  # probability of moving in the intended direction
        self.action_space = 4
        self.observation_space = size * size

        # Define hole positions
        self.hole_positions = hole_positions if hole_positions is not None else [5, 7, 11, 12]

        # Define goal position (always at the bottom-right corner)
        self.goal_position = size * size - 1

        self.reset()

        # Generate transition probability matrix
        self.P = self.generate_transition_matrix()

        # Define rewards
        self.R = np.zeros(self.observation_space)
        self.R[self.goal_position] = 1.0
        for hole in self.hole_positions:
            self.R[hole] = -1.0

    def reset(self):
        self.position = 0
        self.done = False
        return self.position
    def generate_data(self, policy, num_episodes, max_steps_per_episode=100):
        """
        Generate a dataset of state-action pairs based on the given policy.

        Args:
        policy (np.array): The policy to follow, shape (S, A) where each row is a probability distribution over actions.
        num_episodes (int): The number of episodes to simulate.
        max_steps_per_episode (int): Maximum number of steps per episode to prevent infinite loops.

        Returns:
        list: A list of tuples (state, action) representing the generated data.
        """
        data = []

        for _ in range(num_episodes):
            state = self.reset()
            for _ in range(max_steps_per_episode):
                action = np.random.choice(self.action_space, p=policy[state])
                data.append((state, action))

                next_state, reward, done, _ = self.step(action)

                if done:
                    break

                state = next_state

        return data

    def step(self, action):
        if self.done:
            return self.position, 0, True, {}

        directions = [
            -1,  # left
            self.size,  # down
            1,  # right
            -self.size  # up
        ]

        intended_direction = directions[action]
        slip_direction = intended_direction * 2

        if np.random.random() < self.p:
            # Move in the intended direction
            new_position = self.position + intended_direction
        else:
            # Slip to the next next position
            new_position = self.position + slip_direction

        # Check boundaries and update position
        if 0 <= new_position < self.size * self.size and abs(new_position % self.size - self.position % self.size) <= 2:
            self.position = new_position

        # Check if reached goal, fell in hole, or still in game
        if self.position == self.goal_position:
            reward = 1.0
            self.done = True
        elif self.position in self.hole_positions:
            reward = -1.0
            self.done = True
        else:
            reward = 0.0
            self.done = False

        return self.position, reward, self.done, {}

    def render(self):
        for i in range(self.size):
            for j in range(self.size):
                pos = i * self.size + j
                if pos == self.position:
                    print('P', end=' ')
                elif pos in self.hole_positions:
                    print('H', end=' ')
                elif pos == self.goal_position:
                    print('G', end=' ')
                else:
                    print('.', end=' ')
            print()
        print()

    def get_transition_prob(self, current_state, action, next_state):
        directions = [
            -1,  # left
            self.size,  # down
            1,  # right
            -self.size  # up
        ]

        intended_direction = directions[action]
        slip_direction = intended_direction * 2

        intended_next_state = current_state + intended_direction
        slip_next_state = current_state + slip_direction

        # Check if the transition is valid (within boundaries)
        is_valid_intended = (0 <= intended_next_state < self.size * self.size and
                             abs(intended_next_state % self.size - current_state % self.size) <= 1)
        is_valid_slip = (0 <= slip_next_state < self.size * self.size and
                         abs(slip_next_state % self.size - current_state % self.size) <= 2)

        if next_state == current_state:
            # Probability of staying in the same state (hitting a wall)
            return 1 - (self.p * is_valid_intended + (1 - self.p) * is_valid_slip)
        elif next_state == intended_next_state and is_valid_intended:
            # Probability of moving to the intended next state
            return self.p
        elif next_state == slip_next_state and is_valid_slip:
            # Probability of slipping to the next next state
            return 1 - self.p
        else:
            # Transition is not possible
            return 0.0


    def generate_transition_matrix(self):
        S = self.observation_space
        A = self.action_space

        # Initialize the transition probability matrix
        P = np.zeros((S, A, S))

        for s in range(S):
            for a in range(A):
                for s_next in range(S):
                    P[s, a, s_next] = self.get_transition_prob(s, a, s_next)

        # Adjust for terminal states (holes and goal)
        for hole in self.hole_positions:
            P[hole, :, :] = 0
            P[hole, :, hole] = 1

        P[self.goal_position, :, :] = 0
        P[self.goal_position, :, self.goal_position] = 1

        return P


    def compute_value_function(self, policy, gamma, theta=1e-8):
        """
        Compute the value function for a given stochastic policy using value iteration.

        Args:
        policy (np.array): The policy to evaluate, shape (S, A) where each row is a probability distribution over actions.
        gamma (float): The discount factor.
        theta (float): The threshold for convergence.

        Returns:
        np.array: The value function, shape (S,).
        """
        V = np.zeros(self.observation_space)

        while True:
            delta = 0
            for s in range(self.observation_space):
                v = V[s]
                # Compute the value as the expected value over all actions
                V[s] = np.sum(policy[s] * np.sum(self.P[s] * (self.R + gamma * V), axis=1))
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break

        return V

    def compute_q_function(self, policy, gamma, theta=1e-8):
        """
        Compute the Q-function (state-action value function) for a given stochastic policy.

        Args:
        policy (np.array): The policy to evaluate, shape (S, A) where each row is a probability distribution over actions.
        gamma (float): The discount factor.
        theta (float): The threshold for convergence.

        Returns:
        np.array: The Q-function, shape (S, A).
        """
        # First, compute the value function
        V = self.compute_value_function(policy, gamma, theta)

        # Now compute the Q-function
        Q = np.zeros((self.observation_space, self.action_space))
        for s in range(self.observation_space):
            for a in range(self.action_space):
                Q[s, a] = np.sum(self.P[s, a] * (self.R + gamma * V))

        return Q

    def generate_random_policy(self):
        """
        Generate a random stochastic policy.

        Returns:
        np.array: A random policy, shape (S, A).
        """
        policy = np.random.rand(self.observation_space, self.action_space)
        # Normalize each row to make it a valid probability distribution
        return policy / policy.sum(axis=1, keepdims=True)

    def compute_occupancy_measure(self, policy, gamma, num_iterations=1000):
        """
        Compute the occupancy measure (state visitation frequency) for a given policy.

        Args:
        policy (np.array): The policy to evaluate, shape (S, A) where each row is a probability distribution over actions.
        gamma (float): The discount factor.
        num_iterations (int): Number of iterations for the computation.

        Returns:
        np.array: The occupancy measure, shape (S,).
        """
        S = self.observation_space
        A = self.action_space

        # Initialize occupancy measure
        occupancy = np.zeros(S)

        # Initial state distribution (assuming always starting at state 0)
        initial_distribution = np.zeros(S)
        initial_distribution[0] = 1

        # Compute state transition probabilities under the given policy
        P_pi = np.sum(self.P * policy[:, :, np.newaxis], axis=1)

        # Iterative computation of occupancy measure
        for _ in range(num_iterations):
            occupancy = initial_distribution + gamma * P_pi.T @ occupancy

        # Normalize the occupancy measure
        occupancy /= np.sum(occupancy)

        return occupancy

    def visualize_occupancy_measure(self, occupancy):
        """
        Visualize the occupancy measure on the grid.

        Args:
        occupancy (np.array): The occupancy measure, shape (S,).
        """
        print("Occupancy Measure Visualization:")
        for i in range(self.size):
            for j in range(self.size):
                pos = i * self.size + j
                if pos in self.hole_positions:
                    print('H'.center(8), end=' ')
                elif pos == self.goal_position:
                    print('G'.center(8), end=' ')
                else:
                    print(f'{occupancy[pos]:.4f}'.center(8), end=' ')
            print()
        print()

ACTION_MAPPING = {
    0: "left",
    1: "down",
    2: "right",
    3: "up"
}


def test_occupancy_measure():
    # Create a FrozenLake instance
    env = FrozenLake(size=4, p=0.8)

    # Generate a random policy
    random_policy = env.generate_random_policy()

    # Compute the occupancy measure
    gamma = 0.99
    occupancy = env.compute_occupancy_measure(random_policy, gamma)

    # Test 1: Check if the occupancy measure sums to 1 (it should be normalized)
    assert np.isclose(np.sum(occupancy), 1.0), "Occupancy measure should sum to 1"

    # Test 2: Check if all values are non-negative
    assert np.all(occupancy >= 0), "All occupancy values should be non-negative"

    # Test 3: Check if the initial state (0) has non-zero occupancy
    assert occupancy[0] > 0, "Initial state should have non-zero occupancy"

    # Test 4: Check if the goal state has non-zero occupancy
    assert occupancy[env.goal_position] > 0, "Goal state should have non-zero occupancy"

    print("All tests passed successfully!")

    # Print occupancy measure statistics
    print(f"\nOccupancy measure statistics:")
    print(f"Min occupancy: {np.min(occupancy):.6f}")
    print(f"Max occupancy: {np.max(occupancy):.6f}")
    print(f"Mean occupancy: {np.mean(occupancy):.6f}")
    print(f"Median occupancy: {np.median(occupancy):.6f}")
    print(f"Standard deviation: {np.std(occupancy):.6f}")
    print(f"\nInitial state (0) occupancy: {occupancy[0]:.6f}")
    print(f"Goal state occupancy: {occupancy[env.goal_position]:.6f}")

    hole_occupancies = occupancy[env.hole_positions]
    non_hole_occupancies = occupancy[
        list(set(range(env.observation_space)) - set(env.hole_positions) - {env.goal_position})]

    print(f"\nAverage hole state occupancy: {np.mean(hole_occupancies):.6f}")
    print(f"Average non-hole, non-goal state occupancy: {np.mean(non_hole_occupancies):.6f}")

    print("\nTop 5 most visited states:")
    top_5_indices = np.argsort(occupancy)[-5:][::-1]
    for idx in top_5_indices:
        state_type = "Initial" if idx == 0 else "Goal" if idx == env.goal_position else "Hole" if idx in env.hole_positions else "Normal"
        print(f"State {idx} ({state_type}): {occupancy[idx]:.6f}")

    print("\nBottom 5 least visited states:")
    bottom_5_indices = np.argsort(occupancy)[:5]
    for idx in bottom_5_indices:
        state_type = "Initial" if idx == 0 else "Goal" if idx == env.goal_position else "Hole" if idx in env.hole_positions else "Normal"
        print(f"State {idx} ({state_type}): {occupancy[idx]:.6f}")

    # Visualize the occupancy measure
    print("\nOccupancy Measure Visualization:")
    env.visualize_occupancy_measure(occupancy)


if __name__ == "__main__":
    test_occupancy_measure()