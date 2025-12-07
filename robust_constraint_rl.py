import numpy as np
import time
from typing import Callable, Tuple
import multiprocessing as mp

class MDPEnvironment:
    def __init__(self, S: int, A: int):
        self.S = S
        self.A = A
        self.transition_tensor = self._generate_transition_tensor()
        self.reward_matrix = self._generate_reward_matrix()
        self.constraint_reward_matrix = self._generate_constraint_reward_matrix()

    def _generate_transition_tensor(self) -> np.ndarray:
        transition_tensor = np.zeros((self.S, self.A, self.S))
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
        return transition_tensor

    def _generate_reward_matrix(self) -> np.ndarray:
        return np.array([[1, 1], [0, 0], [0, 0]])

    def _generate_constraint_reward_matrix(self) -> np.ndarray:
        return np.array([[0, 0], [0, 0], [1, 1]])

    def reward_func(self, s: int, a: int) -> float:
        return self.reward_matrix[s, a]

    def constraint_reward_func(self, s: int, a: int) -> float:
        return self.constraint_reward_matrix[s, a]

class TDLearning:
    def __init__(self, S: int, A: int, transition_tensor: np.ndarray, reward_func: Callable,
                 policy: np.ndarray, gamma: float = 0.9, alpha: float = 0.1, episodes: int = 10):
        self.S = S
        self.A = A
        self.transition_tensor = transition_tensor
        self.reward_func = reward_func
        self.policy = policy
        self.gamma = gamma
        self.alpha = alpha
        self.episodes = episodes

    def learn(self) -> np.ndarray:
        V = np.zeros(self.S)
        for _ in range(self.S):
            s = _
            for _ in range(10000):
                a = np.random.choice(self.A, p=self.policy[s])
                s_prime = np.random.choice(self.S, p=self.transition_tensor[:, a, s])
                reward = self.reward_func(s, a)
                td_error = reward + self.gamma * V[s_prime] - V[s]
                V[s] += self.alpha * td_error
                s = s_prime
                if np.abs(td_error) < 1e-6:
                    break
        return V

class RobustTDLearning(TDLearning):
    def __init__(self, *args, delta: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = delta

    def learn(self) -> np.ndarray:
        V = np.zeros(self.S)
        lr = self.alpha
        count = 0
        for _ in range(self.S):
            s = _
            for _ in range(10000):
                a = np.random.choice(self.A, p=self.policy[s])
                s_prime = np.random.choice(self.S, p=self.transition_tensor[:, a, s])
                reward = self.reward_func(s, a)
                omega = np.mean(V)
                kappa = np.sqrt(np.sum((V - omega)**2))
                if not np.isclose(kappa, 0):
                    u = np.sign(V - omega) * np.abs(V - omega) / kappa
                    td_error = reward + self.gamma * V[s_prime] - V[s] - self.gamma * self.delta * u.dot(V)
                else:
                    td_error = reward + self.gamma * V[s_prime] - V[s]
                V[s] += lr * td_error
                count += 1
                lr = self.alpha / count
                s = s_prime
                if np.abs(td_error) < 1e-6:
                    break
        return V

def solve_lagrangian(env: MDPEnvironment, pi0: float, lambda_: float, c: float = 0.1) -> float:
    policy = np.ones((env.S, env.A))
    policy[0] = [pi0, 1 - pi0]
    policy = policy / policy.sum(axis=1, keepdims=True)

    td_learner = RobustTDLearning(env.S, env.A, env.transition_tensor, env.reward_func, policy)
    rV_primal = td_learner.learn()

    constraint_td_learner = RobustTDLearning(env.S, env.A, env.transition_tensor, env.constraint_reward_func, policy)
    rV_constraint = constraint_td_learner.learn()

    return rV_primal[0] - lambda_ * (c - rV_constraint[0])

def process_row(args):
    env, row, pi0_values, lambda_values = args
    row_results = np.zeros(len(lambda_values))
    for j, (pi0, lambda_) in enumerate(zip(pi0_values, lambda_values)):
        row_results[j] = solve_lagrangian(env, pi0, lambda_)
    return row_results

def grid_search_parallel(env: MDPEnvironment, lambda_range: Tuple[float, float], pi0_range: Tuple[float, float], n_points: int = 50, n_cores: int = 6) -> np.ndarray:
    lambda_values = np.linspace(*lambda_range, n_points)
    pi0_values = np.linspace(*pi0_range, n_points)
    lambda_grid, pi0_grid = np.meshgrid(lambda_values, pi0_values)

    start_time = time.time()

    # Prepare arguments for multiprocessing
    args = [(env, row, pi0_values, lambda_values) for row in range(n_points)]

    # Use multiprocessing to compute rows in parallel
    with mp.Pool(processes=n_cores) as pool:
        results = pool.map(process_row, args)

    lagrangian_values = np.array(results)

    end_time = time.time()
    print(f"Grid search completed. Total time: {end_time - start_time:.2f} seconds")

    return lagrangian_values

def save_lagrangian_values(lagrangian_values: np.ndarray, filename: str = "lagrangian_values.npy") -> None:
    """
    Save the Lagrangian values to a file.

    Args:
    lagrangian_values (np.ndarray): The computed Lagrangian values.
    filename (str): The name of the file to save the values to.
    """
    np.save(filename, lagrangian_values)
    print(f"Lagrangian values saved to {filename}")

if __name__ == "__main__":
    S, A = 3, 2
    env = MDPEnvironment(S, A)
    n_cores = 6  # Number of cores to use
    lagrangian_values = grid_search_parallel(env, (0, 5), (0, 1), n_points=100, n_cores=n_cores)
    print("Grid search completed.")
    # print(lagrangian_values)
    # Further analysis or visualization of lagrangian_values can be added here

    # Save the Lagrangian values
    save_lagrangian_values(lagrangian_values)

    # Print the shape of the Lagrangian values array
    print(f"Shape of Lagrangian values array: {lagrangian_values.shape}")

    primal_value = np.max(np.min(lagrangian_values, axis=1))
    dual_value = np.min(np.max(lagrangian_values, axis=0))

    print(primal_value)
    print(dual_value)
