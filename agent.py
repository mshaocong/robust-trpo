import numpy as np
from scipy.optimize import minimize

class Agent:
    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.95, epsilon=0.1, temperature=1.0):
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.temperature = temperature  # Temperature parameter for softmax

        # Initialize Q-table
        self.q_table = np.zeros((state_space, action_space))
        self.agent_type = "Basic"

    def get_action_probabilities(self, state):
        # Convert Q-values to probabilities using softmax
        q_values = self.q_table[state]
        exp_q = np.exp((q_values - np.max(q_values)) / self.temperature)
        probabilities = exp_q / np.sum(exp_q)
        return probabilities

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            # Explore: select a random action
            return np.random.randint(self.action_space)
        else:
            # Exploit: select the action based on softmax probabilities
            probabilities = self.get_action_probabilities(state)
            return np.random.choice(self.action_space, p=probabilities)

    def update(self, state, action, reward, next_state, done):
        # Q-learning update
        current_q = self.q_table[state, action]
        if done:
            max_next_q = 0
        else:
            max_next_q = np.max(self.q_table[next_state])

        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state, action] = new_q

    def decay_epsilon(self, decay_rate=0.99):
        self.epsilon *= decay_rate
        self.epsilon = max(0.01, self.epsilon)

    def set_temperature(self, temperature):
        self.temperature = temperature


class QLearning(Agent):
    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.95, epsilon=0.1,
                 temperature=1.0):
        super().__init__(state_space, action_space, learning_rate, discount_factor, epsilon, temperature)
        self.agent_type = "PolicyGradient"

    def update_policy(self, state, action, reward, next_state, done):
        """
        Update the policy based on the reward received and the action taken.
        This is a simplified version of policy gradient update without using neural networks.
        """
        # Simple policy gradient update could be just modifying the Q-values based on the reward
        # Here, we use a very basic approach to increase the Q-value of the action that was taken
        # if the reward is positive, and decrease it otherwise.
        # This is a naive implementation and not a true policy gradient update.
        if not done:
            self.q_table[state, action] = self.q_table[state, action] + self.learning_rate * reward
        else:
            # If the episode is done, we might want to adjust the Q-value more significantly
            self.q_table[state, action] =  self.q_table[state, action] + self.learning_rate * reward * 2

    def update(self, state, action, reward, next_state, done):
        """
        Overrides the update method to use the policy gradient update logic.
        """
        self.update_policy(state, action, reward, next_state, done)


class TRPOAgent(Agent):
    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.99, epsilon=0.1,
                 temperature=1.0, lambda_=0.95, max_kl=0.01, batch_size=32):
        super().__init__(state_space, action_space, learning_rate, discount_factor, epsilon, temperature)
        self.lambda_ = lambda_
        self.max_kl = max_kl
        self.batch_size = batch_size

        # Initialize policy
        self.policy = np.ones((state_space, action_space)) / action_space
        self.value_table = np.zeros(state_space)
        self.experience_buffer = []
        self.agent_type = "TRPO"


    def compute_advantages(self, states, rewards, next_states, dones):
        advantages = np.zeros_like(rewards, dtype=float)
        last_advantage = 0
        last_value = self.value_table[next_states[-1]] * (1 - dones[-1])

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.discount_factor * self.value_table[next_states[t]] * (1 - dones[t]) - \
                    self.value_table[states[t]]
            advantages[t] = delta + self.discount_factor * self.lambda_ * last_advantage * (1 - dones[t])
            last_advantage = advantages[t]

        return advantages

    def compute_surrogate_loss(self, states, actions, advantages, old_policy):
        new_policy = self.policy[states, actions]
        old_policy_probs = old_policy[states, actions]
        return -np.mean(advantages * new_policy / (old_policy_probs + 1e-8))

    def compute_kl_divergence(self, states, old_policy):
        old_policy_probs = old_policy[states]
        new_policy_probs = self.policy[states]
        kl = np.sum(old_policy_probs * (np.log(old_policy_probs + 1e-8) - np.log(new_policy_probs + 1e-8)), axis=1)
        return np.mean(kl)

    def update_policy(self, states, actions, advantages, old_policy):
        unique_states = np.unique(states)

        def objective(flat_policy):
            self.policy[unique_states] = flat_policy.reshape(-1, self.action_space)
            return self.compute_surrogate_loss(states, actions, advantages, old_policy)

        def constraint(flat_policy):
            self.policy[unique_states] = flat_policy.reshape(-1, self.action_space)
            return self.max_kl - self.compute_kl_divergence(unique_states, old_policy)

        flat_policy = self.policy[unique_states].flatten()

        results = minimize(
            objective,
            flat_policy,
            method='SLSQP',
            constraints={'type': 'ineq', 'fun': constraint},
            options={'maxiter': 50}
        )

        self.policy[unique_states] = results.x.reshape(-1, self.action_space)
        self.policy[unique_states] /= self.policy[unique_states].sum(axis=1, keepdims=True)

    def update_value_function(self, states, returns):
        for state, ret in zip(states, returns):
            self.value_table[state] += self.learning_rate * (ret - self.value_table[state])

    def update(self, state, action, reward, next_state, done):
        self.experience_buffer.append((state, action, reward, next_state, done))

        if len(self.experience_buffer) >= self.batch_size:
            self.batch_update()
            self.experience_buffer = []
        if not done:
            self.q_table[state, action] = self.q_table[state, action] + self.learning_rate * reward
        else:
            # If the episode is done, we might want to adjust the Q-value more significantly
            self.q_table[state, action] =  self.q_table[state, action] + self.learning_rate * reward * 2
    def batch_update(self):
        if len(self.experience_buffer) == 0:
            return

        states, actions, rewards, next_states, dones = zip(*self.experience_buffer)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        advantages = self.compute_advantages(states, rewards, next_states, dones)
        returns = advantages + self.value_table[states]

        old_policy = self.policy.copy()

        self.update_policy(states, actions, advantages, old_policy)
        self.update_value_function(states, returns)

    def decay_epsilon(self, decay_rate=0.99):
        super().decay_epsilon(decay_rate)


class RobustTRPO(TRPOAgent):
    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.99, epsilon=0.1,
                 temperature=1.0, lambda_=0.95, max_kl=1.0, batch_size=32, confidence=0.95):
        super().__init__(state_space, action_space, learning_rate, discount_factor, epsilon,
                         temperature, lambda_, max_kl, batch_size)
        self.confidence = confidence
        self.transition_model = np.zeros((state_space, action_space, state_space))
        self.target_transition_model = np.zeros((state_space, action_space, state_space))
        self.agent_type = "RobustTRPO"
        self.experience_buffer = []
        self.regularization = 1.0

    def compute_worst_case_transition_probability(self, state, action):
        transitions = self.transition_model[state, action]
        total_transitions = np.sum(transitions)

        if total_transitions == 0:
            return np.ones(self.state_space) / self.state_space

        empirical_probs = transitions / total_transitions
        interval_size = np.sqrt(np.log(2 / (1 - self.confidence)) / (2 * total_transitions))

        def objective(probs):
            return -np.sum(empirical_probs * np.log(probs + 1e-8))

        def gradient(probs):
            return -empirical_probs / (probs + 1e-8)

        worst_case_probs = empirical_probs - interval_size
        worst_case_probs = np.clip(worst_case_probs, 0, 1)
        worst_case_probs /= np.sum(worst_case_probs)

        learning_rate = 0.001
        for _ in range(10):  # Number of iterations for gradient descent
            grad = gradient(worst_case_probs)
            worst_case_probs -= learning_rate * grad
            worst_case_probs = np.clip(worst_case_probs, 0, 1)
            worst_case_probs /= np.sum(worst_case_probs)

        return worst_case_probs

    def compute_advantages(self, states, rewards, next_states, dones):
        advantages = np.zeros_like(rewards, dtype=float)
        last_advantage = 0
        last_value = self.value_table[next_states[-1]] * (1 - dones[-1])

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.discount_factor * self.value_table[next_states[t]] * (1 - dones[t]) - \
                    self.value_table[states[t]]
            advantages[t] = delta + self.discount_factor * self.lambda_ * last_advantage * (1 - dones[t])
            last_advantage = advantages[t]

        return advantages

    def compute_kl_divergence(self, states, old_policy):
        old_policy_probs = old_policy[states]
        new_policy_probs = self.policy[states]
        kl = np.sum(old_policy_probs * (np.log(old_policy_probs + 1e-8) - np.log(new_policy_probs + 1e-8)), axis=1)
        return np.mean(kl)

    def compute_alpha(self, old_policy, new_policy):
        return np.mean(np.abs(new_policy - old_policy))

    def compute_beta(self, old_uncertainty, new_uncertainty):
        return np.max(np.abs(new_uncertainty - old_uncertainty))

    def update_policy(self, states, actions, advantages, old_policy):
        unique_states = np.unique(states)

        def objective(flat_policy):
            self.policy[unique_states] = flat_policy.reshape(-1, self.action_space)
            surrogate_loss = self.compute_surrogate_loss(states, actions, advantages, old_policy)

            alpha = self.compute_alpha(old_policy, self.policy)
            beta = self.compute_beta(
                self.compute_worst_case_transition_probability(states[0], actions[0]),
                self.target_transition_model[states[0], actions[0]] / np.sum(
                    self.target_transition_model[states[0], actions[0]])
            )

            robust_term = (self.discount_factor ** 2 / (1 - self.discount_factor) ** 3) * (beta ** 2 + 2 * alpha * beta)
            robust_term += (2 * self.discount_factor / (1 - self.discount_factor) ** 2) * (
                        2 * alpha ** 2 + alpha * beta) * self.epsilon

            return surrogate_loss + self.regularization * robust_term

        def constraint(flat_policy):
            self.policy[unique_states] = flat_policy.reshape(-1, self.action_space)
            return self.max_kl - self.compute_kl_divergence(unique_states, old_policy)

        flat_policy = self.policy[unique_states].flatten()

        results = minimize(
            objective,
            flat_policy,
            method='SLSQP',
            constraints={'type': 'ineq', 'fun': constraint},
            options={'maxiter': 50}
        )

        self.policy[unique_states] = results.x.reshape(-1, self.action_space)
        self.policy[unique_states] /= self.policy[unique_states].sum(axis=1, keepdims=True)