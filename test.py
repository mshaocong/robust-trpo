import unittest
from frozen_lake import FrozenLake
import numpy as np

class FrozenLakeTests(unittest.TestCase):

    def test_initialization_with_default_parameters_creates_4x4_lake(self):
        lake = FrozenLake()
        self.assertEqual(lake.size, 4)
        self.assertEqual(len(lake.hole_positions), 4)
        self.assertEqual(lake.goal_position, 15)

    def test_reset_starts_player_at_position_zero(self):
        lake = FrozenLake()
        initial_position = lake.reset()
        self.assertEqual(initial_position, 0)

    def test_step_without_slipping_moves_player_in_intended_direction(self):
        lake = FrozenLake(p=1.0)  # Ensure no slipping
        lake.reset()
        new_position, _, _, _ = lake.step(1)  # Move down
        self.assertEqual(new_position, lake.size)

    def test_step_with_slipping_moves_player_beyond_intended_direction(self):
        lake = FrozenLake(size=5, p=0.0)  # Ensure slipping
        lake.reset()
        new_position, _, _, _ = lake.step(1)  # Attempt to move down
        self.assertTrue(new_position == 2 * lake.size or new_position == 0)

    def test_generate_data_produces_correct_length_dataset(self):
        lake = FrozenLake()
        policy = np.ones((lake.observation_space, lake.action_space)) / lake.action_space
        data = lake.generate_data(policy, num_episodes=10, max_steps_per_episode=5)
        self.assertTrue(len(data) <= 50)  # 10 episodes * 5 steps max

    def test_compute_value_function_converges_for_random_policy(self):
        lake = FrozenLake()
        policy = lake.generate_random_policy()
        V = lake.compute_value_function(policy, gamma=0.99)
        self.assertTrue(np.all(V >= -1.0) and np.all(V <= 1.0))

    def test_compute_q_function_returns_correct_shape(self):
        lake = FrozenLake()
        policy = lake.generate_random_policy()
        Q = lake.compute_q_function(policy, gamma=0.99)
        self.assertEqual(Q.shape, (lake.observation_space, lake.action_space))

    def test_generate_transition_matrix_creates_correct_shape(self):
        lake = FrozenLake()
        P = lake.generate_transition_matrix()
        self.assertEqual(P.shape, (lake.observation_space, lake.action_space, lake.observation_space))

    def test_get_transition_prob_returns_zero_for_invalid_transitions(self):
        lake = FrozenLake()
        prob = lake.get_transition_prob(0, 0, lake.observation_space - 1)  # From start to goal directly
        self.assertEqual(prob, 0.0)

    def test_render_displays_correct_initial_state(self):
        lake = FrozenLake()
        lake.reset()
        with self.assertLogs() as cm:
            lake.render()
        self.assertIn('P . . .', cm.output[0])

if __name__ == '__main__':
    unittest.main()