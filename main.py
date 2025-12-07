import numpy as np
import time
from frozen_lake import FrozenLake, ACTION_MAPPING


def simulate_episode(env, policy, max_steps=100):
    state = env.reset()
    done = False
    steps = 0
    total_reward = 0

    print("Starting a new episode...")
    env.render()
    time.sleep(1)

    while not done and steps < max_steps:
        action = np.random.choice(env.action_space, p=policy[state])
        next_state, reward, done, _ = env.step(action)

        print(f"Step {steps + 1}:")
        print(f"Action: {ACTION_MAPPING[action]}")
        env.render()

        total_reward += reward
        state = next_state
        steps += 1

        time.sleep(0.5)

    if done:
        if state == env.goal_position:
            print("Goal reached!")
        elif state in env.hole_positions:
            print("Fell in a hole!")
    else:
        print("Maximum steps reached.")

    print(f"Episode finished after {steps} steps. Total reward: {total_reward}")


def main():
    # Create FrozenLake environment
    env = FrozenLake(size=4, p=0.8)

    # Generate a random policy
    random_policy = env.generate_random_policy()

    # Simulate episodes
    num_episodes = 5
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        simulate_episode(env, random_policy)
        time.sleep(2)


if __name__ == "__main__":
    main()