# Import necessary modules and classes
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from frozen_lake import FrozenLake, ACTION_MAPPING
from agent import Agent, QLearning, TRPOAgent, RobustTRPO
import csv, json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Function to select the agent
def select_agent(agent_type, env):
    if agent_type == 'QLearning':
        return QLearning(env.observation_space, env.action_space)
    elif agent_type == 'TRPOAgent':
        return TRPOAgent(env.observation_space, env.action_space)
    elif agent_type == 'RobustTRPOAgent':
        return RobustTRPO(env.observation_space, env.action_space)
    else:
        return Agent(env.observation_space, env.action_space)


def compute_state_action_occupancy(env, policy, gamma=0.99, num_iterations=1000):
    """
    Compute the state-action occupancy measure for a given policy.
    """
    S = env.observation_space
    A = env.action_space

    # Initialize state-action occupancy measure
    occupancy = np.zeros((S, A))

    # Initial state distribution (assuming always starting at state 0)
    initial_distribution = np.zeros(S)
    initial_distribution[0] = 1

    # Compute state transition probabilities under the given policy
    P_pi = np.sum(env.P * policy[:, :, np.newaxis], axis=1)

    # Iterative computation of state occupancy measure
    state_occupancy = np.zeros(S)
    for _ in range(num_iterations):
        state_occupancy = initial_distribution + gamma * P_pi.T @ state_occupancy

    # Compute state-action occupancy measure
    for s in range(S):
        for a in range(A):
            occupancy[s, a] = state_occupancy[s] * policy[s, a]

    # Normalize the occupancy measure
    occupancy /= np.sum(occupancy)

    return occupancy


def sample_state_action(occupancy):
    """
    Sample a state-action pair from the occupancy measure.
    """
    flat_occupancy = occupancy.flatten()
    index = np.random.choice(len(flat_occupancy), p=flat_occupancy)
    state = index // occupancy.shape[1]
    action = index % occupancy.shape[1]
    return state, action


def sample_next_state(env, state, action):
    probabilities = env.P[state, action, :]
    next_state = np.random.choice(env.observation_space, p=probabilities)
    reward = env.R[next_state]
    done = next_state == env.goal_position or next_state in env.hole_positions
    return next_state, reward, done


def simulate_episode(env, agent, occupancy, max_samples=100, verbose=False):
    total_reward = 0
    samples = 0

    if verbose:
        logger.info("Starting a new episode (sampling from occupancy measure)...")

    for _ in range(max_samples):
        state, action = sample_state_action(occupancy)
        next_state, reward, done = sample_next_state(env, state, action)

        if verbose:
            logger.debug(f"Sample {samples + 1}: State: {state}, Action: {ACTION_MAPPING[action]}, "
                         f"Next State: {next_state}, Reward: {reward}")

        agent.update(state, action, reward, next_state, done)

        total_reward += reward
        samples += 1

        if done:
            break

    if verbose:
        logger.info(f"Episode finished after {samples} samples. Total reward: {total_reward}")

    return total_reward


def plot_rewards(rewards, window_size=100, save_path='reward_curve.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Episode Reward')
    moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
    plt.plot(moving_avg, label=f'Moving Average ({window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    return moving_avg


def save_data(rewards, eval_rewards, moving_avg, params, results, agent_type):
    # Format file names to include agent type and key hyper-parameters
    base_filename = f"{agent_type}_ep{params['num_episodes']}"
    rewards_filename = f"{base_filename}_episode_rewards.csv"
    eval_rewards_filename = f"{base_filename}_evaluation_rewards.csv"
    moving_avg_filename = f"{base_filename}_moving_average.csv"
    training_info_filename = f"{base_filename}_training_info.json"

    # Save episode rewards
    with open(rewards_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Reward'])
        for i, reward in enumerate(rewards):
            writer.writerow([i + 1, reward])

    # Save evaluation rewards
    with open(eval_rewards_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Evaluation Reward'])
        for i, reward in enumerate(eval_rewards):
            writer.writerow([i * params['log_interval'] + 1, reward])

    # Save moving average
    with open(moving_avg_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Moving Average'])
        for i, avg in enumerate(moving_avg):
            writer.writerow([i + 1, avg])

    # Save parameters and results
    with open(training_info_filename, 'w') as f:
        json.dump({**params, **results}, f, indent=4)


# Ensure to pass the agent_type and any other relevant information to save_data within the train function


def train(env, agent, num_episodes=1000, max_samples_per_episode=100, log_interval=10, plot_interval=100):
    rewards = []
    eval_rewards = []
    start_time = time.time()

    for episode in tqdm(range(num_episodes), desc="Training Progress"):
        # Compute the current policy from the agent
        policy = np.array([agent.get_action_probabilities(s) for s in range(env.observation_space)])

        # Compute the occupancy measure for the current policy
        occupancy = compute_state_action_occupancy(env, policy)

        # Simulate episode using occupancy measure
        episode_reward = simulate_episode(env, agent, occupancy, max_samples_per_episode,
                                          verbose=(episode % log_interval == 0))

        rewards.append(episode_reward)

        agent.decay_epsilon()

        eval_avg_reward = -1.0
        if episode % log_interval == 0:
            avg_reward = np.mean(rewards[-log_interval:])
            logger.info(f"Episode {episode + 1}/{num_episodes} - Average Reward: {avg_reward:.2f}")

            eval_env = FrozenLake(size=4, p=0.8)
            occupancy = compute_state_action_occupancy(eval_env, policy)
            eval_reward = [simulate_episode(eval_env, agent, occupancy, max_samples_per_episode) for _ in range(10)]
            eval_avg_reward = np.mean(eval_reward)
            logger.info(f"Evaluation - Average Reward: {eval_avg_reward:.2f}")
        eval_rewards.append(eval_avg_reward)

        if episode % plot_interval == 0:
            plot_rewards(rewards)

    plot_rewards(eval_rewards, save_path='evaluation_curve.png')
    # Final plot and get moving average
    moving_avg = plot_rewards(rewards)

    end_time = time.time()
    training_time = end_time - start_time

    # Prepare parameters and results
    params = {
        "num_episodes": num_episodes,
        "max_samples_per_episode": max_samples_per_episode,
        "log_interval": log_interval,
        "plot_interval": plot_interval
    }
    results = {
        "training_time": training_time,
        "final_avg_reward": np.mean(rewards[-100:]),
        "final_eval_avg_reward": eval_rewards[-1] if eval_rewards else None
    }

    # Save data with agent_type and evaluation rewards
    save_data(rewards, eval_rewards, moving_avg, params, results, agent.agent_type)

    return rewards, params, results


# Main function updated to accept agent type
def main(agent_type='Basic'):
    # Create FrozenLake environment
    env = FrozenLake(size=4, p=1.0)

    # Select agent based on the provided type
    agent = select_agent(agent_type, env)

    # Train the agent
    logger.info(f"Starting training with {agent_type} agent...")
    rewards, params, results = train(env, agent, num_episodes=100000)

    logger.info(f"Training completed in {results['training_time']:.2f} seconds")
    logger.info(f"Final average reward (last 100 episodes): {results['final_avg_reward']:.2f}")


# Ensure the rest of the functions like compute_state_action_occupancy, simulate_episode, plot_rewards, save_data, and train are defined here as well.

if __name__ == "__main__":
    # Example usage: main('PolicyGradient')
    # main('QLearning')
    # main('TRPOAgent')
    main('RobustTRPOAgent')
    # main('Basic')
