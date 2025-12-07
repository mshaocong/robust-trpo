import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
from scipy.signal import savgol_filter
import matplotlib.cm as cm

# Constants for file paths
EPISODE_REWARDS_PATH_BASIC = 'Basic_ep100000_episode_rewards.csv'
MOVING_AVERAGES_PATH_BASIC = 'Basic_ep100000_moving_average.csv'
EPISODE_REWARDS_PATH_TRPO = 'TRPO_ep100000_episode_rewards.csv'
MOVING_AVERAGES_PATH_TRPO = 'TRPO_ep100000_moving_average.csv'
EPISODE_REWARDS_PATH_ROBUST_TRPO = 'RobustTRPO_ep100000_episode_rewards.csv'
MOVING_AVERAGES_PATH_ROBUST_TRPO = 'RobustTRPO_ep100000_moving_average.csv'
EVALUATION_REWARDS_PATH_TRPO = 'TRPO_ep100000_evaluation_rewards.csv'
EVALUATION_REWARDS_PATH_ROBUST_TRPO = 'RobustTRPO_ep100000_evaluation_rewards.csv'

def smooth_data(data, window_size=101, polyorder=5):
    """
    Apply Savitzky-Golay filter to smooth the data.
    """
    return savgol_filter(data, window_size, polyorder)

def moving_average(data, window_size=10):
    """
    Calculate moving average with a specific window size.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def load_data(episode_rewards_path, moving_averages_path):
    """
    Load episode rewards and moving averages from CSV files.
    """
    episode_rewards = pd.read_csv(episode_rewards_path)
    moving_averages = pd.read_csv(moving_averages_path)
    return episode_rewards, moving_averages

def plot_data(episode_rewards, moving_averages, label_prefix, color_index, smoothing=None):
    """
    Plot episode rewards and moving averages with optional smoothing, using tab10 colormap.
    """
    rewards = episode_rewards['Reward'].values
    if smoothing == 'moving_average':
        rewards = moving_average(rewards)
    elif smoothing == 'savgol':
        rewards = smooth_data(rewards)

    color = cm.tab10(color_index)
    # plt.plot(episode_rewards['Episode'][:len(rewards)], rewards, label=f'{label_prefix} Episode Reward' if smoothing else f'{label_prefix} Reward', color=color)
    plt.plot(moving_averages['Episode'], moving_averages['Moving Average'], label=f'{label_prefix}', linewidth=3, color=color)

def visualize_comparison(smoothing=None):
    """
    Visualize comparison between TRPO, RobustTRPO, and Basic method data using tab10 colormap.
    """
    # episode_rewards_basic, moving_averages_basic = load_data(EPISODE_REWARDS_PATH_BASIC, MOVING_AVERAGES_PATH_BASIC)
    episode_rewards_trpo, moving_averages_trpo = load_data(EPISODE_REWARDS_PATH_TRPO, MOVING_AVERAGES_PATH_TRPO)
    episode_rewards_robust_trpo, moving_averages_robust_trpo = load_data(EPISODE_REWARDS_PATH_ROBUST_TRPO, MOVING_AVERAGES_PATH_ROBUST_TRPO)

    plt.figure(figsize=(8, 8))
    # plot_data(episode_rewards_basic, moving_averages_basic, 'Basic', 0, smoothing)
    plot_data(episode_rewards_trpo, moving_averages_trpo, 'TRPO', 1, smoothing)
    plot_data(episode_rewards_robust_trpo, moving_averages_robust_trpo, 'RobustTRPO', 2, smoothing)

    plt.legend()
    plt.grid(True)
    plt.savefig('comparison.png')
    plt.show()

def exponential_average(data, alpha=0.01):
    """
    Calculate exponential moving average with a specific alpha.
    """
    exp_avg = [data[0]]  # Start with the first data point
    for point in data[1:]:
        exp_avg.append(alpha * point + (1 - alpha) * exp_avg[-1])
    return exp_avg

def plot_comparision(smoothing=None):
    """
    Plot comparison on moving average and evaluation rewards between TRPO and RobustTRPO method data using tab10 colormap.
    """
    _, moving_averages_trpo = load_data(EPISODE_REWARDS_PATH_TRPO, MOVING_AVERAGES_PATH_TRPO)
    _, moving_averages_robust_trpo = load_data(EPISODE_REWARDS_PATH_ROBUST_TRPO, MOVING_AVERAGES_PATH_ROBUST_TRPO)
    evaluation_rewards_trpo = pd.read_csv(EVALUATION_REWARDS_PATH_TRPO)
    evaluation_rewards_robust_trpo = pd.read_csv(EVALUATION_REWARDS_PATH_ROBUST_TRPO)

    if smoothing == 'moving_average':
        moving_averages_trpo['Moving Average'] = moving_average(moving_averages_trpo['Moving Average'])
        moving_averages_robust_trpo['Moving Average'] = moving_average(moving_averages_robust_trpo['Moving Average'])
        evaluation_rewards_trpo['Evaluation Reward'] = moving_average(evaluation_rewards_trpo['Evaluation Reward'])
        evaluation_rewards_robust_trpo['Evaluation Reward'] = moving_average(evaluation_rewards_robust_trpo['Evaluation Reward'])
    elif smoothing == 'savgol':
        moving_averages_trpo['Moving Average'] = smooth_data(moving_averages_trpo['Moving Average'])
        moving_averages_robust_trpo['Moving Average'] = smooth_data(moving_averages_robust_trpo['Moving Average'])
        evaluation_rewards_trpo['Evaluation Reward'] = smooth_data(evaluation_rewards_trpo['Evaluation Reward'])
        evaluation_rewards_robust_trpo['Evaluation Reward'] = smooth_data(evaluation_rewards_robust_trpo['Evaluation Reward'])
    elif smoothing == 'exponential':
        moving_averages_trpo['Moving Average'] = exponential_average(moving_averages_trpo['Moving Average'])
        moving_averages_robust_trpo['Moving Average'] = exponential_average(moving_averages_robust_trpo['Moving Average'])
        evaluation_rewards_trpo['Evaluation Reward'] = exponential_average(evaluation_rewards_trpo['Evaluation Reward'])
        evaluation_rewards_robust_trpo['Evaluation Reward'] = exponential_average(evaluation_rewards_robust_trpo['Evaluation Reward'])

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(moving_averages_trpo['Episode'], moving_averages_trpo['Moving Average'], label='TRPO', color=cm.tab10(1))
    plt.plot(moving_averages_robust_trpo['Episode'], moving_averages_robust_trpo['Moving Average'], label='RobustTRPO', color=cm.tab10(2))
    plt.xlabel('Episode')
    plt.ylabel('Moving Average')
    plt.title('Moving Average Comparison')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(evaluation_rewards_trpo['Episode'], evaluation_rewards_trpo['Evaluation Reward'], label='TRPO', color=cm.tab10(1))
    plt.plot(evaluation_rewards_robust_trpo['Episode'], evaluation_rewards_robust_trpo['Evaluation Reward'], label='RobustTRPO', color=cm.tab10(2))
    plt.xlabel('Episode')
    plt.ylabel('Evaluation Reward')
    plt.title('Evaluation Reward Comparison')
    plt.legend()
    plt.grid(True)

    plt.savefig('comparison_new.png')
    plt.show()
import seaborn as sns
def smooth_data(data, window_size=101, polyorder=5):
    """
    Apply Savitzky-Golay filter to smooth the data.
    """
    return savgol_filter(data, window_size, polyorder)

def plot_boxplot_evaluation_rewards():
    """
    Plot box plot for smoothed evaluation rewards between TRPO and RobustTRPO method data using seaborn.
    """
    evaluation_rewards_trpo = pd.read_csv(EVALUATION_REWARDS_PATH_TRPO)
    evaluation_rewards_robust_trpo = pd.read_csv(EVALUATION_REWARDS_PATH_ROBUST_TRPO)

    smoothed_rewards_trpo = smooth_data(evaluation_rewards_trpo['Evaluation Reward'])
    smoothed_rewards_robust_trpo = smooth_data(evaluation_rewards_robust_trpo['Evaluation Reward'])

    data = pd.DataFrame({
        'Evaluation Reward': pd.concat([pd.Series(smoothed_rewards_trpo), pd.Series(smoothed_rewards_robust_trpo)]),
        'Method': ['TRPO'] * len(smoothed_rewards_trpo) + ['RobustTRPO'] * len(smoothed_rewards_robust_trpo)
    })

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Method', y='Evaluation Reward', data=data, palette='tab10')
    plt.xlabel('Method')
    plt.ylabel('Evaluation Reward')
    plt.title('Smoothed Evaluation Reward Box Plot Comparison')
    plt.grid(True)
    plt.savefig('boxplot_comparison_smoothed.png')
    plt.show()

if __name__ == "__main__":
    plot_boxplot_evaluation_rewards()
    # # visualize_comparison(smoothing='savgol')
    # plot_comparision(smoothing='exponential')