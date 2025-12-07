import gymnasium as gym
import keyboard
import time

def get_action_from_key():
    while True:
        if keyboard.is_pressed('up'):
            return 3  # Up
        elif keyboard.is_pressed('down'):
            return 1  # Down
        elif keyboard.is_pressed('left'):
            return 0  # Left
        elif keyboard.is_pressed('right'):
            return 2  # Right

if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="human")

    observation, info = env.reset()
    env.render()

    while True:
        action = get_action_from_key()
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            print("Episode finished. Press 'r' to reset or 'q' to quit.")
            while True:
                if keyboard.is_pressed('r'):
                    observation, info = env.reset()
                    env.render()
                    break
                elif keyboard.is_pressed('q'):
                    env.close()
                    exit()

        time.sleep(0.1)  # Small delay to prevent the game from running too fast