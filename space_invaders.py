import gymnasium as gym
import ale_py
import numpy as np
from tabulate import tabulate
import cv2
from tqdm import tqdm
import argparse
import os

argparser = argparse.ArgumentParser(description="Train a Q-learning agent on Space Invaders")
argparser.add_argument("--play", action="store_true", help="Skip training and use a saved q-table.")
args = argparser.parse_args()

def preprocess(observation, num_bins=10):
    input_image = observation
    input_image[195:210 ,:, :] = 0
    input_image[0:27, :, :] = 0

    gray = input_image.mean(axis=-1)

    shooter_mask = (gray < 80) & (gray > 75)
    invader_mask = gray == 99
    bullet_mask = gray == 142

    # Get shooter position
    _, _, _, shooter_centroids = cv2.connectedComponentsWithStats(shooter_mask.astype(np.uint8), connectivity=8)
    shooter_x = shooter_centroids[1][0] if len(shooter_centroids) > 1 else 0

    # Get invader positions
    _, _, _, invader_centroids = cv2.connectedComponentsWithStats(invader_mask.astype(np.uint8))
    invader_x = invader_centroids[1][0] if len(invader_centroids) > 1 else 0

    # Get bullet position (Y-axis is more useful)
    _, _, _, bullet_centroids = cv2.connectedComponentsWithStats(bullet_mask.astype(np.uint8), connectivity=8)
    bullet_y = bullet_centroids[1][1] if len(bullet_centroids) > 1 else 210

    # Bucket positions
    shooter_bin = int(shooter_x * num_bins // 160)
    invader_bin = int(invader_x * num_bins // 160)
    bullet_bin = int(bullet_y * num_bins // 210)

    return observation, (shooter_bin, invader_bin, bullet_bin)

def print_q_table(Q, episode):
    os.system('clear')  # Use 'cls' on Windows
    print(f"\nQ-table snapshot after episode {episode} (showing first few entries):\n")
    for i, (state, actions) in enumerate(list(Q.items())[:10]):
        print(f"State {state}: {np.round(actions, 2)}")

# Setup environment
gym.register_envs(ale_py)
env = gym.make('ALE/SpaceInvaders-v5', render_mode='rgb_array')

# Q-table and parameters
num_actions = 6
Q = {}  # Q[state] = np.array of action values
alpha = 0.3
gamma = 0.99
epsilon = 0.2

# Train Q table
if not args.play:
    num_episodes = 1000
    for i in tqdm(range(num_episodes), desc="Training Live View"):
        obs, _ = env.reset()
        _, q_state = preprocess(obs)
        terminated = truncated = False
        while not (terminated or truncated):
            if q_state not in Q:
                Q[q_state] = np.zeros(num_actions)

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[q_state])

            obs, reward, terminated, truncated, info = env.step(action)
            _, next_q_state = preprocess(obs)

            if next_q_state not in Q:
                Q[next_q_state] = np.zeros(num_actions)

            Q[q_state][action] = (1 - alpha) * Q[q_state][action] + \
                                  alpha * (reward + gamma * np.max(Q[next_q_state]))
            q_state = next_q_state

        if (i + 1) % 100 == 0:
            print_q_table(Q, i + 1)

    np.save("q_table.npy", Q, allow_pickle=True)
else:
    # Load the saved Q table
    with open("q_table.npy", "rb") as f:
        Q_loaded = np.load(f, allow_pickle=True).item()
    Q.update(Q_loaded)

# Test Q table
while True:
    obs, _ = env.reset()
    _, q_state = preprocess(obs)
    terminated = truncated = False

    while not (terminated or truncated):
        frame = env.render()
        cv2.imshow("Space Invaders (Trained Agent)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            env.close()
            cv2.destroyAllWindows()
            exit()

        if q_state not in Q:
            Q[q_state] = np.zeros(num_actions)

        action = np.argmax(Q[q_state])
        obs, reward, terminated, truncated, info = env.step(action)
        _, q_state = preprocess(obs)
