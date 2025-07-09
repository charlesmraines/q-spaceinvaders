import gymnasium as gym
import ale_py
import numpy as np
from tabulate import tabulate
from sklearn.cluster import KMeans
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import argparse
import os
import statistics

# shooter, space invaders, bullets
color_buckets = [[0, 135, 34], [133, 135, 0], [142, 142, 142]]

argparser = argparse.ArgumentParser(description="Train a Q-learning agent on Space Invaders")
argparser.add_argument("--play", action="store_true", help="Use this tag to skip training ans use a saved q-table.")

args = argparser.parse_args()



def preprocess(observation):
    input_image = observation
    input_image[195:210 ,:, :] = 0
    input_image[0:27, :, :] = 0

    grey = input_image.mean(axis=-1)


    shooter_binary = (grey < 80) & (grey > 75)
    invaders_binary = grey == 99
    bullets_binary = grey == 142



    _, shooter_labels, shooter_values, shooter_centroids = cv2.connectedComponentsWithStats(shooter_binary.astype(np.uint8), connectivity=8)
    _, invaders_labels, _, invaders_centroids = cv2.connectedComponentsWithStats(invaders_binary.astype(np.uint8))
    _, bullets_labels, _, bullets_centroids = cv2.connectedComponentsWithStats(bullets_binary.astype(np.uint8), connectivity=8)

    state = 0

    if len(shooter_centroids) > 1:
        for i in invaders_centroids:
            if abs((i + 1)[1] - shooter_centroids[1][0]) < 8:
                state = 1
                break
    

    

    return observation, state



gym.register_envs(ale_py)

# Create an instance of the SpaceInvaders environment
env = gym.make('ALE/SpaceInvaders-v5', render_mode='rgb_array')

# Initialize Q table
num_actions = 6
num_states = 2
Q = np.zeros((num_states, num_actions))

def print_q_table(Q, episode):
    os.system('clear')  # Use 'cls' on Windows
    print(f"\nQ-table after episode {episode}:\n")
    headers = [f"Action {i}" for i in range(Q.shape[1])]
    table = tabulate(Q, headers=headers, showindex=["State 0", "State 1"], floatfmt=".2f")
    print(table)


# Set hyperparameters
alpha = 0.3  # learning rate
gamma = 0.99  # discount factor
epsilon = 0.2  # exploration rate

# Train Q table
if not args.play:
    num_episodes = 1000
    for i in tqdm(range(num_episodes), desc="Training Live View"):
        state, _ = env.reset()
        state, q_state = preprocess(state)
        terminated = truncated = False
        while not (terminated or truncated):
            # Select action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # explore
            else:
                #print(state)
                action = np.argmax(Q[q_state])  # exploit
            
            # Take action and observe next state and reward
            observation, reward, terminated, truncated, info = env.step(action)
            frame = env.render()
            observation, q_state = preprocess(observation)
            
            # Update Q table
            Q[q_state, action] = (1 - alpha) * Q[q_state, action] + alpha * (reward + gamma * np.max(Q[:, action]))
            
            # Update state
            state = observation
        
        # Update plot
        if (i + 1) % 1 == 0:
            print_q_table(Q, i + 1)

    np.save("q_table.npy", Q)
else:
    # Load the saved Q table
    Q = np.load("q_table.npy")



# Test Q table
state = env.reset()
done = False
while not done:
    frame = env.render()

    cv2.imshow("Space Invaders (Trained Agent)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    action = np.argmax(Q[q_state])
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

# Close the environment
env.close()
cv2.destroyAllWindows()
