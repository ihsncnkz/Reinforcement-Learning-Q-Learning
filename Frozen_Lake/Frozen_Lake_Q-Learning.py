import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1')

# Q Table
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# HyperParameter
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# Plotting Matrix
reward_list = []

episode_number = 25000
for i in range(1 , episode_number):
    
    # Initialize enviroment
    state = env.reset()[0]
    reward_count = 0
    
    while True:
        
        # Expoint vs Explore to find action
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        # Action process and take reward / Obsercation
        next_state, reward, done, _ , _ = env.step(action)
        
        # Q Learning Functin
        old_value = q_table[state, action] # old_value
        next_max = np.max(q_table[next_state]) # next_max
        
        next_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        
        # Q table update
        q_table[state, action] = next_value
        
        # Update state
        state = next_state
        
        reward_count += reward
        
        if done:
            break
        
    if i%100 == 0:
        reward_list.append(reward_count)
        print("Epsilon: {}, Reward: {}".format(i, reward_count))
        
# Visiualization

plt.plot(reward_list)
plt.xlabel("episode")
plt.ylabel("reward")
        