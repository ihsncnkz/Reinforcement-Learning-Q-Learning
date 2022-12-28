import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env =  gym.make('CliffWalking-v0')

# Q Table
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# HyperParameter
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# Plotting Matrix
reward_list = []
droputs_list = []

episode_number = 75000
for i in range(1, episode_number):
    
    # initialize enviroment
    state = env.reset()[0]
    reward_count = 0
    dropouts = 0
    
    while True:
        
        # Exploit vs Explore to find action
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        # Action process and take reward / observation
        next_state, reward, done, _ , _ = env.step(action)
        
        # Q Learning Function
        old_value = q_table[state, action] # Old Value
        next_max = np.max(q_table[next_state]) # Next State
        
        next_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        
        # Q table
        q_table[state, action] = next_value
        
        # Update state
        state = next_state
        
        # Find wrong  droputs
        if reward == -100:
            dropouts += 1
        
        reward_count += reward
        
        if done:
            break
        
    if i%100 == 0:
        droputs_list.append(dropouts)
        reward_list.append(reward_count)
        print("Episode: {}, Reward: {}, Wrong Dropout: {}".format(i, reward_count, dropouts))
        
# Visualization

fig ,axs = plt.subplots(1,2)
axs[0].plot(reward_list)
axs[0].set_xlabel("episode")
axs[0].set_ylabel("reward")

axs[1].plot(droputs_list)
axs[1].set_xlabel("episode")
axs[1].set_ylabel("dropouts")

axs[0].grid(True)
axs[1].grid(True)

plt.show()
        