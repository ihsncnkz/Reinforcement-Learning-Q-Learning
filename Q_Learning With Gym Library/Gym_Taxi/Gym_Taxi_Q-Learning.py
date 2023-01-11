# import library
import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3",render_mode = "ansi").env

# Q Table
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameter
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# Plottinig Metrix
reward_list = []
droputs_list = []

episode_number = 10000
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
        
        # action process and take reward / observation
        next_state, reward, done, _ , _ = env.step(action)
        
        # Q Learning Fuction
        old_value = q_table[state, action] # old_value
        next_max = np.max(q_table[next_state]) # next_max
        
        next_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        
        # Q table update
        q_table[state,action] = next_value
        
        # update state
        state = next_state
        
        # Find wrong droputs
        if reward == -10:
            dropouts += 1
        
        reward_count += reward
        
        if done:
            break
        
    if i%10 == 0:
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

# Q Table


"""
Actions: 
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east 
    - 3: move west 
    - 4: pickup passenger
    - 5: dropoff passenger
"""  
# taxi row, taxi, column, passenger index, destination 
       
env.s = env.encode(0,0,3,4) 
env.render()   

  
env.s = env.encode(4,4,4,3) 
env.render()  
























        