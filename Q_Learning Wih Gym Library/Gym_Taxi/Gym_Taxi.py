import gym

env = gym.make("Taxi-v3",render_mode = "ansi").env

env.reset()
env.render()

        
"""
Blue : Passenger
Purple : Destination
Yellow / Red : Empty taxi
Green : Full Taxi
RGBY : Location for destination and passanger
"""

#%% 

print("State: ", env.observation_space)
print("Action: ", env.action_space)

#taxi row, taxi column, passenger index, destination
state = env.encode(3,1,2,3)
print("State Number: ",state)

env.s = state

# env.reset()
# env.render()

"""
There are 6 discrete deterministic actions:

0: move south
1: move north
2: move east
3: move west
4: pickup passenger
5: drop off passenger

"""
print("Probability, next_state, reward, done")
for i in env.P[state]:
    print("{} : {}".format(i, env.P[state][i]))
    
#%%
total_reward_list = []

# 5 Episode
for i in range(5):
    env.reset()
    time_step = 0
    total_reward = 0
    list_visualize = []
    while True:
        
        time_step +=1 
        
        #Choose action
        action = env.action_space.sample()
        
        # Preform action and get reward
        state, reward, done, _ , _= env.step(action)
        
        # total reward
        total_reward += reward
        
        # Visualization
        list_visualize.append({"frame" : env.render(),
                               "state" : state,
                               "action" : action,
                               "reward" : reward,
                               "Total Reward" : total_reward})
        
        if done:
            total_reward_list.append(total_reward)
            break

import time
for i, frame in enumerate(list_visualize):
    print(frame["frame"])
    print("Time Step: ", i + 1)
    print("State: ", frame["state"])
    print("action: ", frame["action"]) 
    print("reward: ", frame["reward"]) 
    print("Total Reward: ", frame["Total Reward"])
    #time.sleep(1)
    
    

    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
