import gym

env = gym.make('CliffWalking-v0')

env.reset()
env.render()

print("State: ", env.observation_space)
print("Action: ", env.action_space)

"""
There are 4 discrete deterministic actions:
0: move up
1: move right
2: move down
3: move left
"""

total_reward_list = []

for i in range(5):
    env.reset()
    time_step = 0
    total_reward = 0
    list_visualize = []
    
    while True:
        time_step += 1
        
        # Choose Action
        action = env.action_space.sample()
        
        # Preform action and get reward
        state, reward, done, _ , _ = env.step(action)
        
        # Total Reward
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