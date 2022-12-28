import gym
import time

env = gym.make('FrozenLake-v1')


env.reset()
env.render()

print("State: ", env.observation_space)
print("Action: ", env.action_space)

"""
Action Space
The agent takes a 1-element vector for actions. The action space is (dir), where dir decides direction to move in which can be:

0: LEFT
1: DOWN
2: RIGHT
3: UP
"""
total_reward_list = []

for i in range(5):
    env.reset()
    time_step = 0
    total_reward = 0
    list_visualize = []
    while True:
        time_step +=1
        
        action = env.action_space.sample()
        
        state , reward, done, _ , _ = env.step(action)
        
        total_reward += reward
        
        list_visualize.append({"frame" : env.render(),
                               "state" : state,
                               "action" : action,
                               "reward" : reward,
                               "Total Reward" : total_reward})
        
        if done:
            total_reward_list.append(total_reward)
            
            list_visualize.append({"frame" : env.render(),
                                   "state" : state,
                                   "action" : action,
                                   "reward" : reward,
                                   "Total Reward" : total_reward})
            
            break
        
for i, frame in enumerate(list_visualize):
    print(frame["frame"])
    print("Time Step: ", i + 1)
    print("State: ", frame["state"])
    print("action: ", frame["action"]) 
    print("reward: ", frame["reward"]) 
    print("Total Reward: ", frame["Total Reward"])
    time.sleep(1)



        