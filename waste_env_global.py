import gym
import copy
import numpy as np
import pandas as pd
from gym import spaces


START_STATE = [[10, 10, 50],
               [120, 1, 20],
               [15, 17, 40],
               [15, 17, 10],
               [15, 17, 10],]

C = [3, 20, 4, 4, 4] #Min selling. This decides how popular an item is.
K = [0.80, 0.10, 0.60, 0.60, 0.60] #This constant decides how robust each item is to price.
P = [50, 20, 40, 10, 10] #Base price of items

C, K, P = np.array(C), np.array(K), np.array(P)
K *= C
START_STATE = np.array(START_STATE)

class WasteEnv(gym.Env):

    """Environment to handle waste and dynamic pricing"""

    def __init__(self, N, lambda1=0.5, lambda2=0.5):
        super(WasteEnv, self).__init__()
        # self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.N = N
        self.action_space = spaces.Box(low=-100, high=100, shape=
                                (1, N), dtype=np.uint8)
        self.observation_space = spaces.Box(low=0, high=100, shape=
                                (N, 2), dtype=np.uint8)
        self.state = "something"
        self.lambda1, self.lambda2 = lambda1, lambda2

    def step(self, action):
        # Execute one time step within the environment
        return self.take_action(action)

    def take_action(self, action):

        demand = [K[i]*a for i, a in enumerate(action)]
        sold = C - demand + np.random.randn(self.N)/10
        print(f"K: {K}")
        print(f"K: {type(K)}")
        print(f"DEMAND: {demand}")
        print(f"SOLD: {sold}")

        profit = 0
        total_profit = 0
        total_items_left = 0
        wasted = 0
        total_wasted = 0
        for i, sold_i in enumerate(sold):
            sold_i = max(sold_i, 0)
            if self.state[i][1] > 0: #Sell only if item is not expired
                self.state[i][0] = max(self.state[i][0]-sold_i, 0)
            print(f"==== ITEM: [{i}]")
            print(f"Price change: {action[i]}")
            print(f"Original: {START_STATE[i]}")
            print(f"Sold: {sold_i}")
            profit = sold_i*action[i]*P[i] / (START_STATE[i][0]*P[i])
            total_profit += profit
            print(f"Profit on this item: {profit}")
            self.state[i][1] = max(self.state[i][1]-1, 0) #Reduce item's life
            if self.state[i][1] == 0: #If no life, rest of the remaining items are wasted
                wasted += self.state[i][0]
                wasted /= START_STATE[i][0] #Normalize
                print(f"This item wasted: {wasted}")
                total_wasted += wasted
                self.state[i][0] = 0 #No more items left, all wasted.
            total_items_left += self.state[i][0]
            # Update price of items
            self.state[i][2] += self.state[i][2]*action[i]

        print(f" ===================== ")
        print(f"Profit: {total_profit}")
        print(f"Wasted: {total_wasted}")
        reward = self.lambda1*total_profit - self.lambda2*total_wasted
        if total_items_left:
            self.done = False
        else:
            self.done = True
        print(f"FINAL REWARD: {reward}")

        return self.state, reward, self.done, "something"



    def reset(self):
        self.state = copy.deepcopy(START_STATE)
        return self.state

    def render(self):
        print("Kek")

if __name__ == "__main__":
    env = WasteEnv(5)
    s = env.reset()
    print(s)
    action = np.array([-20, 20, 0, -10, 10])/100

    s, r, done, _ = env.step(action)
    print(f"State: {s}")
