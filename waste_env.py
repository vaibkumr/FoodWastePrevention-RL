import gym
import copy
import numpy as np
import pandas as pd
from gym import spaces
from utils import get_mapping, item_conf, seed_everything
import configparser
import argparse
seed_everything(42) #We love deterministic algorithms

# C = 2 #Min selling. This decides how popular an item is.
# K = 0.10 #This constant decides how robust each item is to price.
# P = 50 #Base price of items

class WasteEnv(gym.Env):

    """Environment to handle waste and dynamic pricing"""

    def __init__(self, START_STATE = [9, 4], C=2, K=0.10, P=50,
                lambda1=0.5, lambda2=0.5, render=False):
        super(WasteEnv, self).__init__()

        K *= C
        self.K = K
        self.C = C
        self.P = P

        self.MAPPING = get_mapping()
        self.START_STATE = np.array(START_STATE)

        self.action_space = spaces.Discrete(len(self.MAPPING))
        self.observation_space = spaces.Box(low=0, high=100, shape=(1, 2),
                                    dtype=np.uint8)
        self.state = START_STATE
        self.lambda1, self.lambda2 = lambda1, lambda2


    def step(self, action):
        # Execute one time step within the environment
        return self.take_action(action)

    def take_action(self, action):
        action = self.MAPPING[action]
        demand = self.K*action
        sold = max(self.C - demand + np.random.randn()/10, 0)

        # print(f"K: {self.K}")
        # print(f"DEMAND: {demand}")
        # print(f"SOLD: {sold}")
        # print(f"some random: {np.random.randn()/10}")

        profit = 0
        total_profit = 0
        total_items_left = 0
        wasted = 0
        total_wasted = 0
        self.state = list(self.state)
        if self.state[1] > 0: #Only sell items if not expired
            self.state[0] = int(round(max(self.state[0]-sold, 0))) #Update inventory
        profit = sold*action*self.P/(self.START_STATE[0]*self.P)
        self.state[1] = int(max(self.state[1]-1, 0)) #Update life
        if self.state[1] == 0: #If no life, rest of the remaining items are wasted
                wasted = self.state[0]
                wasted /= self.START_STATE[0]
                self.state[0] = 0
        reward = self.lambda1*profit - self.lambda2*wasted
        # print(f"Profit: {profit}")
        # print(f"Wasted: {wasted}")
        if self.state[0]:
            self.done = False
        else: #inventory empty
            self.done = True
        info = {"profit":profit, "wasted":wasted}
        # print(f"Returning state: {self.state}")
        return tuple(self.state), reward, self.done, info

    def reset(self):
        self.state = copy.deepcopy(self.START_STATE)
        return tuple(self.state)

    def render(self):
        print("Kek")

if __name__ == "__main__":
    """Test environment"""

    config = configparser.ConfigParser()
    config.read('config/config.ini')
    conf = config['ITEM1']
    print(config)
    START_STATE = [conf.getint('START_STATE_N'), conf.getint('START_STATE_L')]
    C = conf.getint('C')
    K = conf.getint('K')
    P = conf.getint('P')

    # env = WasteEnv(START_STATE = START_STATE, C=C, K=K, P=P)
    env = WasteEnv(*item_conf('ITEM1'))
    s = env.reset()
    print(s)
    action = 11
    # MAPPING is: {0: 0.0, 1: -0.1, 2: -0.2, 3: -0.3, 4: -0.4, 5: -0.5,
    # 6: -0.6, 7: -0.7, 8: -0.8, 9: -0.9, 10: 0.1, 11: 0.2, 12: 0.3,
    # 13: 0.4, 14: 0.5, 15: 0.6, 16: 0.7, 17: 0.8, 18: 0.9, 19: 1.0}


    s, r, done, _ = env.step(action)
    print(f"State: {s}")
