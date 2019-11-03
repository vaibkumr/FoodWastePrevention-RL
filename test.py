import os
import gym
import numpy as np
import time
from waste_env import WasteEnv
import pickle
from utils import get_mapping, item_conf, seed_everything
import argparse
seed_everything(42) #We love deterministic algorithms


def load_Q(fname):
    with open(fname, 'rb') as handle:
        return pickle.load(handle)

def greedy(Q, s):
    return np.argmax(Q[s])

def test_agent(Q, env, n_tests, delay=1):
    MAPPING = get_mapping()
    for test in range(n_tests):
        print(f"\n>>>>>>>>>>>> [START] Test #{test}\n")
        s = env.reset()
        done = False
        epsilon = 0
        total_profit = 0
        total_reward = 0
        while True:
            time.sleep(delay/5)
            a = greedy(Q, s)
            print(f"Chose action {a} i.e. {MAPPING[a]} for state {s}")
            s, reward, done, info = env.step(a)
            total_reward += reward
            profit = info["profit"]
            wasted = info["wasted"]
            total_profit += profit
            if done:
                print(f"End of episode")
                print(f"Total profit: {total_profit}")
                print(f"% Items wasted: {wasted}")
                print(f"Total reward: {total_reward} (Negative values are not \"bad\")")
                time.sleep(delay)
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", "-i", help="item name. See config.ini for details.")
    args = parser.parse_args()
    item_name = args.i

    fname = f"logs/{item_name}/9000.pkl"
    Q = load_Q(fname)
    n_tests = 2
    env = WasteEnv(*item_conf(item_name))
    test_agent(Q, env, n_tests)
