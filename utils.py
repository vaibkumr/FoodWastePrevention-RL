import configparser
import random
import os
import numpy as np

def seed_everything(seed=42):
    """42 is the answer to everything. Also, we love deterministic algorithms,
    they are great for reproduction of results and experimentation results."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def get_mapping():
    MAPPING = { 0:-1.0,
                1:-0.9,
                2:-0.8,
                3:-0.7,
                4:-0.6,
                5:-0.5,
                6:-0.4,
                7:-0.3,
                8:-0.2,
                9:-0.1,
                10:0.0,
                11:0.1,
                12:0.2,
                13:0.3,
                14:0.4,
                }
    return MAPPING


def get_all_possible_states(N=90, L=40):
    states = []
    ns = 0
    for l in range(L+1):
        for n in range(N+1):
            states.append((int(n), int(l)))
            ns += 1
    return states, ns

def item_conf(key='ITEM1'):
    config = configparser.ConfigParser()
    config.read('config/config.ini')
    conf = config[key]
    START_STATE = [conf.getint('START_STATE_N'), conf.getint('START_STATE_L')]
    C = conf.getint('C')
    K = conf.getint('K')
    P = conf.getint('P')
    return [START_STATE, C, K, P]
