import os
import gym
import numpy as np
import time
from waste_env import WasteEnv
import pickle
from utils import get_mapping, get_all_possible_states, item_conf
from utils import seed_everything
import argparse
seed_everything(42) #We love deterministic algorithms


"""
Qlearning is an off policy learning python implementation.
This is a python implementation of the qlearning algorithm in the Sutton and
Barto's book on RL. It's called SARSA because - (state, action, reward, state,
action). The only difference between SARSA and Qlearning is that SARSA takes the
next action based on the current policy while qlearning takes the action with
maximum utility of next state.
"""

MAPPING = get_mapping()

def init_q(states, na, type="ones"):
    #states is a list of all states
    #na is number of discrete actions
    Q = {}
    for state in states:
        Q[state] = np.ones((na))
        if type == "ones":
            Q[state] = np.ones((na))/10
        elif type == "random":
            Q[state] = np.random.randn(na)
        elif type == "zeros":
            Q[state] = np.zeros((na))
    return Q


def epsilon_greedy(Q, epsilon, n_actions, s, train=False):
    if train or np.random.rand() >= epsilon:
        action = np.argmax(Q[s])
    else:
        action = np.random.randint(0, n_actions)
    return action


def save_Q(Q, episode, name="item_1", logdir="logs/"):
    path = os.path.join(logdir, name)
    if not os.path.exists(path):
        os.mkdir(path)
    model_path = os.path.join(path, f"{episode}.pkl")
    with open(model_path, 'wb') as handle:
        pickle.dump(Q, handle)
    return True


def qlearning(alpha, gamma, epsilon, episodes,
                max_steps, n_tests, save_freq,
                epislon_end_episode = 200, epsilon_min=0.1,
                render = False, test=False, item_name='item1'):
    env = WasteEnv(*item_conf(item_name))
    states, ns = get_all_possible_states()
    n_states, n_actions = ns, env.action_space.n
    Q = init_q(states, n_actions, type="zeros")
    timestep_reward = []

    epsilon_start = epsilon
    decay = (epsilon_start-epsilon_min)/epislon_end_episode

    for episode in range(episodes):
        print(f"Episode: {episode}")
        if episode%save_freq == 0:
            save_Q(Q, episode, name=item_name)
        if episode < epislon_end_episode:
            epsilon -= decay
        s = env.reset()
        a = epsilon_greedy(Q, epsilon, n_actions, s)
        if render:
            print(f"State: {s}")
            print(f"=====Action: {a}/{MAPPING[a]}")
        t = 0
        total_reward = 0
        done = False
        while t < max_steps:
            t += 1
            s_, reward, done, info = env.step(a)
            total_reward += reward
            a_ = np.argmax(Q[s])

            if render:
                env.render()
                print(f"State: {s_}")
                print(f"Reward: {reward}")
                print(f"=====Action: {a_}/{MAPPING[a_]}")

            if done:
                G = ( reward  - Q[s][a] )
                Q[s][a] += alpha * G
                timestep_reward.append(total_reward)
                if render:
                    print(f">>>>>>> This episode took {t} timesteps and reward: {total_reward}")
                    print(f"G: {G} | alpha*G: {alpha*G} | Q[s][a]: {Q[s][a]}")
            else:
                G = ( reward + (gamma * Q[s_][a_]) - Q[s][a] )
                Q[s][a] += alpha * G
                if render:
                    print(f"G: {G} | alpha*G: {alpha*G} | Q[s][a]: {Q[s][a]}")

            s, a = s_, a_
    return Q




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", "-i",
                        help="item name. See config.ini for details.")
    args = parser.parse_args()
    item_name = args.i

    alpha = 0.9
    gamma = 0.99
    epsilon = 0.3
    episodes = 10000
    epislon_end_episode = episodes // 30
    max_steps = 2500
    n_tests = 1
    save_freq = 1000

    Q_table = qlearning(alpha, gamma, epsilon, episodes,
                                max_steps, n_tests, save_freq,
                                epislon_end_episode, test = False,
                                item_name=item_name, render=False)
