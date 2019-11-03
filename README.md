# Preventing food wastage with RL
Multi-Agent RL for dynamic pricing of food to save wastage while maximizing profit

# Disclaimer
I haven't tested the economic sanity of my model here. This is a hobby project and I take no responsibility for the catastrophe you might face on deploying this for actual use. However, I am trying to fix as many things as I can.

# What
This is an experimental repository where I've written a multi-agent RL algorithm to prevent food wastage while maximizing profit. I wrote this during a hackathon (which I lost) and have made some minor fixes later (still in development stage). Hopefully, I will continue working on making this better with time. (for now, the task is so simple that even humans can do it)

Some specifications:
- Multiple agents for multiple items (one for each).
- Q-learning algorithm.
- Making decisions by taking into account the economics of the market and food shelf life.
- what decisions? they are changes made in price (percentage of increase/decrease from the base price).

`waste_env.py` is a custom environment made by extending openai gym sepcially written for this task.

# How
Steps to follow:
- Define the product specification in the config file (`config/config.ini`).
- run `python train.py -i <item name>` for training a qlearning policy for an agent on `item name`.
- run `python test.py -i <item name>` for testing the policy.
- QTables are stored in the log folder.

### Config file format:
```
[<item name>]
logs_dir = logs/item_1
START_STATE_N = 90
START_STATE_L = 40
C = 20
K = 2
P = 50
```
- `START_STATE_N` is the number of item in inventory at the beginning.
- `START_STATE_L` is the life of item in timsteps (can be days, weeks or hours) at the beginning.
- `C` This constant determines how popular an item is.
- `K` This constant determines how robust each item's demand is to the price (Basically the ratio of demand and price).
- `P` Base price of the item.

### logs
- logs folder contains the Q-table agent uses for each item
- `logs/<item name>/<episode number>.pkl` is the serialized Q-table for `<item name>` at `<episode number>`
- save frequency is an argument that can be changed... somewhere (`train.py`)

# More
- Algorithm used: Qlearning (can be anything, SARSA, TD-lambda etc etc). Because of the implementation of the problem as a multi agent RL, the task is fairly simple here.
- Reward function: `lambda1*profit - lambda2*waste` where `lambda1` and `lambda2` are two constants such that `lambda1+lambda2=1` (Basically linear interpolation). Maximizing this reward would mean maximizing profit while minimizing waste.
- `lambda1 > lambda2`: focus more on maximizing profit.
- `lambda1 < lambda2`: focus more on minimizing the waste.
- `lambda1 = lambda2`: balanced.
- Look at `waste_env.py` for more implementation details of the environment.


# TODO
- Instead of Multi-Agent RL, write a single RL algorithm to handle all items at once
- ~~Write a global environment~~ DONE
- Write DQN algorithm for the global environment: in progress
- Write a proper document/blog/paper for explanation, motivation and reproducibility of this work
- Read arguments from a config file. (this is a todo for a lot of my projects, I just never do it. a bad habit.)


### Don't waste food. Some people really love it.
![](https://i.imgur.com/s9DjU2T.gif)

### Credits
My team for the hackathon during which I made the first version of this project. Thanks to them:
- [Utsav Prabhakar](https://github.com/utsavprabhakar)
- [Vaibhav kumar](https://github.com/vaibhavk97)
- [Vikas Tomar](https://github.com/rootone-lab)
