from model import Net
from torch.optim import Adam

LR = 1e-3
MEMORY_CAPACITY = 6400
N_STATES = 15
N_ACTIONS = 5
EPSILON = 0.9
BATCH_SIZE = 64
GAMMA = 0.99

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.PATH = 'logs/'

    def get_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x.flatten()), 0)
        print(x.shape)
        if np.random.uniform() < EPSILON:   # greedy
            action = self.eval_net(x)
        else:   # random
            action = (np.random.rand(0, N_ACTIONS) - 0.5)*2
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        # q_eval = self.eval_net(b_s).gather(1, b_a - 1)  # shape (batch, 1) (-1 because action + 1 was done previously)
        q_eval = self.eval_net(b_s).detach()  # shape (batch, 1) (-1 because action + 1 was done previously)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, eval_name = 'eval_net', train_name = 'train_net'):
        torch.save(self.eval_net.state_dict(), self.PATH  + str(N_STATES) + eval_name)
        torch.save(self.target_net.state_dict(), self.PATH  + str(N_STATES)+ train_name)

    def load_model(self, eval_name = 'eval_net.m', train_name = 'train_net.m'):
        self.eval_net.load_state_dict(torch.load(self.PATH +  str(N_STATES) + eval_name))
        self.target_net.load_state_dict(torch.load(self.PATH  + str(N_STATES) + train_name))


if __name__ == "__main__":
    start = time.time()
    dqn = DQN()
    dqn.load_model()
    episodes = 60000

    print("Start")
    print('Collecting Experience...')
    for i in range(episodes):
        s = env.reset()
        total_reward = 0
        while True:
            a = dqn.get_action(s)
            s_, r, done, info = env.step(a)
            # modify the reward
            dqn.store_transition(s.flatten(), a, r, s_.flatten())
            total_reward += r
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
            if(i % 500 == 0):
                print(f"Episode: {i} | NetReward: {total_reward}")
            if done:
                break
            s = s_

    end = time.time()
