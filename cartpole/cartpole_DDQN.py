import random
import numpy as np
import matplotlib.pyplot as plt
import gym

from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display

from collections import namedtuple

import torch
from torch import nn, optim
import torch.nn.functional as F

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

ENV = 'CartPole-v0' # env name
GAMMA = 0.99 # time discount rate
MAX_STEPS = 200 # 1試行のstep数
NUM_EPISODES = 500 # 最大試行回数

class ReplayMemory():

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0 # save index

    def push(self, state, action, state_next, reward):
        '''transition = (state, action, next_state, reward)'''

        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.index] = Transition(state, action, state_next, reward)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Net(nn.Module):

    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_out)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        output = self.fc3(h2)
        return output

class Brain():
    
    BATCH_SIZE = 32
    CAPACITY = 10000

    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        self.memory = ReplayMemory(Brain.CAPACITY)

        n_in, n_mid, n_out = num_states, 32, num_actions
        self.main_q_network = Net(n_in, n_mid, n_out)
        self.target_q_network = Net(n_in, n_mid, n_out)

        print(self.main_q_network)
        
        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=10**(-4))

    def replay(self):
        if len(self.memory) < Brain.BATCH_SIZE:
            return

        ## create minibatch
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch()

        ## get Q(s_t, a_t) of teacher signal
        self.expected_state_action_values = self.get_expected_state_action_values()

        ## train network
        self.update_main_q_network()

    def decide_action(self, state, episode):
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0,1):
            self.main_q_network.eval()
            with torch.no_grad():
                action = self.main_q_network(state).max(1)[1].view(1,1) # view(1,1) : [torch.LongTensor of size 1] -> size 1x1
        else:
            action = torch.LongTensor([[random.randrange(self.num_actions)]]) # 0 or 1

        return action

    def make_minibatch(self):
        transitions = self.memory.sample(Brain.BATCH_SIZE)

        # - transision: (state, action, state_next, reward)
        # transision * BATCH_SIZE
        #  to
        # (state * BS, action * BS, state_next * BS, reward * BS)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self):
        self.main_q_network.eval()
        self.target_q_network.eval()

        # in order to get what done action's Q value, use `gather` for get index of left or right
        self.state_action_values = self.main_q_network(self.state_batch).gather(1, self.action_batch)

        # get max{Q(s_t+1, a)}

        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, self.batch.next_state)))
        next_state_values = torch.zeros(Brain.BATCH_SIZE)

        a_m = torch.zeros(Brain.BATCH_SIZE).type(torch.LongTensor)

        ## ニューラルネットの出力からmax(1)で列方向の最大値の[value, index]を求め，マスクに沿ってdetachした各値を設定
        ##  r = [0,0,0]
        ##  mask = [True,False,True]
        ##  r[mask] = [1,2]
        ##  -> r: [1,0,2] 
        a_m[non_final_mask] = self.main_q_network(self.non_final_next_states).detach().max(1)[1]

        # size 32 -> 32x1
        a_m_non_final_next_states = a_m[non_final_mask].view(-1,1)

        # get a_m's Q value by target Q-network
        ## detachで取り出し，sqeezeでsize[minibatch x 1]を[minibatch]に
        next_state_values[non_final_mask] = self.target_q_network(self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        expected_state_action_values = self.reward_batch + GAMMA * next_state_values

        return expected_state_action_values

    def update_main_q_network(self):
        self.main_q_network.train()

        ## expected's size [minibatch] -> [minibatch x 1]
        loss = F.smooth_l1_loss(self.state_action_values, self.expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad() # 勾配をリセット
        loss.backward() # バックプロパゲーション
        self.optimizer.step() # 結合パラメータを更新

    def update_target_q_network(self):
        # taget be same main 
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())

class Agent():

    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)

    def update_Q_function(self):
        self.brain.replay()

    def get_action(self, state, episode):
        return self.brain.decide_action(state, episode)

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_function(self):
        self.brain.update_target_q_network()

class Environment():

    def __init__(self):
        self.env = gym.make(ENV)
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        self.agent = Agent(self.num_states, self.num_actions)

    def run(self):
        episode_10_list = np.zeros(10)

        complete_episodes = 0
        is_episode_final = False
        frames = []

        # convert observation to state what torch style; FloatTensor of size 4 -> 1x4
        #  o: observation
        obs_to_state = lambda o: torch.unsqueeze(torch.from_numpy(o).type(torch.FloatTensor), 0)

        for episode in range(NUM_EPISODES):

            observation = self.env.reset()
            state = obs_to_state(observation)

            for step in range(MAX_STEPS):
                if is_episode_final:
                    frames.append(self.env.render(mode='rgb_array'))

                action = self.agent.get_action(state, episode)
                observation_next, _, done, _ = self.env.step(action.item())

                if done:
                    state_next = None

                    episode_10_list = np.hstack((episode_10_list[1:], step+1))

                    if step < 195:
                        reward = torch.FloatTensor([-1.0])
                        complete_episodes = 0 # 連続記録が途絶える
                    else:
                        reward = torch.FloatTensor([1.0])
                        complete_episodes += 1 # 連続記録を更新
                else:
                    reward = torch.FloatTensor([0.0])
                    state_next = obs_to_state(observation_next)

                self.agent.memorize(state, action, state_next, reward)
                # Experience replay
                self.agent.update_Q_function()
                state = state_next

                if done:
                    print(f"{episode} Episode: Finished after {step+1} steps : 10 average = {episode_10_list.mean()}")

                    # 2試行に1度，Target Q-NetworkをMainと同期
                    if episode % 2 == 0:
                        self.agent.update_target_q_function()
                    break

            if is_episode_final:
                display_frames_as_gif(frames)
                break

            if complete_episodes >= 10:
                print("10回連続成功")
                is_episode_final = True

def display_frames_as_gif(frames):
    from PIL import Image, ImageDraw
    
    frs = [Image.fromarray(f, mode='RGB') for f in frames]
    frs[0].save('cartpole_DDQN.gif', save_all=True, append_images=frs[1:], optimize=False, duration=40, loop=0)

def main():
#     # CartPoleをランダムに動かす
#     frames = []
#     env = gym.make('CartPole-v0')
#     env.reset()
#     for step in range(0, 200):
#         frames.append(env.render(mode='rgb_array'))  # framesに各時刻の画像を追加していく
#         action = np.random.choice(2)  # 0(カートを左に押す),1(カートを右に押す)をランダムに返す
#         observation, reward, done, info = env.step(action)  # actionを実行する

#     display_frames_as_gif(frames)

    cartpole_env = Environment()
    cartpole_env.run()


if __name__ == "__main__":
    main()