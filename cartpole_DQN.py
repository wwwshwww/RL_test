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

class Brain():
    BATCH_SIZE = 32
    CAPACITY = 10000
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        self.memory = ReplayMemory(Brain.CAPACITY)

        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(num_states, 32))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('fc2', nn.Linear(32, 32))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(32, num_actions))

        print(self.model)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=10**(-4))

    def replay(self):
        if len(self.memory) < Brain.BATCH_SIZE:
            return

        ############# p.132~

        transitions = self.memory.sample(Brain.BATCH_SIZE)

        # (state, action, state_next, reward) * BATCH_SIZE
        # to
        # (state * BS, action * BS, state_next * BS, reward * BS)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        self.model.eval()

        state_action_values = self.model(state_batch).gather(1, action_batch)

        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        next_state_values = torch.zeros(Brain.BATCH_SIZE)

        # ニューラルネットの出力からmax(1)で列方向の最大値の[value, index]を求め，マスクに沿ってdetachした各値を設定
        #  r = [0,0,0]
        #  mask = [True,False,True]
        #  r[mask] = [1,2]
        #  r -> [1,0,2] 
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()

        # 教師となるQ(s_t, a_t)値を、Q学習の式から求める
        expected_state_action_values = reward_batch + GAMMA * next_state_values

        self.model.train()

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decide_action(self, state, episode):
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0,1):
            self.model.eval()
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1,1) # view(1,1) : [torch.LongTensor of size 1] -> size 1x1
        else:
            action = torch.LongTensor([[random.randrange(self.num_actions)]]) # 0 or 1

        return action

class Agent():
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)

    def update_Q_function(self):
        self.brain.replay()

    def get_action(self, state, episode):
        return self.brain.decide_action(state, episode)

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)

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
                self.agent.update_Q_function()
                state = state_next

                if done:
                    print(f"{episode} Episode: Finished after {step+1} steps : 10 average = {episode_10_list.mean()}")
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
    frs[0].save('cartpole_DQN.gif', save_all=True, append_images=frs[1:], optimize=False, duration=40, loop=0)

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