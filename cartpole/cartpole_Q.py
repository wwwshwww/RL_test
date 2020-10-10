import numpy as np
import matplotlib.pyplot as plt
import gym

from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display
from numpy.core.records import fromarrays

ENV = 'CartPole-v0' # env name
NUM_DIZITIZED = 6 # dizitize split num
GAMMA = 0.09 # time discount rate
ETA = 0.5 # 学習係数
MAX_STEPS = 200 # 1試行のstep数
NUM_EPISODES = 1000 # 最大試行回数

class Agent():
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)

    def update_Q_function(self, observation, action, reward, observation_next):
        self.brain.update_Q_table(observation, action, reward, observation_next)

    def get_action(self, observation, step):
        action = self.brain.decide_action(observation, step)
        return action

class Brain():
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        self.q_table = np.random.uniform(low=0, high=1, size=(NUM_DIZITIZED**num_states, num_actions))

    def bins(self, clip_min, clip_max, num):
        return np.linspace(clip_min, clip_max, num+1)[1:-1]

    def digitize(self, observation):
        cart_pos, cart_v, pole_angle, pole_v = observation

        # >>> np.digitize(300, bins=[10,22,310,400]) : 2
        # >>> np.digitize(399, bins=[10,22,310,400]) : 3
        # >>> np.digitize(400~, bins=[10,22,310,400]) : 4

        digitized = [
            np.digitize(cart_pos, bins=self.bins(-2.4, 2.4, NUM_DIZITIZED)),
            np.digitize(cart_v, bins=self.bins(-3.0, 3.0, NUM_DIZITIZED)),
            np.digitize(pole_angle, bins=self.bins(-0.5, 0.5, NUM_DIZITIZED)),
            np.digitize(pole_v, bins=self.bins(-2.4, 2.4, NUM_DIZITIZED))]

        return sum([x*(NUM_DIZITIZED**i) for i,x in enumerate(digitized)])

    def update_Q_table(self, observation, action, reward, observation_next):
        state = self.digitize(observation)
        state_next = self.digitize(observation_next)
        max_q_next = max(self.q_table[state_next][:])
        self.q_table[state, action] += ETA * (reward + GAMMA * max_q_next - self.q_table[state, action])

    def decide_action(self, observation, episode):
        ## ε-greedy 徐々に最適行動のみを採用するように
        state = self.digitize(observation)
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0,1):
            action = np.argmax(self.q_table[state][:])
        else:
            action = np.random.choice(self.num_actions)
        return action

class Environment():
    def __init__(self):
        self.env = gym.make(ENV)
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n

        self.agent = Agent(num_states, num_actions)

    def run(self):
        complete_episodes = 0
        is_episode_final = False
        frames = []

        for episode in range(NUM_EPISODES):
            observation = self.env.reset()
            for step in range(MAX_STEPS):
                if is_episode_final:
                    frames.append(self.env.render(mode='rgb_array'))

                action = self.agent.get_action(observation, episode)
                observation_next, _, done, _ = self.env.step(action)

                if done:
                    if step < 195:
                        reward = -1
                        complete_episodes = 0 # 連続記録が途絶える
                    else:
                        reward = 1
                        complete_episodes += 1 # 連続記録を更新
                else:
                    reward = 0

                self.agent.update_Q_function(observation, action, reward, observation_next)
                observation = observation_next

                if done:
                    print(f"{episode} Episode: Finished after {step+1}")
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
    frs[0].save('cartpole_Q.gif', save_all=True, append_images=frs[1:], optimize=False, duration=40, loop=0)

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