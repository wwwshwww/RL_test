from typing import Tuple
from environment import Action, Environment, State
import numpy as np

class Planner():

    def __init__(self, env: Environment) -> None:
        self.env = env
        self.log = []

    def initialize(self) -> None:
        self.env.reset()
        self.log = []

    def plan(self, gamma=0.9, threshold=0.0001):
        raise NotImplementedError

    def transition_at(self, state: State, action: Action) -> Tuple[float, State, float]:
        transition_probs = self.env.transit_func(state, action)
        for next_state in transition_probs:
            prob = transition_probs[next_state]
            reward, _ = self.env.reward_func(next_state)
            yield prob, next_state, reward

    def dict_to_grid(self, state_reward_dict):
        grid = np.zeros_like(self.env.grid)
        for s in state_reward_dict:
            grid[s.row, s.column] = state_reward_dict[s]

        return grid