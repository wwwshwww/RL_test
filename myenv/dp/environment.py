from typing import Dict, List, Tuple
import numpy as np
from enum import Enum 

class State():

    def __init__(self, row=-1, column=-1):
        self.row = row
        self.column = column

    def __repr__(self) -> str:
        return f"<State: [{self.row}, {self.column}]>"

    def clone(self):
        return State(self.row, self.column)

    def __hash__(self) -> int:
        return hash((self.row, self.column))

    def __eq__(self, o: object) -> bool:
        return self.row == o.row and self.column == o.column

class Action(Enum):
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2

class Environment():

    def __init__(self, grid: List[List[int]], move_prob=0.8):
        #  0: ordinary cell
        # -1: damage cell(game end)
        #  1: reward cell(game end)
        #  9: block cell(can't locate agent)

        self.grid = grid
        self.agent_state = State()
        self.default_reward = -0.04
        self.move_prob = move_prob
        self.reset()

    @property
    def row_length(self) -> int:
        return len(self.grid)

    @property
    def column_length(self) -> int:
        return len(self.grid[0])

    @property
    def actions(self) -> List[Action]:
        return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

    @property
    def states(self) -> List[State]:
        states = []
        for row in range(self.row_length):
            for column in range(self.column_length):
                if self.grid[row][column] != 9:
                    states.append(State(row, column))

        return states

    def transit_func(self, state: State, action: Action) -> Dict[Action, float]:
        transition_probs = {}
        if not self.can_action_at(state):
            return transition_probs

        opposite_direction = Action(action.value * -1)
        for a in self.actions:
            if a == action:
                prob = self.move_prob
            elif a == opposite_direction:
                prob = 0
            else:
                prob = (1-self.move_prob)/2
            
            next_state = self._move(state, a)
            if next_state not in transition_probs:
                transition_probs[next_state] = prob
            else:
                transition_probs[next_state] += prob

        return transition_probs

    def can_action_at(self, state: State) -> bool:
        return self.grid[state.row][state.column] == 0

    def _move(self, state: State, action: Action) -> State:
        if not self.can_action_at(state):
            raise Exception("Can't move from here!")

        next_state: State = state.clone()

        if action == Action.UP:
            next_state.row -= 1
        elif action == Action.DOWN:
            next_state.row += 1
        elif action == Action.LEFT:
            next_state.column -= 1
        elif action == Action.RIGHT:
            next_state.column += 1

        if not (0 <= next_state.row < self.row_length):
            next_state = state
        if not (0 <= next_state.column < self.column_length):
            next_state = state
        if self.grid[next_state.row][next_state.column] == 9:
            next_state = state

        return next_state

    def reward_func(self, state: State) -> Tuple[float, bool]:
        reward = self.default_reward
        done = False

        attribute = self.grid[state.row][state.column]
        if attribute == 1:
            reward = 1
            done = True
        elif attribute == -1:
            reward = -1
            done = True
        
        return reward, done

    def reset(self) -> State:
        # Locate the agent at lower left corner
        self.agent_state = State(self.row_length-1, 0)
        return self.agent_state

    def step(self, action: Action) -> Tuple[State, float, bool]:
        next_state, reward, done = self.transit(self.agent_state, action)
        if next_state is not None:
            self.agent_state = next_state
        
        return next_state, reward, done

    def transit(self, state: State, action: Action) -> Tuple[State, float, bool]:
        transition_probs = self.transit_func(state, action)
        if len(transition_probs) == 0:
            return None, None, True

        next_states = [s for s in transition_probs]
        probs = [transition_probs[s] for s in transition_probs]

        next_state = np.random.choice(next_states, p=probs)
        reward, done = self.reward_func(next_state)
        return next_state, reward, done


