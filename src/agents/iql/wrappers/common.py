from typing import Tuple
import gym
import numpy as np

TimeStep = Tuple[np.ndarray, float, bool, dict]

class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env, num_repeats, discount):
        super().__init__(env)
        self._num_repeats = num_repeats
        self.discount = discount

    def step(self, action):
        reward = 0.0
        discount = 1.0
        observation, done, info = None, None, None
        for i in range(self._num_repeats):
            observation, r, done, info = self.env.step(action)
            reward += (r or 0.0) * discount
            discount *= self.discount
            if done:
                break

        return observation, reward, done, info


