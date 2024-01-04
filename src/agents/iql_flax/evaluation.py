from typing import Dict

import flax.linen as nn
import gym
import numpy as np


def evaluate(agent: nn.Module, env: gym.Env,
             num_episodes: int) -> Dict[str, float]:
    stats = {'return': [], 'length': []}
    trajs = []

    for _ in range(num_episodes):
        observation, done = env.reset(), False
        tr = [observation]

        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, _, done, info = env.step(action)
            tr.append(observation)

        for k in stats.keys():
            stats[k].append(info['episode'][k])

        trajs.append(np.stack(tr))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs
