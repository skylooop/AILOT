import d4rl
import gym
import numpy as np

from typing import *
from jaxrl_m.dataset import Dataset

def get_dataset(env: gym.Env,
                expert: bool = False,
                num_episodes: int = 5,
                clip_to_eps: bool = True,
                eps: float = 1e-5,
                ):
        if 'antmaze' in env.spec.id:
            dataset = d4rl.qlearning_dataset(env)
        else:
            dataset = env.get_dataset()
        
        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)
            
        if 'antmaze' in env.spec.id:
            dones_float = np.zeros_like(dataset['rewards'])
            for i in range(len(dones_float) - 1):
                if np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6 or dataset['terminals'][i] == 1.0:
                    dones_float[i] = 1
                else:
                    dones_float[i] = 0
            dones_float[-1] = 1
        else:
            dones_float = np.logical_or(dataset['timeouts'], dataset['terminals'])
        
        cutoff = dones_float[0]
        if expert:
            dataset['observations'] = dataset['observations'][:num_episodes * (cutoff+1)]
            dataset['next_observations'] = dataset['next_observations'][:num_episodes * (cutoff+1)]
            dataset['actions'] = dataset['actions'][:num_episodes * (cutoff+1)]
            dataset['rewards'] = dataset['rewards'][:num_episodes * (cutoff+1)]
    
        return Dataset.create(
            observations=dataset['observations'].astype(np.float32),
            next_observations=dataset['next_observations'].astype(np.float32),
            actions=dataset['actions'].astype(np.float32),
            rewards=dataset['rewards'].astype(np.float32),
            masks=1.0 - dones_float.astype(np.float32),
            dones_float=dones_float.astype(np.float32),
        )

def modify_reward(
    dataset: Dict[str, np.ndarray], env_name: str, max_episode_steps: int = 1000
):
    if any(s in env_name.lower() for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = get_normalization(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0
    return dataset

def get_normalization(dataset):
    returns = []
    ret = 0
    for r, term in zip(dataset['rewards'], dataset['dones_float']):
        ret += r
        if term:
            returns.append(ret)
            ret = 0
    return (max(returns) - min(returns)) / 1000
