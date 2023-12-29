import d4rl
import gym
import numpy as np

from typing import *
#from jaxrl_m.dataset import Dataset
from src.dataset import Dataset


def get_dataset(env: gym.Env,
                path_to_data: str = None,
                expert: bool = False,
                expert_num: int = 5,
                clip_to_eps: bool = True,
                eps: float = 1e-5,
                ):
        if path_to_data is not None:
            # add possibility to load from npz
            pass
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
            dones_float[-1] = 1
        else:
            dones_float = dataset['terminals']
        dataset['masks'] = 1.0 - dataset['terminals']
        dataset['dones'] = dones_float
        
        if expert:
            from d3rlpy.datasets import MDPDataset
            
            mdp_dataset = MDPDataset(observations=dataset['observations'],
                                     actions=dataset['actions'],
                                     rewards=dataset['rewards'],
                                     terminals=dataset['dones'])
            if 'antmaze' in env.spec.id:
                expert_episodes = []
                ep_rets = []
                max_len = []
                for episode_i in mdp_dataset.episodes:
                    ep_rets.append(episode_i.rewards.sum())
                    max_len.append(len(episode_i))
                ranked_idx = np.argsort(ep_rets)[::-1]
                j = 0
                i = 0
                while i < expert_num:
                    if len(mdp_dataset.episodes[ranked_idx[j]]) > 200 and len(mdp_dataset.episodes[ranked_idx[j]]) < 400:
                        expert_episodes.append(mdp_dataset.episodes[ranked_idx[j]])
                        i += 1
                    j += 1
            else:
                expert_episodes = mdp_dataset.episodes[:expert_num]

            dataset = convert_mdp_to_dict(env=env, episodes=expert_episodes)
        return Dataset(dataset_dict=dataset)
        # return Dataset.create(
        #     observations=dataset['observations'].astype(np.float32),
        #     next_observations=dataset['next_observations'].astype(np.float32),
        #     actions=dataset['actions'].astype(np.float32),
        #     rewards=dataset['rewards'].astype(np.float32),
        #     masks=dataset['masks'],
        #     dones_float=dataset['dones'],
        # ), dataset

def convert_mdp_to_dict(episodes, env):
    observations = []
    actions = []
    rewards = []
    terminals = []
    episode_terminals = []
    dataset = {}
    next_observations = []
    ep_rets = []
    for episode in episodes:
      ep_rets.append(episode.rewards.sum())

      for idx, transition in enumerate(episode):

        observations.append(transition.observation)
        if isinstance(env.action_space, gym.spaces.Box):
          actions.append(np.reshape(transition.action, env.action_space.shape))
        else:
          actions.append(transition.action)
        rewards.append(transition.reward)
        terminals.append(transition.terminal)
        episode_terminals.append(idx == len(episode) - 1)
        next_observations.append(transition.next_observation)
        
    dataset['rewards'] = np.stack(rewards)
    dataset['observations'] = np.stack(observations)
    dataset['dones'] = np.stack(terminals)
    dataset['masks'] = 1 - dataset['dones']
    dataset['actions'] = np.stack(actions)
    dataset['next_observations'] = np.stack(next_observations)
    print(f"ep_reward_sum   Max/Mean/Median/Min: {np.max(ep_rets)}/{np.mean(ep_rets)}/{np.median(ep_rets)}/{np.min(ep_rets)}")
    print(f"state-action reward     Max/Mean/Median/Min:   {np.max(rewards)}/{np.mean(rewards)}/{np.median(rewards)}/{np.min(rewards)} ")
    return dataset
