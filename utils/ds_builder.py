import os

from jaxrl_m.evaluation import EpisodeMonitor
import jax.numpy as jnp
import jax

import numpy as np
from tqdm.auto import tqdm
from typing import *
from src.dataset import Dataset
import d4rl
import gym
    
    
def get_dataset(env: gym.Env, expert: bool = False,
                num_episodes: int = 1,
                clip_to_eps: bool = True,
                eps: float = 1e-5,
                normalize_agent_states: bool = False,
                fix_antmaze_timeout=True):
        if 'antmaze' in env.spec.id and fix_antmaze_timeout:
            dataset = qlearning_dataset_with_timeouts(env)
        else:
            dataset = d4rl.qlearning_dataset(env)
        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)
            
        dones_float = np.zeros_like(dataset['rewards'])
        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0
        dones_float[-1] = 1
        if 'realterminals' in dataset:
            masks = 1.0 - dataset['realterminals'].astype(np.float32)
        else:
            masks = 1.0 - dataset['terminals'].astype(np.float32)
        dataset['masks'] = masks
        dataset['dones'] = dones_float
        if expert:
            trajectories = split_into_trajectories(
                        observations=dataset['observations'].astype(jnp.float32),
                        actions=dataset['actions'].astype(jnp.float32),
                        rewards=dataset['rewards'].astype(jnp.float32),
                        masks=masks,
                        dones_float=dones_float.astype(jnp.float32),
                        next_observations=dataset['next_observations'].astype(jnp.float32))
            if 'antmaze' in env.spec.id:
                returns = [
                    sum([t[2]
                        for t in traj]) /
                    (1e-4 + np.linalg.norm(traj[0][0][:2]))
                    for traj in trajectories
                ]
            else:
                returns = [sum([t[2] for t in traj]) for traj in trajectories]
            idx = np.argpartition(returns, -num_episodes)[-num_episodes:]
            demo_returns = [returns[i] for i in idx]
            print(f"Expert returns {demo_returns}, mean {np.mean(demo_returns)}")
            expert_demo = [trajectories[i] for i in idx]
            expert_demos = merge_trajectories(expert_demo)
            dataset = {"observations": expert_demos[0],
                       'next_observations': expert_demos[-1],
                       'actions': expert_demos[1],
                       'rewards': expert_demos[2],
                       'masks': expert_demos[3],
                       'dones': 1-expert_demos[3]}
        else:
            if "antmaze" in env.spec.id:
                dataset['rewards'] -= 1.0
            if normalize_agent_states:
                dataset['observations'], dataset['next_observations'], mean, std = normalize_states(dataset['observations'], dataset['next_observations'])
                print(f"MEAN: {mean}, STD: {std}")
            else:
                mean, std = 0, 1
            return Dataset(dataset_dict=dataset), mean, std
        return Dataset(dataset_dict=dataset)

def normalize_states(data,next_state, eps=1e-3):
    mean = data.mean(0, keepdims=True)
    std = data.std(0, keepdims=True) + eps
    data = (data - mean) / std
    next_state = (next_state - mean) / std
    return data, next_state, mean, std

def merge_trajectories(trajs):
  flat = []
  for traj in trajs:
    for transition in traj:
      flat.append(transition)
  return jax.tree_util.tree_map(lambda *xs: np.stack(xs), *flat)

def setup_datasets(expert_env_name: str, agent_env_name: str, expert_num: int, normalize_agent_states): #taking all trajs from agent ds
    expert_env = gym.make(expert_env_name)
    agent_env = gym.make(agent_env_name)
    
    if 'antmaze' in agent_env_name:
        agent_env = EpisodeMonitor(agent_env)
        
        os.environ['CUDA_VISIBLE_DEVICES']="4" # for headless server
        agent_env.render(mode='rgb_array', width=200, height=200)
        os.environ['CUDA_VISIBLE_DEVICES']="0,1,2,3,4"
        
        if 'large' in agent_env.spec.id:
            agent_env.viewer.cam.lookat[0] = 18
            agent_env.viewer.cam.lookat[1] = 12
            agent_env.viewer.cam.distance = 50
            agent_env.viewer.cam.elevation = -90

        elif 'ultra' in agent_env.spec.id:
            agent_env.viewer.cam.lookat[0] = 26
            agent_env.viewer.cam.lookat[1] = 18
            agent_env.viewer.cam.distance = 70
            agent_env.viewer.cam.elevation = -90
        else:
            agent_env.viewer.cam.lookat[0] = 18
            agent_env.viewer.cam.lookat[1] = 12
            agent_env.viewer.cam.distance = 50
            agent_env.viewer.cam.elevation = -90
        
    dataset_expert = get_dataset(expert_env, expert=True, num_episodes=expert_num)
    dataset_agent, mean, std = get_dataset(agent_env, expert=False, normalize_agent_states=normalize_agent_states)
        
    return agent_env, dataset_expert, dataset_agent, mean, std

def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
  trajs = [[]]

  for i in tqdm(range(len(observations))):
    trajs[-1].append(
        [
            observations[i],
            actions[i],
            rewards[i],
            masks[i],
            next_observations[i]])
    if dones_float[i] == 1.0 and i + 1 < len(observations):
      trajs.append([])

  return trajs

def load_trajectories(name: str, rewards, fix_antmaze_timeout=True):
    env = gym.make(name)
    if "antmaze" in name and fix_antmaze_timeout:
        dataset = qlearning_dataset_with_timeouts(env)
    else:
        dataset = d4rl.qlearning_dataset(env)
        dones_float = np.zeros_like(rewards)
    dones_float = np.zeros_like(rewards)
    for i in range(len(dones_float) - 1):
        if np.linalg.norm(dataset['observations'][i + 1] -
                          dataset['next_observations'][i]
                         ) > 1e-6 or dataset['terminals'][i] == 1.0:
          dones_float[i] = 1
        else:
          dones_float[i] = 0
    dones_float[-1] = 1
    
    if 'realterminals' in dataset:
        masks = 1.0 - dataset['realterminals'].astype(np.float32)
    else:
        masks = 1.0 - dataset['terminals'].astype(np.float32)
    traj = split_into_trajectories(
          observations=dataset['observations'].astype(np.float32),
          actions=dataset['actions'].astype(np.float32),
          rewards=rewards.astype(np.float32), #dataset['rewards'].astype(np.float32),
          masks=masks,
          dones_float=dones_float.astype(np.float32),
          next_observations=dataset['next_observations'].astype(np.float32))
    return traj

def qlearning_dataset_with_timeouts(env,
                                    dataset=None,
                                    terminate_on_end=False,
                                    disable_goal=True,
                                    **kwargs):
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    realdone_ = []
    if "infos/goal" in dataset:
        if not disable_goal:
            dataset["observations"] = np.concatenate(
                [dataset["observations"], dataset['infos/goal']], axis=1)
        else:
            pass

    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i]
        new_obs = dataset['observations'][i + 1]
        action = dataset['actions'][i]
        reward = dataset['rewards'][i]
        done_bool = bool(dataset['terminals'][i])
        realdone_bool = bool(dataset['terminals'][i])
        if "infos/goal" in dataset:
            final_timestep = True if (dataset['infos/goal'][i] !=
                                dataset['infos/goal'][i + 1]).any() else False
        else:
            final_timestep = dataset['timeouts'][i]

        if i < N - 1:
            done_bool += final_timestep

        if (not terminate_on_end) and final_timestep:
        # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        realdone_.append(realdone_bool)
        episode_step += 1

    return {
      'observations': np.array(obs_),
      'actions': np.array(action_),
      'next_observations': np.array(next_obs_),
      'rewards': np.array(reward_)[:],
      'terminals': np.array(done_)[:],
      'realterminals': np.array(realdone_)[:],
  }