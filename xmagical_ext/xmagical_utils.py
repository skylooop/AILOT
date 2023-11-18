import gym
import xmagical
xmagical.register_envs()

from PIL import Image
import numpy as np
from jaxrl_m.evaluation import EpisodeMonitor
from jaxrl_m.dataset import Dataset
import jax

def make_env(modality, visual='Pixels', randomize=False):
    modality = modality.capitalize()
    assert modality in ['Gripper', 'Shortstick', 'Mediumstick', 'Longstick']
    if randomize:
        env = gym.make(f'SweepToTop-{modality}-{visual}-Allo-TestLayout-v0')
    else:
        env = gym.make(f'SweepToTop-{modality}-{visual}-Allo-Demo-v0')
    if visual == 'Pixels':
        transform_obs = lambda obs: np.asarray(Image.fromarray(obs).resize((64, 64)))
        env = gym.wrappers.TransformObservation(env, transform_obs)
    env = EpisodeMonitor(env)
    return env

def get_dataset(modality, dir_name='d4rl_ext/xmagical/xmagical_replay', keys=None):
    """ If keys is None, return all keys, else return specified keys"""
    modality = modality.lower()
    assert modality in ['gripper', 'shortstick', 'mediumstick', 'longstick']
    fname = f'{dir_name}/{modality}_train.npz'
    buffer = dict(np.load(fname, mmap_mode="r"))
    if keys is None: keys = buffer.keys()
    buffer['observations'] = buffer['states'][:, :17]
    buffer['next_observations'] = buffer['next_states'][:, :17]
    return Dataset({k: buffer[k] for k in keys})

def get_all_datasets(dir_name='d4rl_ext/xmagical/xmagical_replay'):
    return {modality: get_dataset(modality, dir_name) for modality in ['gripper', 'shortstick', 'mediumstick', 'longstick']}

def crossembodiment_dataset(not_modality, dir_name='d4rl_ext/xmagical/xmagical_replay'):
    datasets = []
    keys = ['observations', 'next_observations', 'rewards', 'masks', 'dones_float']
    for i, modality in enumerate(['gripper', 'shortstick', 'mediumstick', 'longstick']):
        if modality == not_modality: continue
        dataset = get_dataset(modality, dir_name, keys=keys)
        dataset = dataset.copy({'embodiment': np.full(dataset.size, i)})
        datasets.append(dataset._dict)
    full_dataset = Dataset(jax.tree_map(lambda *arrs: np.concatenate(arrs, axis=0), *datasets))
    return full_dataset


def evaluate_with_trajectories_xmagical( policy_fn, high_policy_fn, policy_rep_fn, env, num_episodes,
                    base_observation, goal, num_video_episodes,
                    use_waypoints,
                    eval_temperature,
                    goal_info, config):
    
    from gymnasium.utils.save_video import save_video
    from collections import defaultdict
    
    renders = []
    observation, done = env.reset(), False
    
    for i in range(num_episodes + num_video_episodes):
        while not done:
            if not use_waypoints:
                cur_obs_goal = goal
                if config['use_rep']:
                    cur_obs_goal_rep = policy_rep_fn(targets=cur_obs_goal, bases=observation)
                else:
                    cur_obs_goal_rep = cur_obs_goal
            else:
                cur_obs_goal = high_policy_fn(observations=observation, goals=goal, temperature=eval_temperature)
                if config['use_rep']:
                    cur_obs_goal = cur_obs_goal / np.linalg.norm(cur_obs_goal, axis=-1, keepdims=True) * np.sqrt(cur_obs_goal.shape[-1])
                else:
                    cur_obs_goal = observation + cur_obs_goal
                cur_obs_goal_rep = cur_obs_goal
                
            action = policy_fn(observations=observation, goals=cur_obs_goal_rep, low_dim_goals=True, temperature=eval_temperature)
            next_observation, r, done, info = env.step(action)
            renders.append(next_observation)
        save_video(frames=renders, video_folder="test", fps=4)