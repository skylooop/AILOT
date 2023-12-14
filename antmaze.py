import warnings
from training import GCSDataset, pretrain_agent
warnings.filterwarnings("ignore")
import sys
import os

os.environ['LD_LIBRARY_PATH'] = '/home/nazar/.mujoco/mujoco210/bin:/usr/lib/nvidia'
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
import gym
import d4rl
import numpy as np
from src.agents import icvf
import equinox as eqx
import jax
import jax.numpy as jnp
import functools
from src.d4rl_utils import get_dataset
import dataclasses


print(jax.devices())


env = gym.make("antmaze-large-diverse-v2")
d4rl_ds = d4rl.qlearning_dataset(env)
dones_float = np.zeros_like(d4rl_ds['rewards'])
for i in range(len(dones_float) - 1):
    if np.linalg.norm(d4rl_ds['observations'][i + 1] - d4rl_ds['next_observations'][i]) > 1e-6 or d4rl_ds['terminals'][i] == 1.0:
        dones_float[i] = 1
    else:
        dones_float[i] = 0
dones_float[-1] = 1
d4rl_ds['dones_float'] = dones_float


dataset = get_dataset(env)
gcsds_params = {"p_currgoal": 0.2, "p_randomgoal": 0.3, "p_trajgoal":0.5, "discount": 0.999, "geom_sample": True}
expert_trajectory = d4rl_ds['observations'][np.arange(start=10000, stop=10190)]
gc_dataset = GCSDataset(dataset, **gcsds_params, expert_trajectory=expert_trajectory, way_steps=20)

icvf_model = icvf.create_eqx_learner(seed=42,
                                     observations=d4rl_ds['observations'][0],
                                     hidden_dims=[256, 256],
                                     load_pretrained_icvf=True)


intents_learner, actor_learner = pretrain_agent(env, icvf_model, gc_dataset, expert_trajectory, batch_size=1024, num_iter=3_000)