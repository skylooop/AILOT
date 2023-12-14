import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("..")

import os
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'
os.environ['LD_LIBRARY_PATH'] = '/home/nazar/.mujoco/mujoco210/bin:/usr/lib/nvidia'

import gym
import d4rl

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib import patches

import equinox as eqx
import jax
import jax.numpy as jnp
import functools

from tqdm.auto import tqdm
from jaxrl_m.common import TrainStateEQX
# from src.agents.iql_equinox import GaussianPolicy, GaussianIntentPolicy

from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from ott.tools import plot, sinkhorn_divergence
from ott.solvers.linear import implicit_differentiation as imp_diff

import optax

@eqx.filter_vmap(in_axes=dict(ensemble=eqx.if_array(0), s=None))
def eval_ensemble_psi(ensemble, s):
    return eqx.filter_vmap(ensemble.psi_net)(s)

@eqx.filter_vmap(in_axes=dict(ensemble=eqx.if_array(0), s=None))
def eval_ensemble_phi(ensemble, s):
    return eqx.filter_vmap(ensemble.phi_net)(s)

@eqx.filter_vmap(in_axes=dict(ensemble=eqx.if_array(0), s=None, g=None, z=None))
def eval_ensemble_icvf_viz(ensemble, s, g, z):
    return eqx.filter_vmap(ensemble.classic_icvf_initial)(s, g, z)

@eqx.filter_vmap(in_axes=dict(ensemble=eqx.if_array(0), s=None, g=None, z=None)) # V(s, g, z), g - dim 29, z - dim 256
def eval_ensemble_icvf_latent_z(ensemble, s, g, z):
    return eqx.filter_vmap(ensemble.classic_icvf)(s, g, z)

@eqx.filter_vmap(in_axes=dict(ensemble=eqx.if_array(0), s=None, g=None, z=None)) # V(s, g ,z ), g, z - dim 256
def eval_ensemble_icvf_latent_zz(ensemble, s, g, z):
    return eqx.filter_vmap(ensemble.icvf_zz)(s, g, z)

@eqx.filter_vmap(in_axes=dict(ensemble=eqx.if_array(0), s=None, z=None)) 
def eval_ensemble_gotil(ensemble, s, z):
    return eqx.filter_vmap(ensemble.gotil)(s, z)

@eqx.filter_jit
def get_gcvalue(agent, s, g, z):
    v_sgz_1, v_sgz_2 = eval_ensemble_icvf_viz(agent.value_learner.model, s, g, z)
    return (v_sgz_1 + v_sgz_2) / 2

def get_v_gz(agent, initial_state, target_goal, observations):
    initial_state = jnp.tile(initial_state, (observations.shape[0], 1))
    target_goal = jnp.tile(target_goal, (observations.shape[0], 1))
    return -1 * get_gcvalue(agent, initial_state, observations, target_goal)
    
def get_v_zz(agent, goal, observations):
    goal = jnp.tile(goal, (observations.shape[0], 1))
    return get_gcvalue(agent, observations, goal, goal)

@eqx.filter_vmap(in_axes=dict(agent=None, obs=None, goal=0))
def get_v_zz_heatmap(agent, obs, goal): # goal - whole traj
    goal = jnp.tile(goal, (obs.shape[0], 1))
    return get_gcvalue(agent, obs, goal, goal)


from jaxrl_m.dataset import Dataset

from jaxtyping import *
import dataclasses
import numpy as np
import jax

@dataclasses.dataclass
class GCDataset:
    dataset: Dataset
    
    p_randomgoal: float
    p_trajgoal: float
    p_currgoal: float
    geom_sample: int
    discount: float
    p_samegoal: float = 0.5
    terminal_key: str = 'dones_float'
    reward_scale: float = 1.0
    reward_shift: float = -1.0
    terminal: bool = True
    curr_goal_shift: int = 0
    
    expert_trajectory: ArrayLike = None
    expert_subgoals: ArrayLike = None
        
    def __post_init__(self):
        self.terminal_locs, = np.nonzero(self.dataset[self.terminal_key] > 0)
        assert np.isclose(self.p_randomgoal + self.p_trajgoal + self.p_currgoal, 1.0)

    def sample_goals(self, indx, p_randomgoal=None, p_trajgoal=None, p_currgoal=None):
        if p_randomgoal is None:
            p_randomgoal = self.p_randomgoal
        if p_trajgoal is None:
            p_trajgoal = self.p_trajgoal
        if p_currgoal is None:
            p_currgoal = self.p_currgoal

        batch_size = len(indx)
        goal_indx = np.random.randint(self.dataset.size, size=batch_size)
        final_state_indx = self.terminal_locs[np.searchsorted(self.terminal_locs, indx)]

        distance = np.random.rand(batch_size)
        if self.geom_sample:
            us = np.random.rand(batch_size)
            middle_goal_indx = np.minimum(indx + np.ceil(np.log(1 - us) / np.log(self.discount)).astype(int), final_state_indx)
        else:
            middle_goal_indx = np.round((np.minimum(indx + 1, final_state_indx) * distance + final_state_indx * (1 - distance))).astype(int)

        goal_indx = np.where(np.random.rand(batch_size) < p_trajgoal / (1.0 - p_currgoal), middle_goal_indx, goal_indx)
        goal_indx = np.where(np.random.rand(batch_size) < p_currgoal, indx, goal_indx)
        
        return goal_indx

    def sample(self, batch_size: int, indx=None):
        if indx is None:
            indx = np.random.randint(self.dataset.size-1, size=batch_size)
        
        batch = self.dataset.sample(batch_size, indx)
        goal_indx = self.sample_goals(indx)

        success = (indx == goal_indx)
        batch['rewards'] = success.astype(float) * self.reward_scale + self.reward_shift
        if self.terminal:
            batch['masks'] = (1.0 - success.astype(float))
        else:
            batch['masks'] = np.ones(batch_size)
        batch['goals'] = jax.tree_map(lambda arr: arr[goal_indx], self.dataset['observations'])

        return batch

@dataclasses.dataclass
class GCSDataset(GCDataset):
    way_steps: int = 25
    # high_p_randomgoal: float = 0.3
    
    def sample(self, batch_size: int):
        
        indx = np.random.randint(self.dataset.size-1, size=batch_size)

        batch = self.dataset.sample(batch_size, indx)

        final_state_indx = self.terminal_locs[np.searchsorted(self.terminal_locs, indx)] # find boudaries of traj for curr state
        way_indx = np.minimum(indx + self.way_steps, final_state_indx)
        batch['ailot_way_goals'] = jax.tree_map(lambda arr: arr[way_indx], self.dataset['observations']) # s_{t+k}
        goal_add = np.random.randint(1, self.way_steps, size=batch_size)
        way_indx_2 = np.minimum(indx + goal_add, final_state_indx)
        batch['ailot_subway_goals'] = jax.tree_map(lambda arr: arr[way_indx_2], self.dataset['observations']) # s_{t+[1,k]}
        
        return batch


import random


def actor_loss(model, obs, actions, v, next_v, goals):
    adv = next_v - v
    exp_a = jnp.minimum(jnp.exp(adv * 10.0), 100.0)
    actor_dist = eqx.filter_vmap(model)(obs, goals)
    log_prob = actor_dist.log_prob(actions)
    loss = -(exp_a * log_prob).mean()
    return loss


def intents_loss(model, obs, goals, key):
    z_dist = eqx.filter_vmap(model)(obs)
    z = z_dist.sample(seed=key)
    return jnp.sqrt(((z - goals)**2).sum(-1)).mean()
    # return optax.l2_loss(z, goals).mean()


def ot_intents_loss(models, obs, goals, key) -> float:
    z_model, v_model = models
    z_dist = eqx.filter_vmap(model)(obs)
    z = z_dist.sample(seed=key)

    v = eval_ensemble_gotil(v_model, obs, z).mean(0).squeeze()     
    an = v / v.sum()
    bn = jnp.ones(goals.shape[0]) / goals.shape[0]

    geom = pointcloud.PointCloud(x=z, y=goals, epsilon=0.001)

    ot = sinkhorn_divergence.sinkhorn_divergence(
        geom,
        a=an,
        b=bn,
        x=geom.x,
        y=geom.y,
        epsilon=0.001,
        static_b=True,
        sinkhorn_kwargs={
            "implicit_diff": imp_diff.ImplicitDiff(),
            "use_danskin": True,
            "max_iterations": 1000
        },
    )

    return ot.divergence


def pretrain_agent(
    env,
    icvf_model,
    gc_dataset,
    expert_trajectory,
    num_iter: int = 6_000,
    batch_size: int = 1024,
):

    key = jax.random.PRNGKey(42)

    intents_learner = TrainStateEQX.create(
        model=GaussianIntentPolicy(key=key,
                             hidden_dims=[512, 512],
                             state_dim=29,
                             intent_dim=256), 
                             optim=optax.adam(learning_rate=1e-3))

    actor_learner = TrainStateEQX.create(
            model=GaussianPolicy(key=key,
                                 hidden_dims=[512, 512],
                                 state_dim=env.observation_space.shape[0],
                                 intents_dim=256,
                                 action_dim=env.action_space.shape[0]),
            optim=optax.adam(learning_rate=1e-3)
        )
    
        
    intents_vg = eqx.filter_jit(eqx.filter_value_and_grad)(intents_loss, has_aux=False)
    actor_vg = eqx.filter_jit(eqx.filter_value_and_grad)(actor_loss, has_aux=False)
 
    @eqx.filter_jit
    def make_step(intents_learner, actor_learner, batch, key):
        obs = batch["observations"]
        next_obs = batch["next_observations"]
        actions = batch["actions"]
        goals = eval_ensemble_psi(icvf_model.value_learner.model, batch['ailot_subway_goals']).mean(axis=0) 

        sampled_goals = eqx.filter_vmap(intents_learner.model)(obs).sample(seed=key)

        v = eval_ensemble_icvf_latent_zz(icvf_model.value_learner.model, obs, sampled_goals, sampled_goals).mean(0)
        next_v = eval_ensemble_icvf_latent_zz(icvf_model.value_learner.model, next_obs, sampled_goals, sampled_goals).mean(0) 

        intents_cost, intents_grads = intents_vg(intents_learner.model, obs, goals, key)
        
        actor_cost, actor_grads = actor_vg(actor_learner.model, obs, actions, v, next_v, sampled_goals)

        intents_learner = intents_learner.apply_updates(intents_grads)
        actor_learner = actor_learner.apply_updates(actor_grads)

        return intents_learner, actor_learner, intents_cost, actor_cost
        
    pbar = tqdm(range(num_iter))
    
    for i in pbar:
        key, sample_key = jax.random.split(key, 2)
        batch = gc_dataset.sample(batch_size)
        intents_learner, actor_learner, intents_cost, actor_cost = make_step(
            intents_learner, actor_learner, key=sample_key, batch=batch
        )

        pbar.set_postfix({"intents_cost": intents_cost, "actor_cost": actor_cost})
    return intents_learner, actor_learner
    


def train_gotil(
    env,
    icvf_model,
    gc_dataset,
    expert_trajectory,
    intents_learner, 
    num_iter: int = 6_000,
    batch_size: int = 1024,
):

    key = jax.random.PRNGKey(42)
        
    ot_intents_vg = eqx.filter_jit(eqx.filter_value_and_grad)(ot_intents_loss, has_aux=False)
    
    @eqx.filter_jit
    def make_step(intents_learner, value_learner, batch, key):
        obs = batch["observations"]
        
        sampled_goals = eqx.filter_vmap(intents_learner.model)(obs).sample(seed=key)

        T = expert_trajectory.shape[0]
        goal_add = np.random.randint(1, 20, size=T)
        way_indx = np.minimum(np.arange(0, T) + goal_add, T-1)
        expert_goals = eval_ensemble_psi(icvf_model.value_learner.model, expert_trajectory[way_indx]).mean(axis=0)
        
        gotil_cost, (intents_grads, gotil_grads) = ot_intents_vg((intents_learner.model, value_learner.model), obs, expert_goals, key)
        value_learner = value_learner.apply_updates(gotil_grads)
        intents_learner = intents_learner.apply_updates(intents_grads)
        
        return intents_learner, value_learner, gotil_cost
        
    pbar = tqdm(range(num_iter))

    value_learner = TrainStateEQX.create(
        model=icvf_model.value_learner.model, 
        optim=optax.adam(learning_rate=1e-3)
    )
    
    for i in pbar:
        key, sample_key = jax.random.split(key, 2)
        batch = gc_dataset.sample(batch_size)
        intents_learner, value_learner, gotil_cost = make_step(
            intents_learner, value_learner, key=sample_key, batch=batch
        )
        pbar.set_postfix({"gotil_cost": gotil_cost})
    return intents_learner, value_learner


def intents_loss_gotil(model, obs, v_model, key):
    z_dist = eqx.filter_vmap(model)(obs)
    z = z_dist.sample(seed=key)
    loss = -eval_ensemble_gotil(v_model, obs, z).mean(0) * 10
    return loss.mean()


def train_agent_with_gotil(
    env,
    gc_dataset,
    actor_learner,
    intents_learner,
    value_learner,
    num_iter: int = 6_000,
    batch_size: int = 1024,
):

    key = jax.random.PRNGKey(42)    
        
    actor_vg = eqx.filter_jit(eqx.filter_value_and_grad)(actor_loss, has_aux=False)
    # intents_vg = eqx.filter_jit(eqx.filter_value_and_grad)(intents_loss_gotil, has_aux=False)
 
    @eqx.filter_jit
    def make_step(intents_learner, actor_learner, batch, key):
        obs = batch["observations"]
        next_obs = batch["next_observations"]
        actions = batch["actions"]

        sampled_goals = eqx.filter_vmap(intents_learner.model)(obs).sample(seed=key)

        v = eval_ensemble_gotil(value_learner.model, obs, sampled_goals).mean(0)
        next_v = eval_ensemble_gotil(value_learner.model, next_obs, sampled_goals).mean(0) 
        
        actor_cost, actor_grads = actor_vg(actor_learner.model, obs, actions, v, next_v, sampled_goals)
        actor_learner = actor_learner.apply_updates(actor_grads)

        # intents_cost, intents_grads = intents_vg(intents_learner.model, obs, value_learner.model, key)
        # intents_learner = intents_learner.apply_updates(intents_grads)

        return intents_learner, actor_learner, intents_cost, actor_cost
        
    pbar = tqdm(range(num_iter))
    
    for i in pbar:
        key, sample_key = jax.random.split(key, 2)
        batch = gc_dataset.sample(batch_size)
        intents_learner, actor_learner, intents_cost, actor_cost = make_step(
            intents_learner, actor_learner, key=sample_key, batch=batch
        )

        pbar.set_postfix({"actor_cost": actor_cost, "intents_cost": intents_cost})
    return intents_learner, actor_learner
    
