from jaxrl_m.dataset import Dataset
from flax.core.frozen_dict import FrozenDict
from flax.core import freeze
import dataclasses
import numpy as np
import jax

@dataclasses.dataclass
class GCDataset:
    dataset: Dataset
    ds_type: str
    
    p_randomgoal: float
    p_trajgoal: float
    p_currgoal: float
    geom_sample: int
    discount: float
    terminal_key: str = 'dones_float'
    reward_scale: float = 1.0
    reward_shift: float = -1.0
    terminal: bool = True
    
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
        final_state_indx = self.terminal_locs[np.searchsorted(self.terminal_locs, indx) -1 ]

        distance = np.random.rand(batch_size)
        if self.geom_sample:
            us = np.random.rand(batch_size)
            middle_goal_indx = np.minimum(indx + np.ceil(np.log(1 - us) / np.log(self.discount)).astype(int), final_state_indx)
        else:
            middle_goal_indx = np.round((indx * distance + final_state_indx * (1 - distance))).astype(int)

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
    way_steps: int = None
    high_p_randomgoal: float = 0.
    
    def sample(self, batch_size: int, indx=None):
        if indx is None:
            indx = np.random.randint(self.dataset.size-1, size=batch_size)

        batch = self.dataset.sample(batch_size, indx)
        goal_indx = self.sample_goals(indx)
        
        # HER relabelling
        icvf_desired_goal_indx = self.sample_goals(indx) # intermidiate state
        icvf_goal_indx = np.where(np.random.rand(batch_size) < self.p_randomgoal, icvf_desired_goal_indx, goal_indx)
        
        batch['icvf_goals'] = jax.tree_map(lambda arr: arr[icvf_goal_indx], self.dataset['observations'])
        batch['icvf_desired_goals'] = jax.tree_map(lambda arr: arr[icvf_desired_goal_indx], self.dataset['observations'])
        
        icvf_success = (indx == icvf_goal_indx)
        icvf_desired_success = (indx == icvf_desired_goal_indx)
        
        batch['icvf_rewards'] = icvf_success.astype(float) * self.reward_scale + self.reward_shift
        batch['icvf_desired_rewards'] = icvf_desired_success.astype(float) * self.reward_scale + self.reward_shift
        
        batch['icvf_goals'] = jax.tree_map(lambda arr: arr[icvf_goal_indx], self.dataset['observations'])
        batch['icvf_desired_goals'] = jax.tree_map(lambda arr: arr[icvf_desired_goal_indx], self.dataset['observations'])
        
        batch['icvf_masks'] = (1.0 - icvf_success.astype(float))
        batch['icvf_desired_masks'] = (1.0 - icvf_desired_success.astype(float))
        
        return batch
