from jaxrl_m.dataset import Dataset
import ml_collections

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
    intent_sametraj: bool = False
    
    @staticmethod
    def get_default_config():
        return ml_collections.ConfigDict({
            'p_randomgoal': 0.3,
            'p_trajgoal': 0.5,
            'p_currgoal': 0.2,
            'geom_sample': 0,
            'reward_scale': 1.0,
            'reward_shift': -1.0,
            'terminal': True,
        })
        
    def sample(self, batch_size: int, indx=None):
        if indx is None:
            indx = np.random.randint(self.dataset.size-1, size=batch_size)

        batch = self.dataset.sample(batch_size, indx)
        
        ailot_subgoals = self.expert_subgoals[np.random.randint(low=0, high=self.expert_subgoals.shape[0], size=batch_size)]
        batch['ailot_subgoals'] = ailot_subgoals # subgoals from trajectory
        batch['ailot_goal'] = self.expert_subgoals[-1][None] # final goal
        
        return batch
