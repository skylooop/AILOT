import collections
import os.path
from typing import Optional
import jax
import jax.numpy as jnp
from ott.geometry import pointcloud, costs
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
import d4rl
import gym
import numpy as np
from tqdm import tqdm

from absl import flags

FLAGS = flags.FLAGS


Batch = collections.namedtuple(
    "Batch", ["observations", "actions", "rewards", "masks", "next_observations"]
)


def split_into_trajectories(
    observations, actions, rewards, masks, dones_float, next_observations
):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append(
            (
                observations[i],
                actions[i],
                rewards[i],
                masks[i],
                dones_float[i],
                next_observations[i],
            )
        )
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    masks = []
    dones_float = []
    next_observations = []

    for traj in trajs:
        for (obs, act, rew, mask, done, next_obs) in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)

    return (
        np.stack(observations),
        np.stack(actions),
        np.stack(rewards),
        np.stack(masks),
        np.stack(dones_float),
        np.stack(next_observations),
    )


class Dataset:
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        masks: np.ndarray,
        dones_float: np.ndarray,
        next_observations: np.ndarray,
        size: int,
    ) -> None:

        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(
            observations=self.observations[indx],
            actions=self.actions[indx],
            rewards=self.rewards[indx],
            masks=self.masks[indx],
            next_observations=self.next_observations[indx],
        )


class D4RLDataset(Dataset):
    def __init__(self, env: gym.Env, clip_to_eps: bool = True, eps: float = 1e-5):
        if os.path.isfile(
            os.path.join(FLAGS.path_to_save_env, f"dataset_{env.spec.id}.npz")
        ):
            dataset = dict(
                np.load(
                    os.path.join(FLAGS.path_to_save_env, f"dataset_{env.spec.id}.npz")
                )
            )
        else:
            dataset = d4rl.qlearning_dataset(env)
            np.savez(
                os.path.join(FLAGS.path_to_save_env, f"dataset_{env.spec.id}.npz"),
                **dataset,
            )
            print("saved")

        if clip_to_eps:
            lim = 1 - eps
            dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

        dones_float = np.zeros_like(dataset["rewards"])

        for i in range(len(dones_float) - 1):
            if (
                np.linalg.norm(
                    dataset["observations"][i + 1] - dataset["next_observations"][i]  # first term - before terminal state, second - next obs after reset
                ) > 1e-6 or dataset["terminals"][i] == 1.0):
                    dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        super().__init__(
            dataset["observations"].astype(np.float32),
            actions=dataset["actions"].astype(np.float32),
            rewards=dataset["rewards"].astype(np.float32),
            masks=1.0 - dataset["terminals"].astype(np.float32),
            dones_float=dones_float.astype(np.float32),
            next_observations=dataset["next_observations"].astype(np.float32),
            size=len(dataset["observations"]),
        )


class ReplayBuffer(Dataset):
    def __init__(
        self, observation_space: gym.spaces.Box, action_dim: int, capacity: int
    ):

        observations = np.empty(
            (capacity, *observation_space.shape), dtype=observation_space.dtype
        )
        actions = np.empty((capacity, action_dim), dtype=np.float32)
        rewards = np.empty((capacity,), dtype=np.float32)
        masks = np.empty((capacity,), dtype=np.float32)
        dones_float = np.empty((capacity,), dtype=np.float32)
        next_observations = np.empty(
            (capacity, *observation_space.shape), dtype=observation_space.dtype
        )
        super().__init__(
            observations=observations,
            actions=actions,
            rewards=rewards,
            masks=masks,
            dones_float=dones_float,
            next_observations=next_observations,
            size=0,
        )

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity
        print("RN", capacity)

    def initialize_with_dataset(
        self, dataset: Dataset, num_samples: Optional[int], *args
    ):
        assert (
            self.insert_index == 0
        ), "Can insert a batch online in an empty replay buffer."

        dataset_size = len(dataset.observations)

        if num_samples is None:
            num_samples = dataset_size
        else:
            num_samples = min(dataset_size, num_samples)
        assert (
            self.capacity >= num_samples
        ), "Dataset cannot be larger than the replay buffer capacity."

        if num_samples < dataset_size:
            perm = np.random.permutation(dataset_size)
            indices = perm[:num_samples]
        else:
            indices = np.arange(num_samples)

        self.observations[:num_samples] = dataset.observations[indices]
        self.actions[:num_samples] = dataset.actions[indices]
        self.rewards[:num_samples] = dataset.rewards[indices]
        self.masks[:num_samples] = dataset.masks[indices]
        self.dones_float[:num_samples] = dataset.dones_float[indices]
        self.next_observations[:num_samples] = dataset.next_observations[indices]

        self.insert_index = num_samples
        self.size = num_samples

    def insert(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        mask: float,
        done_float: float,
        next_observation: np.ndarray,
    ):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.dones_float[self.insert_index] = done_float
        self.next_observations[self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


def compute_rewards(self_states_pair: np.ndarray, expert_states_pair: np.ndarray):

    x = jnp.asarray(self_states_pair)
    y = jnp.asarray(expert_states_pair)
    a = jnp.ones((x.shape[0],))
    b = jnp.ones((y.shape[0],))
    a, b = a / jnp.sum(a), b / jnp.sum(b)

    geom = pointcloud.PointCloud(x, y, epsilon=0.01, cost_fn=costs.Euclidean())
    ot_prob = linear_problem.LinearProblem(geom, a, b)
    solver = sinkhorn.Sinkhorn()
    ot_sink = solver(ot_prob)

    transp_cost = jnp.sum(ot_sink.matrix * geom.cost_matrix, axis=1)
    # rewards = 5 * jnp.exp(-transp_cost * transp_cost.shape[0]) - 2
    # rewards = 2 - 2000 * transp_cost
    rewards = -transp_cost

    return rewards


def compute_rewards_2(x, y, epsilon=0.01, niter=500):
    mu_x = np.ones(x.shape[0]) * (1 / x.shape[0])
    mu_y = np.ones(y.shape[0]) * (1 / y.shape[0])
    c_m = ot.utils.dist(x, y, metric="cosine")
    transport_plan = ot.sinkhorn_unbalanced(
        mu_x, mu_y, c_m, epsilon, 20, numItermax=niter
    )

    transp_cost = jnp.sum(transport_plan * c_m, axis=1)
    rewards = 5 * jnp.exp(-transp_cost * transp_cost.shape[0]) - 2

    return rewards


class RewardsScaler:
    def init(self, rewards: np.ndarray):
        self.min = np.quantile(np.abs(rewards).reshape(-1), 0.01)
        self.max = np.quantile(np.abs(rewards).reshape(-1), 0.99)
        print("min max", self.min, self.max)

    def scale(self, rewards: np.ndarray):
        return 5 * np.exp((rewards - self.min) / self.max) - 2


class OTReplayBuffer(ReplayBuffer):
    def __init__(
        self, observation_space: gym.spaces.Box, action_dim: int, capacity: int
    ):
        super().__init__(observation_space, action_dim, capacity)

        self.observations_cur = []
        self.actions_cur = []
        self.masks_cur = []
        self.dones_float_cur = []
        self.next_observations_cur = []

        self.scaler = RewardsScaler()

    def initialize_with_dataset(
        self,
        dataset: Dataset,
        num_samples: Optional[int],
        expert_states_pair: np.ndarray,
    ):

        self.expert_states_pair = expert_states_pair

        i0 = 0

        for i in tqdm(range(num_samples)):
            if dataset.dones_float[i] == 1.0 and i + 1 < len(dataset.observations):
                i1 = i

                self.observations[i0:i1] = dataset.observations[i0:i1]
                self.actions[i0:i1] = dataset.actions[i0:i1]
                self.masks[i0:i1] = dataset.masks[i0:i1]
                self.dones_float[i0:i1] = dataset.dones_float[i0:i1]
                self.next_observations[i0:i1] = dataset.next_observations[i0:i1]

                self_states_pair = np.concatenate(
                    [self.observations[i0:i1], self.next_observations[i0:i1]], axis=1
                )
                self.rewards[i0:i1] = np.asarray(
                    compute_rewards(self_states_pair, expert_states_pair)
                )

                self.insert_index = i1
                self.size = i1

                i0 = i1

        self.scaler.init(self.rewards[:i1])
        self.rewards[:i1] = self.scaler.scale(self.rewards[:i1])
        print(
            "rewards:",
            np.min(self.rewards[:i1]),
            np.mean(self.rewards[:i1]),
            np.max(self.rewards[:i1]),
        )

    def insert(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        mask: float,
        done_float: float,
        next_observation: np.ndarray,
    ):

        self.observations_cur.append(observation)
        self.actions_cur.append(action)
        self.masks_cur.append(mask)
        self.dones_float_cur.append(done_float)
        self.next_observations_cur.append(next_observation)

        if done_float == 1.0:

            i0 = self.insert_index
            i1 = min(i0 + len(self.observations_cur), self.capacity)

            self.observations[i0:i1] = np.stack(self.observations_cur)[: i1 - i0]
            self.actions[i0:i1] = np.stack(self.actions_cur)[: i1 - i0]
            self.masks[i0:i1] = np.stack(self.masks_cur)[: i1 - i0]
            self.dones_float[i0:i1] = np.stack(self.dones_float_cur)[: i1 - i0]
            self.next_observations[i0:i1] = np.stack(self.next_observations_cur)[
                : i1 - i0
            ]

            self_states_pair = np.concatenate(
                [self.observations[i0:i1], self.next_observations[i0:i1]], axis=1
            )
            self.rewards[i0:i1] = self.scaler.scale(
                np.asarray(compute_rewards(self_states_pair, self.expert_states_pair))
            )

            self.insert_index = i1 % self.capacity
            self.size = min(self.size + (i1 - i0), self.capacity)

            self.observations_cur = []
            self.actions_cur = []
            self.masks_cur = []
            self.dones_float_cur = []
            self.next_observations_cur = []
