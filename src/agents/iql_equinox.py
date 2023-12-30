import dataclasses
from dataclasses import dataclass, asdict
from typing import *
import contextlib
import functools

import warnings
warnings.filterwarnings('ignore')

import jax
import jax.numpy as jnp
import distrax

from tqdm.auto import tqdm
import numpy as np
import equinox as eqx
import optax
from jaxtyping import *

import gym
import d4rl
import minari

import wandb
import pyrallis

@dataclass
class TrainConfig:
    project: str = "OfflineRL"
    group: str = "IQL-EQX_ICVF_Scale"
    name: str = "IQL_EQX"
    dataset_id: str = "maze2d-umaze-v1" #antmaze-large-diverse-v2
    discount: float = 0.999
    tau: float = 0.005
    beta: float = 10.0
    iql_tau: float = 0.9 #expectile
    # total gradient updates during training
    max_timesteps: int = int(1e6)
    # training batch size
    batch_size: int = 512
    # whether to normalize states
    normalize_state: bool = True
    # whether to normalize reward (like in IQL)
    normalize_reward: bool = True
    # V-critic function learning rate
    vf_lr: float = 3e-4
    # Q-critic learning rate
    qf_lr: float = 3e-4
    # actor learning rate
    actor_lr: float = 3e-4
    # evaluation frequency, will evaluate every eval_freq training steps
    eval_freq: int = int(50_000)
    # number of episodes to run during evaluation
    n_episodes: int = 100
    # path for checkpoints saving, optional
    checkpoints_path: Optional[str] = None
    actor_schedule: str = "none"
    use_icvf_pretrain: bool = False
    # training random seed
    seed: int = 42

class TrainState(eqx.Module):
    model: eqx.Module
    optim: optax.GradientTransformation
    optim_state: optax.OptState

    @classmethod
    def create(cls, *, model, optim, **kwargs):
        optim_state = optim.init(eqx.filter(model, eqx.is_array))
        return cls(model=model, optim=optim, optim_state=optim_state,
                   **kwargs)
    
    @eqx.filter_jit
    def apply_updates(self, grads):
        updates, new_optim_state = self.optim.update(grads, self.optim_state, self.model)
        new_model = eqx.apply_updates(self.model, updates)
        return dataclasses.replace(
            self,
            model=new_model,
            optim_state=new_optim_state
        )

class TrainTargetState(TrainState):
    target_model: eqx.Module

    @classmethod
    def create(cls, *, model, target_model, optim, **kwargs):
        optim_state = optim.init(eqx.filter(model, eqx.is_array))
        return cls(model=model, optim=optim, optim_state=optim_state, target_model=target_model,
                   **kwargs)

    def soft_update(self, tau: float = 0.005):
        model_params = eqx.filter(self.model, eqx.is_array)
        target_model_params, target_model_static = eqx.partition(self.target_model, eqx.is_array)

        new_target_params = optax.incremental_update(model_params, target_model_params, tau)
        return dataclasses.replace(
            self,
            model=self.model,
            target_model=eqx.combine(new_target_params, target_model_static)
        )
    
    def apply_updates(self, grads):
        
        updates, new_optim_state = self.optim.update(grads, self.optim_state)
        new_model = eqx.apply_updates(self.model, updates)
        return dataclasses.replace(
            self,
            model=new_model,
            optim_state=new_optim_state
        )

class FixedDistrax(eqx.Module):
    cls: type
    args: PyTree[Any]
    kwargs: PyTree[Any]

    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def sample_and_log_prob(self, *, seed):
        return self.cls(*self.args, **self.kwargs).sample_and_log_prob(seed=seed)

    def sample(self, *, seed):
        return self.cls(*self.args, **self.kwargs).sample(seed=seed)

    def log_prob(self, x):
        return self.cls(*self.args, **self.kwargs).log_prob(x)

    def sample_n(self, key, n):
        return self.cls(*self.args, **self.kwargs)._sample_n(key=key, n=n)
    
    def mean(self):
        return self.cls(*self.args, **self.kwargs).mean()
    
    
class ReplayBuffer(eqx.Module):
    data: Dict[str, jax.Array]

    @classmethod
    def create_from_d4rl(cls, env, normalize_reward=True, normalize_state=True) -> "ReplayBuffer":
        dataset = d4rl.qlearning_dataset(env)
        if normalize_reward:
            dataset = modify_reward(dataset, env_name=env.spec.id)
        if normalize_state:
            state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
        else:
            state_mean, state_std = 0, 1
            
        dataset["observations"] = normalize_states(
            dataset["observations"], state_mean, state_std
        )
        dataset["next_observations"] = normalize_states(
            dataset["next_observations"], state_mean, state_std
        )
        buffer = {
            "observations": jnp.asarray(dataset["observations"], dtype=jnp.float32),
            "actions": jnp.asarray(dataset["actions"], dtype=jnp.float32),
            "rewards": jnp.asarray(dataset["rewards"], dtype=jnp.float32),
            "next_observations": jnp.asarray(dataset["next_observations"], dtype=jnp.float32),
            "dones": jnp.asarray(dataset['terminals'], dtype=jnp.float32)
        }
        return cls(data=buffer), state_mean, state_std

    @property
    def size(self):
        return self.data["observations"].shape[0]

    @functools.partial(jax.jit, static_argnames='batch_size')
    def sample_batch(self, key: jax.random.PRNGKey, batch_size: int) -> Dict[str, jax.Array]:
        indices = jax.random.randint(key=key, shape=(batch_size, ), minval=0, maxval=self.size)
        batch = jax.tree_map(lambda arr: arr[indices], self.data)
        return batch

def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    
    def normalize_state(state):
        return (state - state_mean) / state_std
    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


def qlearning_dataset(dataset: minari.MinariDataset) -> Dict[str, np.ndarray]:
    obs, next_obs, actions, rewards, dones = [], [], [], [], []

    for idx, episode in enumerate(dataset):
        obs.append(episode.observations[:-1].astype(jnp.float32)) # fix for other than antmaze
        next_obs.append(episode.observations[1:].astype(jnp.float32))
        actions.append(episode.actions.astype(jnp.float32))
        rewards.append(episode.rewards)
        dones.append(episode.terminations)
        
    return {
        "observations": jnp.concatenate(obs),
        "actions": jnp.concatenate(actions),
        "next_observations": jnp.concatenate(next_obs),
        "rewards": jnp.concatenate(rewards),
        "terminals": jnp.concatenate(dones),
    }

def return_reward_range(
    dataset: Dict[str, np.ndarray], max_episode_steps: int
) -> Tuple[float, float]:
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)

def modify_reward(
    dataset: Dict[str, np.ndarray], env_name: str, max_episode_steps: int = 1000
):
    if any(s in env_name.lower() for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0
    return dataset

def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std

def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std

class QNet(eqx.Module):
    hidden_dims: tuple[int] = (256, 256)
    net: eqx.Module
    
    def __init__(self, key, state_dim, action_dim):
        key, mlp_key = jax.random.split(key, 2)
        self.net = eqx.nn.MLP(in_size=state_dim + action_dim, 
                              out_size=1, depth=len(self.hidden_dims), width_size=self.hidden_dims[-1],
                              key=mlp_key)
    
    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], axis=-1)
        return self.net(x)

@eqx.filter_vmap(in_axes=dict(ensemble=eqx.if_array(0), state=None, action=None), out_axes=0)
def eval_ensemble(ensemble, state, action):
    return eqx.filter_vmap(ensemble)(state, action)

class VNet(eqx.Module):
    hidden_dims: tuple[int] = (256, 256)
    net: eqx.Module
    icvf_weights: Any = None
    
    def __init__(self, key, state_dim, use_icvf: bool = False):
        key, mlp_key = jax.random.split(key, 2)
        net = eqx.nn.MLP(in_size=state_dim, 
                              out_size=1, depth=len(self.hidden_dims), width_size=self.hidden_dims[-1],
                              key=mlp_key)
        if use_icvf:
            print("Loading Pretrained ICVF")
            icvf_net = eqx.nn.MLP(in_size=state_dim, 
                              out_size=self.hidden_dims[-1], depth=len(self.hidden_dims), width_size=self.hidden_dims[-1],
                              key=mlp_key)
            loaded_net = eqx.tree_deserialise_leaves("src/agents/icvf_model.eqx", icvf_net)
            net = eqx.tree_at(lambda mlp: mlp.layers[-1], loaded_net, net.layers[-1])
            # for loss
            is_linear = lambda x: isinstance(x, eqx.nn.Linear)
            get_weights = lambda m: [x.weight
                                    for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                                    if is_linear(x)]
            self.icvf_weights = jax.lax.stop_gradient(get_weights(net)[:-1]) # only first two layers are pretrained by icvf
            
        self.net = net
    
    def __call__(self, obs):
        return self.net(obs)

class TanhNormal(distrax.Transformed):
    def __init__(self, loc, scale):
        normal_dist = distrax.Normal(loc, scale)
        tanh_bijector = distrax.Tanh()
        super().__init__(distribution=normal_dist, bijector=tanh_bijector)

    def mean(self):
        return self.bijector.forward(self.distribution.mean())


class GaussianPolicy(eqx.Module):
    net: eqx.Module
    
    log_std_min: int = -20.0
    log_std_max: int = 2.0
    
    def __init__(self, key, state_dim, intents_dim, action_dim, hidden_dims):
        key, key_means, key_log_std = jax.random.split(key, 3)
        
        self.net = eqx.nn.MLP(in_size=state_dim + intents_dim,
                              out_size=2 * action_dim,
                              width_size=hidden_dims[0],
                              depth=len(hidden_dims),
                              activation=jax.nn.gelu,
                              key=key_means)
    
    def sample_actions(self, state, intentions, temperature):
        return self(state, intentions, temperature)
    
    def __call__(self, state, intentions, temperature=1.0):
        # Film conditioning??
        x = jnp.concatenate([state, intentions], axis=-1)
        means, log_std = jnp.split(self.net(x), 2)
        log_stds = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        dist = FixedDistrax(TanhNormal, means, jnp.exp(log_stds) * temperature)
        return dist

class GaussianIntentPolicy(eqx.Module):
    net: eqx.Module
    
    log_std_min: int = -20.0
    log_std_max: int = 2.0
    log_stds: Array
    
    def __init__(self, key, state_dim, intent_dim, hidden_dims):
        key, key_means, key_log_std = jax.random.split(key, 3)
        
        self.net = eqx.nn.MLP(in_size=state_dim,
                              out_size=intent_dim,
                              width_size=hidden_dims[0],
                              depth=len(hidden_dims),
                              activation=jax.nn.gelu,
                              key=key_means)
        self.log_stds = jnp.zeros(shape=(intent_dim, ), dtype=jnp.float32)
        
    def __call__(self, state):
        means = self.net(state)
        log_stds = jnp.clip(self.log_stds, self.log_std_min, self.log_std_max)
        dist = FixedDistrax(distrax.MultivariateNormalDiag, loc=means,
                            scale_diag=jnp.exp(log_stds))
        return dist

def expectile_loss(diff, expectile=0.9):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

class IQLagent(eqx.Module):
    q_learner: TrainState
    v_learner: TrainTargetState
    actor_learner: TrainState

    temperature: float = 10.0
    expectile: float = 0.9
    discount: float = 0.999

    @eqx.filter_jit
    def eval_actor(self, key, obs):
        return jnp.clip(self.actor_learner.model(obs).sample(seed=key), -1.0, 1.0)

is_linear = lambda x: isinstance(x, eqx.nn.Linear)
get_weights = lambda m: [x.weight
                         for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                         if is_linear(x)]

@eqx.filter_jit
def update_agent(agent, batch, buffer_key):
    def l2_norm(model):
        total = 0
        curr_weights = get_weights(model)[:-1]
        for idx, w in enumerate(curr_weights):
            total = total + jnp.sum(w ** 2 - agent.v_learner.model.icvf_weights[idx] ** 2)
        return total
    
    def v_loss_fn(v_net, q_learner):
        q1, q2 = eval_ensemble(q_learner, batch['observations'], batch['actions'])
        q = jnp.minimum(q1, q2)
        
        v = eqx.filter_vmap(v_net)(batch['observations'])
        value_loss = expectile_loss(q - v, expectile=agent.expectile).mean() #+ 100.0 * (agent.v_learner.model.icvf_weights).mean()
        advantage = q - v
        return value_loss, {
            'value_loss': value_loss,
            'abs adv mean': jnp.abs(advantage).mean(),
        }
        
    def q_loss_fn(q_net, v_learner):
        next_v = eqx.filter_vmap(v_learner)(batch['next_observations'])
        target = batch['rewards'][:, None] + agent.discount * (1.0 - batch['dones'][:, None]) * next_v
        q1, q2 = eval_ensemble(q_net, batch['observations'], batch['actions'])
        q_loss = ((q1 - target)**2 + (q2 - target)**2).mean()
        return q_loss, {
            'q_loss': q_loss
        }
        
    def actor_loss_fn(actor_net, v_learner, q_learner):
        v = eqx.filter_vmap(v_learner)(batch['observations'])
        q1, q2 = eval_ensemble(q_learner, batch['observations'], batch['actions'])
        q = jnp.minimum(q1, q2)
                
        exp_a = jnp.exp((q - v) * agent.temperature)
        exp_a = jnp.minimum(exp_a, 100.0)
        dist = eqx.filter_vmap(actor_net)(batch['observations'])
        
        log_probs = dist.log_prob(batch['actions'])
        actor_loss = -(exp_a.squeeze() * log_probs).mean()
        
        return actor_loss, {
            'actor_loss': actor_loss
        }    
    
    (val_v, aux_v), v_grads = eqx.filter_value_and_grad(v_loss_fn, has_aux=True)(agent.v_learner.model, agent.q_learner.target_model)
    updated_v_learner = agent.v_learner.apply_updates(v_grads)
    
    (val_q, aux_q), q_grads = eqx.filter_value_and_grad(q_loss_fn, has_aux=True)(agent.q_learner.model, updated_v_learner.model)
    updated_q_learner = agent.q_learner.apply_updates(q_grads).soft_update()
    
    (val_actor, aux_actor), actor_grads = eqx.filter_value_and_grad(actor_loss_fn, has_aux=True)(agent.actor_learner.model, updated_v_learner.model, updated_q_learner.target_model)
    updated_actor_learner = agent.actor_learner.apply_updates(actor_grads)

    rng, new_buffer_key = jax.random.split(buffer_key, 2)
    return dataclasses.replace(agent, v_learner=updated_v_learner, q_learner=updated_q_learner, actor_learner=updated_actor_learner), new_buffer_key, {**aux_v, **aux_q, **aux_actor}

def create_iql_eqx_learner(seed, state_dim, action_dim, lr_schedule, max_timesteps,hidden_dims=(256, 256, 256),
                           temperature=9,
                           actor_lr:float=3e-4, value_lr:float=3e-4, critic_lr:float=3e-4, expectile=0.9):
    
    key = jax.random.PRNGKey(seed=seed)
    key, q_key, val_key_main_model, actor_key = jax.random.split(key, 4)
    
    @eqx.filter_vmap
    def ensemblize(keys):
        return QNet(key=keys, state_dim=state_dim, action_dim=action_dim)
    
    if lr_schedule == "cosine":
        schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_timesteps)
        actor_tx = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn))
        print("Using Cosine scheduler")
    else:
        actor_tx = optax.adam(actor_lr)

    q_learner = TrainTargetState.create(
        model=ensemblize(jax.random.split(q_key, 2)),
        target_model=ensemblize(jax.random.split(q_key, 2)),
        optim=optax.adam(learning_rate=critic_lr)
    )
    v_learner = TrainState.create(
        model=VNet(key=val_key_main_model, state_dim=state_dim, use_icvf=False),
        optim=optax.adam(learning_rate=value_lr)
    )
    
    actor_learner = TrainState.create(
        model=GaussianPolicy(key=actor_key,
                             state_dim=state_dim,
                             action_dim=action_dim,
                             hidden_dims=hidden_dims),
        optim=actor_tx
    )
    
    iql_agent = IQLagent(
        q_learner=q_learner,
        v_learner=v_learner,
        actor_learner=actor_learner,
        expectile=expectile,
        discount=0.99,
        temperature=temperature,
    )
    return iql_agent


@pyrallis.wrap()
def train(config: TrainConfig):
    wandb.init(
        config=asdict(config),
        project=config.project,
        group=config.group,
        name=config.dataset_id,
        save_code=False,
    )
    eval_env = gym.make(config.dataset_id)
    state_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]
    replay_buffer, state_mean, state_std = ReplayBuffer.create_from_d4rl(eval_env, normalize_reward=config.normalize_reward,
                                                                         normalize_state=config.normalize_state)
    eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std)
    
    key = jax.random.PRNGKey(seed=config.seed)
    key, q_key, val_key_main_model, actor_key, buffer_key = jax.random.split(key, 5)
    
    @eqx.filter_vmap
    def ensemblize(keys):
        return QNet(key=keys, state_dim=state_dim, action_dim=action_dim)
    
    if config.actor_schedule == "cosine":
        schedule_fn = optax.cosine_decay_schedule(-config.actor_lr, config.max_timesteps)
        actor_tx = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn))
        print("Using Cosine scheduler")
    else:
        actor_tx = optax.adam(config.actor_lr)

    q_learner = TrainTargetState.create(
        model=ensemblize(jax.random.split(q_key, 2)),
        target_model=ensemblize(jax.random.split(q_key, 2)),
        optim=optax.adam(learning_rate=config.qf_lr)
    )
    v_learner = TrainState.create(
        model=VNet(key=val_key_main_model, state_dim=state_dim, use_icvf=config.use_icvf_pretrain),
        optim=optax.adam(learning_rate=config.vf_lr)
    )
    
    actor_learner = TrainState.create(
        model=GaussianPolicy(key=actor_key,
                             state_dim=state_dim,
                             action_dim=action_dim,
                             hidden_dims=(256, 256, 256)),
        optim=actor_tx
    )
    
    iql_agent = IQLagent(
        q_learner=q_learner,
        v_learner=v_learner,
        actor_learner=actor_learner,
        expectile=config.iql_tau,
        discount=config.discount,
        temperature=config.beta,
    )
    
    for step in tqdm(range(1, config.max_timesteps + 1), smoothing=0.1, desc="Training"):
        batch = replay_buffer.sample_batch(key=buffer_key, batch_size=config.batch_size)
        iql_agent, buffer_key, statistics = update_agent(iql_agent, batch, buffer_key)
        wandb.log(statistics)
        
        if step % config.eval_freq == 0 or step == config.max_timesteps - 1:
            returns, norm_returns = evaluate_d4rl(eval_env, iql_agent, config.n_episodes, seed=config.seed)
            wandb.log({"Returns": returns.mean(), 
                       "Normalized D4RL return": norm_returns.mean()})

def evaluate_d4rl(env: gym.Env, actor: IQLagent, num_episodes: int, seed: int):
    env.seed(seed)
    print("Evaluating Agent")

    key = jax.random.PRNGKey(seed)
    returns = []
    norm_returns = []
    for _ in tqdm(range(num_episodes)):
        obs, done = env.reset(), False
        total_reward = 0.0
        
        while not done:
            key, sample_key = jax.random.split(key, 2)
            action = actor.eval_actor(sample_key, obs)
            obs, reward, done, _ = env.step(jax.device_get(action))
            if reward != 0:
                print("Success!")
            #env.render()
            total_reward += reward
        returns.append(total_reward)
        norm_returns.append(env.get_normalized_score(total_reward) * 100.0)
    return np.asarray(returns), np.asarray(norm_returns)

if __name__ == "__main__":
    train()