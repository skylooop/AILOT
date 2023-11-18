from jaxrl_m.typing import *

import jax
from jaxrl_m.common import TrainTargetStateEQX, TrainStateEQX
from src.agents.iql_equinox import GaussianPolicy, GaussianIntentPolicy
import equinox as eqx
from src.agents.icvf import update, eval_ensemble
import dataclasses
import optax
from src.special_networks import MonolithicVF_EQX
from tqdm.auto import tqdm

# Optimal Transport Imports
from collections import defaultdict
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from ott.tools import plot, sinkhorn_divergence
from ott.solvers.linear import implicit_differentiation as imp_diff
from tqdm.auto import tqdm

class JointGotilAgent(eqx.Module):
    expert_icvf: TrainTargetStateEQX
    agent_icvf: TrainTargetStateEQX
    
    value_net: TrainTargetStateEQX
    actor_intents_learner: TrainStateEQX
    actor_learner: TrainStateEQX

    config: dict
    
    @eqx.filter_jit
    def sample_intentions(self, observation, seed):
        dist = self.actor_intents_learner.model(observation)
        return dist.sample(seed=seed)
    
    @eqx.filter_jit
    def sample_actions(self, observations: np.array, intents: np.ndarray, seed: PRNGKey) -> jnp.ndarray:
        action = self.actor_learner.model(observations, intents).sample(seed=seed)
        return jnp.clip(action, -1.0, 1.0)
        
    def pretrain_expert(self, pretrain_batch):
        agent, update_info = update(self.expert_icvf, pretrain_batch)
        return dataclasses.replace(self, expert_icvf=agent), update_info
    
    @eqx.filter_jit
    def pretrain_agent(self, pretrain_batch, seed):
        # Sample intents from current obs
        intents = eqx.filter_vmap(self.actor_intents_learner.model)(pretrain_batch['observations']).sample(seed=seed)
        # Update actor
        updated_actor, aux_info = update_actor(self.actor_learner, pretrain_batch, self.value_net, intents)
        # Update ICVF using V(s, z) as advantage
        agent, agent_gotil_info = update(self.agent_icvf, pretrain_batch, intents)
        # Update intents of actor using OT
        expert_intents1, expert_intents2 = eqx.filter_jit(get_expert_intents)(self.expert_icvf.value_learner.model.psi_net, pretrain_batch['icvf_desired_goals'])
        expert_marginals1, expert_marginals2 = eqx.filter_jit(eval_ensemble)(self.expert_icvf.value_learner.model, pretrain_batch['next_observations'], pretrain_batch['icvf_desired_goals'], intents, None)
        agent_updated_v, updated_intent_actor, ot_info = eqx.filter_jit(ot_update)(self.actor_intents_learner, self.value_net, pretrain_batch, expert_marginals1, expert_intents1, key=seed)

        return dataclasses.replace(self, agent_icvf=agent, value_net=agent_updated_v, actor_intents_learner=updated_intent_actor, actor_learner=updated_actor), "hi", intents

@eqx.filter_jit
def update_actor(actor_learner, batch, agent_value, intents):
    def actor_loss(actor, intents):
        v1, v2 = eval_value_ensemble(agent_value.model, batch['observations'], intents)
        nv1, nv2 = eval_value_ensemble(agent_value.model, batch['next_observations'], intents)
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2

        adv = nv - v
        exp_a = jnp.exp(adv * 10.0)
        exp_a = jnp.minimum(exp_a, 100.0).squeeze()
        # Take s, z as input -> action
        dist = eqx.filter_vmap(actor)(batch['observations'], intents)
        log_probs = dist.log_prob(batch['actions']) # here - actions from agent dataset
        actor_loss = -(exp_a * log_probs).mean()

        return actor_loss, {
            'low actor_loss': actor_loss,
            'low adv': abs(adv).mean(),
        }
    
    (loss_actor, aux_actor), actor_grads = eqx.filter_value_and_grad(actor_loss, has_aux=True)(actor_learner.model, intents)
    updated_actor = actor_learner.apply_updates(actor_grads)
    return updated_actor, aux_actor
    
def expectile_loss(adv, diff, expectile=0.85):
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * (diff**2)

@eqx.filter_vmap(in_axes=dict(ensemble=eqx.if_array(0), obs=None))
def get_expert_intents(ensemble, obs):
    return eqx.filter_vmap(ensemble)(obs)

@eqx.filter_vmap(in_axes=dict(ensemble=eqx.if_array(0), obs=None, z=None))
def eval_value_ensemble(ensemble, obs, z):
    return eqx.filter_vmap(ensemble)(obs, z)

def sink_div(combined_agent, states, expert_intents, marginal_expert, key) -> tuple[float, float]:
    agent_value, agent_policy = combined_agent
    intents_dist = eqx.filter_vmap(agent_policy)(states)
    intents, log_prob = intents_dist.sample_and_log_prob(seed=key)
    log_prob = jax.lax.stop_gradient(log_prob)
    geom = pointcloud.PointCloud(intents, expert_intents, epsilon=0.01)
    
    a1, a2 = eval_value_ensemble(agent_value, states, intents).squeeze()
    an = jax.nn.softplus(a1 - jnp.quantile(a1, 0.001)) 
    bn = jax.nn.softplus(marginal_expert - jnp.quantile(marginal_expert, 0.001))
    an = an / an.sum()
    bn = bn / bn.sum()
    ot = sinkhorn_divergence.sinkhorn_divergence(
        geom,
        x=geom.x,
        a=an,
        b=bn,
        y=geom.y,
        static_b=True,
        sinkhorn_kwargs={
            "implicit_diff": imp_diff.ImplicitDiff(),
            "use_danskin": True,
            "threshold": 1e-4,
            "max_iterations": 2000
        },
    )
    return ot.divergence, intents

def ot_update(actor_intents_learner, agent_value, batch, expert_marginals, expert_intents, key):
    def v_loss(agent_policy, agent_value, states) -> float:
        z_dist = eqx.filter_vmap(agent_policy)(states)
        z, _ = z_dist.sample_and_log_prob(seed=key)
        v = eval_value_ensemble(agent_value, states, z).squeeze()
        return -v.mean() * 0.1
    
    cost_fn_vg = eqx.filter_jit(eqx.filter_value_and_grad(sink_div, has_aux=True))
    v_loss_vg = eqx.filter_jit(eqx.filter_value_and_grad(v_loss, has_aux=False))
    
    (cost, pmin), (value_grads, intent_policy_grads) = cost_fn_vg((agent_value.model, actor_intents_learner.model), batch['observations'], expert_intents, expert_marginals, key)
    val_loss, intent_policy_grads_2 = v_loss_vg(actor_intents_learner.model, agent_value.model, batch['observations'])
    intent_policy_grads = jax.tree_map(lambda g1, g2: g1 + g2, intent_policy_grads, intent_policy_grads_2)
    agent_value = agent_value.apply_updates(value_grads).soft_update()
    actor_intents_learner = actor_intents_learner.apply_updates(intent_policy_grads)
    
    z = eqx.filter_vmap(actor_intents_learner.model)(batch['observations']).sample(seed=key)
    a1, a2 = eval_value_ensemble(agent_value.model, batch['observations'], z).squeeze()
    
    an = jax.nn.softplus(a1 - jnp.quantile(a1, 0.001))
    bn = jax.nn.softplus(expert_marginals - jnp.quantile(expert_marginals, 0.001))
    an = an / an.sum()
    bn = bn / bn.sum()

    # geom = pointcloud.PointCloud(z, expert_intents, epsilon=0.001)
    # diff = sinkhorn.Sinkhorn()(linear_problem.LinearProblem(geom, a = an, b = bn)).reg_ot_cost
    ot_info = {"OT diff": cost}
    return agent_value, actor_intents_learner, ot_info
    
def create_eqx_learner(seed: int,
                       expert_icvf,
                       agent_icvf,
                       observations,
                       actions,
                       policy_hidden_dims: list = [256, 256],
                       value_hidden_dims: list = [256, 256, 256],
                       intent_codebook_dim: int = 256,
                       discount: float = 0.99,
                       target_update_rate: float = 0.005,
                       expectile: float = 0.9,
                       no_intent: bool = False,
                       min_q: bool = True,
                       periodic_target_update: bool = False,
                       **kwargs):
    
    rng = jax.random.PRNGKey(seed)
    rng, value_model, actor_learner_key = jax.random.split(rng, 3)
    
    @eqx.filter_vmap
    def ensemblize(keys):
        return MonolithicVF_EQX(key=keys, state_dim=observations.shape[-1], intents_dim=intent_codebook_dim, hidden_dims=value_hidden_dims)
    
    value_net_def = TrainTargetStateEQX.create(model=ensemblize(jax.random.split(value_model, 2)),
                                                                target_model=ensemblize(jax.random.split(value_model, 2)),
                                                                optim=optax.adam(learning_rate=3e-4))
    actor_intents_learner = TrainStateEQX.create(
        model=GaussianIntentPolicy(key=actor_learner_key,
                             hidden_dims=policy_hidden_dims,
                             state_dim=observations.shape[-1],
                             intent_dim=intent_codebook_dim),
        optim=optax.adam(learning_rate=3e-4)
    )
    actor_learner = TrainStateEQX.create(
        model=GaussianPolicy(key=actor_learner_key,
                             hidden_dims=policy_hidden_dims,
                             state_dim=observations.shape[-1],
                             intents_dim=intent_codebook_dim,
                             action_dim=actions.shape[-1]),
        optim=optax.adam(learning_rate=3e-4)
    )
    config = dict(
            discount=discount,
            target_update_rate=target_update_rate,
            expectile=expectile,
            no_intent=no_intent, 
            min_q=min_q,
            periodic_target_update=periodic_target_update,
        )
    return JointGotilAgent(expert_icvf=expert_icvf, agent_icvf=agent_icvf, actor_intents_learner=actor_intents_learner,
                           value_net=value_net_def, config=config,
                           actor_learner=actor_learner)
    
def evaluate_with_trajectories_gotil(env, actor, num_episodes, base_observation, seed):
    seed, rng = jax.random.split(seed, 2)
    renders = [] 
    returns = []
    
    for i in range(num_episodes):
        if 'antmaze' in env.spec.id.lower():
            goal = env.wrapped_env.target_goal
            #goal = env.goal_sampler(np.random)
            obs_goal = base_observation.copy()
            obs_goal[:2] = goal
            
        observation, done = env.reset(), False
        total_reward = 0.0
        while not done:
            rng, sampling_rng = jax.random.split(rng, 2)
            intent = actor.sample_intentions(observation, sampling_rng)
            action = actor.sample_actions(observation, intent, sampling_rng)
            if 'antmaze' in env.spec.id.lower():
                next_observation, r, done, info = env.step(jax.device_get(action))
                cur_frame = env.render("rgb_array").transpose(2, 0, 1).copy()
                renders.append(cur_frame)
                if r != 0:
                    print("Success")
                total_reward += r
        returns.append(total_reward)
    return np.asarray(returns).mean(), renders