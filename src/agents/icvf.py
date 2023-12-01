from jaxrl_m.typing import *

import jax
import jax.numpy as jnp
import optax
from jaxrl_m.common import TrainTargetStateEQX

import equinox.nn as nn

import functools
import equinox as eqx
from src.special_networks import MultilinearVF_EQX
import dataclasses

def expectile_loss(adv, diff, expectile=0.8):
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * diff ** 2

def gotil_loss(value_fn, target_value_fn, batch, config, intents):
    # from icvf
    (next_v1_gz, next_v2_gz) = eval_ensemble_icvf(target_value_fn, batch['next_observations'], batch['icvf_desired_goals'], intents)
    q1_gz = batch['icvf_rewards'] + config['discount'] * batch['icvf_masks'] * next_v1_gz
    q2_gz = batch['icvf_rewards'] + config['discount'] * batch['icvf_masks'] * next_v2_gz

    (v1_gz, v2_gz) = eval_ensemble_icvf(value_fn, batch['observations'], batch['icvf_desired_goals'], intents)
    v1_gz, v2_gz = jax.lax.stop_gradient(v1_gz), jax.lax.stop_gradient(v2_gz)
    (next_v1_zz, next_v2_zz) = eval_ensemble_gotil(target_value_fn, batch['next_observations'], intents)
    if config['min_q']:
        next_v_zz = jnp.minimum(next_v1_zz, next_v2_zz)
    else:
        next_v_zz = (next_v1_zz + next_v2_zz) / 2.
        
    q_zz = next_v_zz
    (v1_zz, v2_zz) = eval_ensemble_gotil(target_value_fn, batch['observations'], intents)
    v_zz = (v1_zz + v2_zz) / 2.
    
    adv = q_zz - v_zz
    value_loss1 = expectile_loss(adv, q1_gz-v1_gz, config['expectile']).mean()
    value_loss2 = expectile_loss(adv, q2_gz-v2_gz, config['expectile']).mean()
    value_loss = value_loss1 + value_loss2
    
    advantage = adv
    return value_loss, {
        'gotil_value_loss': value_loss,
        'gotil_abs_adv_mean': jnp.abs(advantage).mean()}
    
class ICVF_EQX_Agent(eqx.Module):
    value_learner: TrainTargetStateEQX
    config: dict
 
@eqx.filter_vmap(in_axes=dict(ensemble=eqx.if_array(0), s=None, g=None, z=None), out_axes=0)
def eval_ensemble_icvf(ensemble, s, g, z):
    return eqx.filter_vmap(ensemble.classic_icvf)(s, g, z)

@eqx.filter_vmap(in_axes=dict(ensemble=eqx.if_array(0), s=None, g=None, z=None), out_axes=0)
def eval_ensemble_icvf_viz(ensemble, s, g, z):
    return eqx.filter_vmap(ensemble.icvf_viz)(s, g, z)

@eqx.filter_vmap(in_axes=dict(ensemble=eqx.if_array(0), s=None, z=None), out_axes=0)
def eval_ensemble_gotil(ensemble, s, z):
    return eqx.filter_vmap(ensemble.gotil)(s, z)

@eqx.filter_jit
def update(agent, batch, intents=None):
    (val, value_aux), v_grads = eqx.filter_value_and_grad(gotil_loss, has_aux=True)(agent.value_learner.model, agent.value_learner.target_model, batch, agent.config, intents)
    
    updated_v_learner = agent.value_learner.apply_updates(v_grads).soft_update()
    return dataclasses.replace(agent, value_learner=updated_v_learner, config=agent.config), value_aux
    
def create_eqx_learner(seed: int,
                       observations: jnp.array,
                       hidden_dims: list,
                       optim_kwargs: dict = {
                            'learning_rate': 0.00005,
                            'eps': 0.0003125
                        },
                        load_pretrained_icvf: bool=False,
                        discount: float = 0.99,
                        target_update_rate: float = 0.005,
                        expectile: float = 0.9,
                        no_intent: bool = False,
                        min_q: bool = True,
                        periodic_target_update: bool = False,
                        **kwargs):
        print('Extra kwargs:', kwargs)
        rng = jax.random.PRNGKey(seed)
        
        if load_pretrained_icvf:
            network_cls_phi = functools.partial(nn.MLP, in_size=observations.shape[-1], out_size=hidden_dims[-1],
                                        width_size=hidden_dims[0], depth=len(hidden_dims),
                                        final_activation=jax.nn.gelu)
            network_cls_psi = functools.partial(nn.MLP, in_size=observations.shape[-1], out_size=hidden_dims[-1],
                                        width_size=hidden_dims[0], depth=len(hidden_dims),
                                        final_activation=jax.nn.gelu)
            network_cls_T = functools.partial(nn.MLP, in_size=hidden_dims[-1], out_size=hidden_dims[-1], width_size=hidden_dims[0], depth=len(hidden_dims),
                                        final_activation=jax.nn.gelu)
            loaded_matrix_a = functools.partial(nn.Linear, in_features=hidden_dims[-1], out_features=hidden_dims[-1])
            loaded_matrix_b = functools.partial(nn.Linear, in_features=hidden_dims[-1], out_features=hidden_dims[-1])
            
            phi_net = network_cls_phi(key=rng)
            psi_net = network_cls_psi(key=rng)
            T_net = network_cls_T(key=rng)
            matrix_a = loaded_matrix_a(key=rng)
            matrix_b = loaded_matrix_b(key=rng)
            
            loaded_phi_net = eqx.tree_deserialise_leaves("../pretrained_icvf/antmaze-large-diverse/icvf_model_phi.eqx", phi_net)
            loaded_psi_net = eqx.tree_deserialise_leaves("../pretrained_icvf/antmaze-large-diverse/icvf_model_psi.eqx", psi_net)
            loaded_T_net = eqx.tree_deserialise_leaves("../pretrained_icvf/antmaze-large-diverse/icvf_model_T.eqx", T_net)
            loaded_matrix_a = eqx.tree_deserialise_leaves("../pretrained_icvf/antmaze-large-diverse/icvf_model_a.eqx", matrix_a)
            loaded_matrix_b = eqx.tree_deserialise_leaves("../pretrained_icvf/antmaze-large-diverse/icvf_model_b.eqx", matrix_b)
        else:
            loaded_phi_net = None
            loaded_psi_net = None
            loaded_T_net = None
            
        @eqx.filter_vmap
        def ensemblize(keys):
            return MultilinearVF_EQX(key=keys, state_dim=observations.shape[-1], hidden_dims=hidden_dims,
                                     pretrained_phi=loaded_phi_net, pretrained_psi=loaded_psi_net, pretrained_T=loaded_T_net,
                                     pretrained_a=loaded_matrix_a, pretrained_b=loaded_matrix_b)
        
        value_learner = TrainTargetStateEQX.create(
            model=ensemblize(jax.random.split(rng, 2)),
            target_model=ensemblize(jax.random.split(rng, 2)),
            optim=optax.adam(**optim_kwargs)
        )
        config = dict(
            discount=discount,
            target_update_rate=target_update_rate,
            expectile=expectile,
            no_intent=no_intent, 
            min_q=min_q,
            periodic_target_update=periodic_target_update,
        )
        return ICVF_EQX_Agent(value_learner=value_learner, config=config)
