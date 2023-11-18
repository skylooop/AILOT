from jaxrl_m.dataset import Dataset
from jaxrl_m.typing import *
from jaxrl_m.networks import *
import jax

import functools
import equinox as eqx
import equinox.nn as eqxnn

class LayerNormMLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    activate_final: int = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:                
                x = self.activations(x)
                x = nn.LayerNorm()(x)
        return x


class LayerNormRepresentation(nn.Module):
    hidden_dims: tuple = (256, 256)
    activate_final: bool = True
    ensemble: bool = True

    @nn.compact
    def __call__(self, observations):
        module = LayerNormMLP
        if self.ensemble:
            module = ensemblize(module, 2)
        return module(self.hidden_dims, activate_final=self.activate_final)(observations)


class Representation(nn.Module):
    hidden_dims: tuple = (256, 256)
    activate_final: bool = True
    ensemble: bool = True

    @nn.compact
    def __call__(self, observations):
        module = MLP
        if self.ensemble:
            module = ensemblize(module, 2)
        return module(self.hidden_dims, activate_final=self.activate_final, activations=nn.gelu)(observations)


class RelativeRepresentation(nn.Module):
    rep_dim: int = 256
    hidden_dims: tuple = (256, 256)
    module: nn.Module = None
    visual: bool = False
    layer_norm: bool = False
    rep_type: str = 'state'
    bottleneck: bool = True  # Meaning that we're using this representation for high-level actions

    @nn.compact
    def __call__(self, targets, bases=None):
        if bases is None:
            inputs = targets
        else:
            if self.rep_type == 'state':
                inputs = targets
            elif self.rep_type == 'diff':
                inputs = jax.tree_map(lambda t, b: t - b + jnp.ones_like(t) * 1e-6, targets, bases)
            elif self.rep_type == 'concat':
                inputs = jax.tree_map(lambda t, b: jnp.concatenate([t, b], axis=-1), targets, bases)
            else:
                raise NotImplementedError

        if self.visual:
            inputs = self.module()(inputs)
        if self.layer_norm:
            rep = LayerNormMLP(self.hidden_dims, activate_final=not self.bottleneck, activations=nn.gelu)(inputs)
        else:
            rep = MLP(self.hidden_dims, activate_final=not self.bottleneck, activations=nn.gelu)(inputs)

        if self.bottleneck:
            rep = rep / jnp.linalg.norm(rep, axis=-1, keepdims=True) * jnp.sqrt(self.rep_dim)

        return rep


class MonolithicVF(nn.Module):
    hidden_dims: tuple = (256, 256)
    readout_size: tuple = (256,)
    use_layer_norm: bool = True
    rep_dim: int = None
    obs_rep: int = 0

    def setup(self) -> None:
        repr_class = LayerNormRepresentation if self.use_layer_norm else Representation
        self.value_net = repr_class((*self.hidden_dims, 1), activate_final=False)

    def __call__(self, observations, goals=None, info=False):
        phi = observations
        psi = goals

        v1, v2 = self.value_net(jnp.concatenate([phi, psi], axis=-1)).squeeze(-1)

        if info:
            return {
                'v': (v1 + v2) / 2,
            }
        return v1, v2

class MonolithicVF_EQX(eqx.Module):
    net: eqx.Module
    
    def __init__(self, key, state_dim, intents_dim, hidden_dims):
        key, mlp_key = jax.random.split(key, 2)
        self.net = eqxnn.MLP(
            in_size=state_dim + intents_dim, out_size=1, width_size=hidden_dims[-1], depth=len(hidden_dims), key=mlp_key, activation=jax.nn.gelu
        )
        
    def __call__(self, observations, intents):
        # TODO: Maybe try FiLM conditioning like in SAC-RND?
        conditioning = jnp.concatenate((observations, intents), axis=-1)
        return self.net(conditioning)
        
class MultilinearVF_EQX(eqx.Module):
    phi_net: eqx.Module
    psi_net: eqx.Module
    T_net: eqx.Module
    matrix_a: eqx.Module
    matrix_b: eqx.Module
    gotil_psi: Any = None
    
    def __init__(self, key, state_dim, hidden_dims, pretrained_phi=None):
        key, phi_key, psi_key, t_key, matrix_a_key, matrix_b_key, gotil_mlp_key = jax.random.split(key, 7)
        
        network_cls = functools.partial(eqxnn.MLP, in_size=state_dim, out_size=hidden_dims[-1],
                                        width_size=hidden_dims[0], depth=len(hidden_dims),
                                        final_activation=jax.nn.relu)
        self.gotil_psi = eqxnn.MLP(
            in_size=hidden_dims[-1], out_size=hidden_dims[-1], width_size=hidden_dims[-1], depth=len(hidden_dims), key=gotil_mlp_key, final_activation=jax.nn.tanh
        )
            
        if pretrained_phi is None:
            self.phi_net = network_cls(key=phi_key)
        else:
            self.phi_net = pretrained_phi
            
        self.psi_net = network_cls(key=psi_key)
        self.T_net = eqxnn.MLP(in_size=hidden_dims[-1], out_size=hidden_dims[-1], width_size=hidden_dims[0], depth=len(hidden_dims),
                                        final_activation=jax.nn.relu, key=t_key)
        self.matrix_a = eqxnn.Linear(in_features=hidden_dims[-1], out_features=hidden_dims[-1], key=matrix_a_key)
        self.matrix_b = eqxnn.Linear(in_features=hidden_dims[-1], out_features=hidden_dims[-1], key=matrix_b_key)
        
    def __call__(self, observations, outcomes, intents, mode):
        if mode is None:
            phi = self.phi_net(observations)
            psi = self.psi_net(outcomes)
            z = intents
            Tz = self.T_net(z)
            
            phi_z = self.matrix_a(Tz * phi)
            psi_z = self.matrix_b(Tz * psi)
            v = (phi_z * psi_z).sum(axis=-1)
        else:
            phi = jax.lax.stop_gradient(self.phi_net(observations))
            Tz = jax.lax.stop_gradient(self.T_net(intents))
            phi_z = self.matrix_a(Tz * phi)
            psi = self.gotil_psi(phi_z)
            v = (phi_z).sum(axis=-1)
        return v

def get_rep(
        encoder: nn.Module, targets: jnp.ndarray, bases: jnp.ndarray = None,
):
    if encoder is None:
        return targets
    else:
        if bases is None:
            return encoder(targets)
        else:
            return encoder(targets, bases)


class HierarchicalActorCritic(nn.Module):
    encoders: Dict[str, nn.Module]
    networks: Dict[str, nn.Module]
    use_waypoints: int

    def value(self, observations, goals, **kwargs):
        state_reps = get_rep(self.encoders['value_state'], targets=observations)
        goal_reps = get_rep(self.encoders['value_goal'], targets=goals, bases=observations)
        return self.networks['value'](state_reps, goal_reps, **kwargs)

    def target_value(self, observations, goals, **kwargs):
        state_reps = get_rep(self.encoders['value_state'], targets=observations)
        goal_reps = get_rep(self.encoders['value_goal'], targets=goals, bases=observations)
        return self.networks['target_value'](state_reps, goal_reps, **kwargs)

    def actor(self, observations, goals, low_dim_goals=False, state_rep_grad=True, goal_rep_grad=True, **kwargs):
        state_reps = get_rep(self.encoders['policy_state'], targets=observations)
        if not state_rep_grad:
            state_reps = jax.lax.stop_gradient(state_reps)

        if low_dim_goals:
            goal_reps = goals
        else:
            if self.use_waypoints:
                # Use the value_goal representation
                goal_reps = get_rep(self.encoders['value_goal'], targets=goals, bases=observations)
            else:
                goal_reps = get_rep(self.encoders['policy_goal'], targets=goals, bases=observations)
            if not goal_rep_grad:
                goal_reps = jax.lax.stop_gradient(goal_reps)

        return self.networks['actor'](jnp.concatenate([state_reps, goal_reps], axis=-1), **kwargs)

    def high_actor(self, observations, goals, state_rep_grad=True, goal_rep_grad=True, **kwargs):
        state_reps = get_rep(self.encoders['high_policy_state'], targets=observations)
        if not state_rep_grad:
            state_reps = jax.lax.stop_gradient(state_reps)

        goal_reps = get_rep(self.encoders['high_policy_goal'], targets=goals, bases=observations)
        if not goal_rep_grad:
            goal_reps = jax.lax.stop_gradient(goal_reps)

        return self.networks['high_actor'](jnp.concatenate([state_reps, goal_reps], axis=-1), **kwargs)

    def value_goal_encoder(self, targets, bases, **kwargs):
        return get_rep(self.encoders['value_goal'], targets=targets, bases=bases)

    def policy_goal_encoder(self, targets, bases, **kwargs):
        assert not self.use_waypoints
        return get_rep(self.encoders['policy_goal'], targets=targets, bases=bases)

    def __call__(self, observations, goals):
        # Only for initialization
        rets = {
            'value': self.value(observations, goals),
            'target_value': self.target_value(observations, goals),
            'actor': self.actor(observations, goals),
            'high_actor': self.high_actor(observations, goals),
        }
        return rets
