from jaxrl_m.typing import *
from jaxrl_m.networks import *
import jax

import functools
import equinox as eqx
import equinox.nn as eqxnn

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
    gotil: Any = None
    
    def __init__(self, key, state_dim, hidden_dims, pretrained_phi=None, pretrained_psi=None, pretrained_T=None,pretrained_a=None,pretrained_b=None):
        key, phi_key, psi_key, t_key, matrix_a_key, matrix_b_key, gotil_key = jax.random.split(key, 7)
        
        network_cls = functools.partial(eqxnn.MLP, in_size=state_dim, out_size=hidden_dims[-1],
                                        width_size=hidden_dims[0], depth=len(hidden_dims), final_activation=jax.nn.relu)
        
        self.gotil = eqxnn.Linear(in_features=hidden_dims[0], out_features=1, key=gotil_key)
        
        if pretrained_phi is None:
            self.phi_net = network_cls(key=phi_key)
        else:
            self.phi_net = pretrained_phi
        
        if pretrained_psi is None:
            self.psi_net = network_cls(key=psi_key)
        else:
            self.psi_net = pretrained_psi
        
        T_cls = functools.partial(eqxnn.MLP, in_size=hidden_dims[-1], out_size=hidden_dims[-1], width_size=hidden_dims[0], depth=len(hidden_dims),
                                        final_activation=jax.nn.relu)
        network_cls_a = functools.partial(eqxnn.Linear, in_features=hidden_dims[-1], out_features=hidden_dims[-1])
        network_cls_b = functools.partial(eqxnn.Linear, in_features=hidden_dims[-1], out_features=hidden_dims[-1])
        
        if pretrained_T is None:
            self.T_net = T_cls(key=t_key)
            self.matrix_a = network_cls_a(key=matrix_a_key)
            self.matrix_b = network_cls_b(key=matrix_b_key)
        else:
            self.T_net = pretrained_T
            self.matrix_a = pretrained_a
            self.matrix_b = pretrained_b
        
    def classic_icvf(self, observations, outcomes, intents):
        phi = self.phi_net(observations)
        psi = self.psi_net(outcomes)
        z = intents
        Tz = self.T_net(z)
        
        phi_z = self.matrix_a(Tz * phi)
        psi_z = self.matrix_b(Tz * psi)
        v = (phi_z * psi_z).sum(axis=-1)
        return v
    
    def icvf_zz(self, observations, outcomes, intents):
        phi = self.phi_net(observations)
        psi = outcomes
        z = intents
        Tz = self.T_net(z)
        
        phi_z = self.matrix_a(Tz * phi)
        psi_z = self.matrix_b(Tz * psi)
        v = (phi_z * psi_z).sum(axis=-1)
        return v
    
    def classic_icvf_initial(self, observations, outcomes, intents):
        phi = self.phi_net(observations)
        psi = self.psi_net(outcomes)
        z = self.psi_net(intents)
        Tz = self.T_net(z)
        
        phi_z = self.matrix_a(Tz * phi)
        psi_z = self.matrix_b(Tz * psi)
        v = (phi_z * psi_z).sum(axis=-1)
        return v
    
    def icvf_viz(self, observations, outcomes, intents):
        phi = self.phi_net(observations)
        psi = outcomes
        z = intents
        Tz = self.T_net(z)
        
        phi_z = self.matrix_a(Tz * phi)
        psi_z = self.matrix_b(Tz * psi)
        v = (phi_z * psi_z).sum(axis=-1)
        return v
    
    def gotil_fn(self, observations, intents):
        phi = self.phi_net(observations)
        Tz = self.T_net(intents)
        phi_z = jax.lax.stop_gradient(self.matrix_a(Tz * phi))
        psi_z = jax.lax.stop_gradient(self.matrix_b(Tz))
        v = self.gotil(phi_z * psi_z).squeeze(axis=-1)
        v = jax.nn.softplus(v)
        return v