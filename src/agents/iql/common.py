import collections
import dataclasses
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import equinox as eqx
from fastapi import params
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
import optax

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
PRNGKey = Any
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
        return x


@flax.struct.dataclass
class Model:
    step: int
    apply_fn: nn.Module = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(cls,
               model_def: nn.Module,
               inputs: Sequence[jnp.ndarray],
               tx: Optional[optax.GradientTransformation] = None) -> 'Model':
        variables = model_def.init(*inputs)

        # print(variables.pop('params').keys())
        params = variables.pop('params')

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(step=1,
                   apply_fn=model_def,
                   params=params,
                   tx=tx,
                   opt_state=opt_state)

    def __call__(self, *args, **kwargs):
        return self.apply_fn.apply({'params': self.params}, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.apply_fn.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn) -> Tuple[Any, 'Model']:
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.params)

        updates, new_opt_state = self.tx.update(grads, self.opt_state,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(step=self.step + 1,
                            params=new_params,
                            opt_state=new_opt_state), info

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> 'Model':
        with open(load_path, 'rb') as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)


class ModelEQX(eqx.Module):
    # model: eqx.Module
    optim: optax.GradientTransformation
    optim_state: optax.OptState
    params: PyTree
    static: PyTree

    @classmethod
    def create(cls, *, model, optim, **kwargs):
        optim_state = optim.init(eqx.filter(model, eqx.is_array))

        params, static = eqx.partition(model, eqx.is_inexact_array)

        return cls(optim=optim, optim_state=optim_state, params = params, static = static,
                   **kwargs)

    def __call__(self, *args, **kwargs):
        model = eqx.combine(self.params, self.static)
        return model(*args, **kwargs)
    
    def apply(self, *args, **kwargs):
        if 'params' in kwargs:
            p = kwargs['params']
            kwargs.pop('params')
        else:
            p = args[0]
            args = args[1:]
        model = eqx.combine(p, self.static)
        return model(*args, **kwargs)
    
    @eqx.filter_jit
    def apply_updates(self, grads):
        model = eqx.combine(self.params, self.static)
        updates, new_optim_state = self.optim.update(grads, self.optim_state, model)
        new_model = eqx.apply_updates(model, updates)
        params, static = eqx.partition(new_model, eqx.is_inexact_array)
        return dataclasses.replace(
            self,
            params=params,
            optim_state=new_optim_state
        )

    def apply_gradient(self, loss_fn) -> Tuple['ModelEQX', Any]:
        grad_fn = eqx.filter_grad(loss_fn, has_aux=True)
        info, grads = grad_fn(self.params)

        return self.apply_updates(grads), info