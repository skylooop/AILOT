from jaxrl_m.typing import *
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import tree_util
import optax
import functools
import equinox as eqx
import dataclasses

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


def shard_batch(batch):
    d = jax.local_device_count()

    def reshape(x):
        assert (
            x.shape[0] % d == 0
        ), f"Batch size needs to be divisible by # devices, got {x.shape[0]} and {d}"
        return x.reshape((d, x.shape[0] // d, *x.shape[1:]))

    return tree_util.tree_map(reshape, batch)


def target_update(
    model: "TrainState", target_model: "TrainState", tau: float
) -> "TrainState":
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), model.params, target_model.params
    )
    return target_model.replace(params=new_target_params)


class TrainState(flax.struct.PyTreeNode):
    """
    Core abstraction of a model in this repository.

    Creation:
    ```
        model_def = nn.Dense(12) # or any other flax.linen Module
        params = model_def.init(jax.random.PRNGKey(0), jnp.ones((1, 4)))['params']
        model = TrainState.create(model_def, params, tx=None) # Optionally, pass in an optax optimizer
    ```

    Usage:
    ```
        y = model(jnp.ones((1, 4))) # By default, uses the `__call__` method of the model_def and params stored in TrainState
        y = model(jnp.ones((1, 4)), params=params) # You can pass in params (useful for gradient computation)
        y = model(jnp.ones((1, 4)), method=method) # You can apply a different method as well
    ```

    More complete example:
    ```
        def loss(params):
            y_pred = model(x, params=params)
            return jnp.mean((y - y_pred) ** 2)

        grads = jax.grad(loss)(model.params)
        new_model = model.apply_gradients(grads=grads) # Alternatively, new_model = model.apply_loss_fn(loss_fn=loss)
    ```
    """

    step: int
    apply_fn: Callable[..., Any] = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Params
    tx: Optional[optax.GradientTransformation] = nonpytree_field()
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(
        cls,
        model_def: nn.Module,
        params: Params,
        tx: Optional[optax.GradientTransformation] = None,
        **kwargs,
    ) -> "TrainState":
        if tx is not None:
            opt_state = tx.init(flax.core.freeze(params))
        else:
            opt_state = None

        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def __call__(
        self,
        *args,
        params=None,
        extra_variables: dict = None,
        method: ModuleMethod = None,
        **kwargs,
    ):
        """
        Internally calls model_def.apply_fn with the following logic:

        Arguments:
            params: If not None, use these params instead of the ones stored in the model.
            extra_variables: Additional variables to pass into apply_fn
            method: If None, use the `__call__` method of the model_def. If a string, uses
                the method of the model_def with that name (e.g. 'encode' -> model_def.encode).
                If a function, uses that function.

        """
        if params is None:
            params = self.params

        variables = {"params": params}

        if extra_variables is not None:
            variables = {**variables, **extra_variables}

        if isinstance(method, str):
            method = getattr(self.model_def, method)

        return self.apply_fn(variables, *args, method=method, **kwargs)

    def apply_gradients(self, *, grads, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

        Note that internally this function calls `.tx.update()` followed by a call
        to `optax.apply_updates()` to update `params` and `opt_state`.

        Args:
            grads: Gradients that have the same pytree structure as `.params`.
            **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

        Returns:
            An updated instance of `self` with `step` incremented by one, `params`
            and `opt_state` updated by applying `grads`, and additional attributes
            replaced as specified by `kwargs`.
        """

        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    def apply_loss_fn(self, *, loss_fn, pmap_axis=None, has_aux=False):
        """
        Takes a gradient step towards minimizing `loss_fn`. Internally, this calls
        `jax.grad` followed by `TrainState.apply_gradients`. If pmap_axis is provided,
        additionally it averages gradients (and info) across devices before performing update.
        """
        if has_aux:
            grads, info = jax.grad(loss_fn, has_aux=has_aux)(self.params)
            if pmap_axis is not None:
                grads = jax.lax.pmean(grads, axis_name=pmap_axis)
                info = jax.lax.pmean(info, axis_name=pmap_axis)

            return self.apply_gradients(grads=grads), info

        else:
            grads = jax.grad(loss_fn, has_aux=has_aux)(self.params)
            if pmap_axis is not None:
                grads = jax.lax.pmean(grads, axis_name=pmap_axis)
            return self.apply_gradients(grads=grads)

class TrainStateEQX(eqx.Module):
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

class TrainTargetStateEQX(TrainStateEQX):
    target_model: eqx.Module

    @classmethod
    def create(cls, *, model, target_model, optim, **kwargs):
        optim_state = optim.init(eqx.filter(model, eqx.is_array))
        return cls(model=model, optim=optim, optim_state=optim_state, target_model=target_model,
                   **kwargs)

    #def update_gotil(self, grads):
        #
    
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