from src import d4rl_ant
import equinox as eqx
import wandb
import numpy as np
from functools import partial
import jax.numpy as jnp

import matplotlib
matplotlib.use('Agg')

import jax
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from src.agents.icvf import eval_ensemble
import matplotlib.gridspec as gridspec
import math

class DebugPlotGenerator:
    def __init__(self, env_name, gc_dataset):
        if 'antmaze' in env_name:
            viz_env, viz_dataset = d4rl_ant.get_env_and_dataset(env_name)
            init_state = np.copy(viz_dataset['observations'][0])
            init_state[:2] = (12.5, 8)
            viz_library = d4rl_ant
            self.viz_things = (viz_env, viz_dataset, viz_library, init_state)
            self.env_name = env_name

        intent_set_indx = np.array([184588, 62200, 162996, 110214, 4086, 191369, 92549, 12946, 192021])
        self.intent_set_batch = gc_dataset.sample(9, indx=intent_set_indx)
        self.example_trajectory = gc_dataset.sample(50, indx=np.arange(1000, 1050))

    def generate_debug_plots(self, agent):
        example_trajectory = self.example_trajectory
        intents = self.intent_set_batch['observations']
        (viz_env, viz_dataset, viz_library, init_state) = self.viz_things

        visualizations = {}
        traj_metrics = get_traj_v_icvf(agent, example_trajectory)
        value_viz = make_visual_no_image(traj_metrics, 
            [
            partial(visualize_metric, metric_name=k) for k in traj_metrics.keys()
                ]
        )
        visualizations['value_traj_viz'] = wandb.Image(value_viz)
    
        if 'maze' in self.env_name:
            print('Visualizing intent policies and values')
            methods = [
                partial(viz_library.plot_policy, policy_fn=partial(get_policy, agent, intent=intents[idx]))
                for idx in range(9)
            ]
            image = viz_library.make_visual(viz_env, viz_dataset, methods)
            visualizations['intent_policies'] = wandb.Image(image)

            # Value visualization
            methods = [
                partial(viz_library.plot_value, value_fn=partial(get_values, agent, intent=intents[idx]))
                for idx in range(9)
            ]
            image = viz_library.make_visual(viz_env, viz_dataset, methods)
            visualizations['intent_values'] = wandb.Image(image)

            for idx in range(3):
                methods = [
                    partial(viz_library.plot_policy, policy_fn=partial(get_policy, agent, intent=intents[idx])),
                    partial(viz_library.plot_value, value_fn=partial(get_values, agent, intent=intents[idx]))
                ]
                image = viz_library.make_visual(viz_env, viz_dataset, methods)
                visualizations[f'intent{idx}'] = wandb.Image(image)

            image_zz = viz_library.gcvalue_image(
                viz_env,
                viz_dataset,
                partial(get_v_zz, agent),
            )
            image_gz = viz_library.gcvalue_image(
                viz_env,
                viz_dataset,
                partial(get_v_gz, agent, init_state),
            )
            visualizations['v_zz'] = wandb.Image(image_zz)
            visualizations['v_gz'] = wandb.Image(image_gz)
        return visualizations

@eqx.filter_jit
def get_values(agent, observations, intent):
    def get_v(observations, intent):
        intent = intent.reshape(1, -1)
        intent_tiled = jnp.tile(intent, (observations.shape[0], 1))
        v1, v2 = eval_ensemble(agent.value_learner.model, observations, intent_tiled, intent_tiled, None)
        return (v1 + v2) / 2    
    return get_v(observations, intent)

def most_squarelike(n):
    c = int(n ** 0.5)
    while c > 0:
        if n %c in [0 , c-1]:
            return (c, int(math.ceil(n / c)))
        c -= 1

def make_visual(images, metrics, visualization_methods=[]):
    
    w, h = most_squarelike(len(visualization_methods))
    gs = gridspec.GridSpec(h + 1, w)

    fig = plt.figure(tight_layout=True)
    canvas = FigureCanvas(fig)
    
    ax = fig.add_subplot(gs[0, :])
    view_images(ax, images, n_images=w * 4)

    for i in range(len(visualization_methods)):
        wi, hi = i % w, i // w
        ax = fig.add_subplot(gs[hi + 1, wi])
        visualization_methods[i](ax=ax, metrics=metrics)

    plt.tight_layout()
    canvas.draw() 
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return out_image

@eqx.filter_jit
def get_traj_v_icvf(agent, trajectory):
    def get_v(s, g):
        return eval_ensemble(agent.value_learner.model, s[None], g[None], g[None], None).mean()
    observations = trajectory['observations']
    all_values = jax.vmap(jax.vmap(get_v, in_axes=(None, 0)), in_axes=(0, None))(observations, observations)
    return {
        'dist_to_beginning': all_values[:, 0],
        'dist_to_end': all_values[:, -1],
        'dist_to_middle': all_values[:, all_values.shape[1] // 2],
    }

@eqx.filter_jit
def get_gcvalue(agent, s, g, z):
    v_sgz_1, v_sgz_2 = eval_ensemble(agent.value_learner.model, s, g, z)
    return (v_sgz_1 + v_sgz_2) / 2

def get_v_zz(agent, goal, observations):
    goal = jnp.tile(goal, (observations.shape[0], 1))
    return get_gcvalue_icvf(agent, observations, goal, goal)

def get_v_gz(agent, initial_state, target_goal, observations):
    initial_state = jnp.tile(initial_state, (observations.shape[0], 1))
    target_goal = jnp.tile(target_goal, (observations.shape[0], 1))
    return get_gcvalue_icvf(agent, initial_state, observations, target_goal)

@eqx.filter_jit
def get_gcvalue_icvf(agent, s, g, z):
    v_sgz_1, v_sgz_2 = eval_ensemble(agent.value_learner.model, s, g, z, None)
    return (v_sgz_1 + v_sgz_2) / 2

@eqx.filter_jit
def get_policy(agent, observations, intent):
    def v(observations):
        def get_v(observations, intent):
            intent = intent.reshape(1, -1)
            intent_tiled = jnp.tile(intent, (observations.shape[0], 1))
            v1, v2 = eval_ensemble(agent.value_learner.model, observations, intent_tiled, intent_tiled, None)
            return (v1 + v2) / 2    
            
        return get_v(observations, intent).mean()

    grads = eqx.filter_grad(v)(observations)
    policy = grads[:, :2]
    return policy / jnp.linalg.norm(policy, axis=-1, keepdims=True)

def make_visual_no_image(metrics, visualization_methods=[]):
    
    w, h = most_squarelike(len(visualization_methods))
    gs = gridspec.GridSpec(h, w)

    fig = plt.figure(tight_layout=True)
    canvas = FigureCanvas(fig)
    
    for i in range(len(visualization_methods)):
        wi, hi = i % w, i // w
        ax = fig.add_subplot(gs[hi, wi])
        visualization_methods[i](ax=ax, metrics=metrics)

    plt.tight_layout()
    canvas.draw() 
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return out_image


def np_unstack(array, axis):
    arr = np.split(array, array.shape[axis], axis)
    arr = [a.squeeze() for a in arr]
    return arr

def view_images(ax, images, n_images=4):
    assert len(images.shape) == 4
    assert images.shape[-1] == 3
    interval = images.shape[0] // n_images
    sel_images = images[::interval]
    sel_images = np.concatenate(np_unstack(sel_images, 0), 1)
    ax.imshow(sel_images)

def visualize_metric(ax, metrics, *, metric_name, linestyle='--', marker='o', **kwargs):
    metric = metrics[metric_name]
    ax.plot(metric, linestyle=linestyle, marker=marker, **kwargs)
    ax.set_ylabel(metric_name)

def visualize_metrics(ax, metrics, *, ylabel=None, metric_names, **kwargs):
    for metric_name in metric_names:
        metric = metrics[metric_name]
        ax.plot(metric, linestyle='--', marker='o', **kwargs)
    ax.set_ylabel(ylabel or metric_names[0])