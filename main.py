# OS params
import os
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
import warnings
warnings.filterwarnings("ignore")

import hydra
import warnings
import rootutils
import functools

# Configs & Printing
import wandb
from omegaconf import DictConfig
ROOT = rootutils.setup_root(search_from=__file__, indicator=[".git", "pyproject.toml"],
                            pythonpath=True, cwd=True)

# Libs
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from tqdm.auto import tqdm
from jaxrl_m.wandb import setup_wandb

import d4rl
from src.agents import icvf, gotil
from src.agents.gotil import evaluate_with_trajectories_gotil
from src.agents.icvf import update, eval_ensemble_gotil, eval_ensemble_icvf, eval_ensemble_icvf_viz
from src.gc_dataset import GCSDataset
from src.utils import record_video
from src import d4rl_utils, d4rl_ant, ant_diagnostics, viz_utils

from src.generate_antmaze_random import get_dataset, combine_ds
from jaxrl_m.evaluation import supply_rng, evaluate_with_trajectories

# Utilities from root folder
from utils.ds_builder import setup_expert_dataset
from utils.rich_utils import print_config_tree

@eqx.filter_jit
def get_gcvalue(agent, s, z):
    v1, v2 = eval_ensemble_gotil(agent.agent_icvf.value_learner.model, s, z)
    return (v1 + v2) / 2

def get_v(agent, observations):
    intents = eqx.filter_vmap(agent.sample_intentions, in_axes=(0, None))(observations, jax.random.PRNGKey(42))
    return get_gcvalue(agent, observations, intents)

@eqx.filter_vmap(in_axes=dict(ensemble=eqx.if_array(0), s=None), out_axes=0)
def eval_ensemble(ensemble, s):
    return eqx.filter_vmap(ensemble)(s)

@eqx.filter_jit
def get_debug_statistics_icvf(agent, batch):
    def get_info(s, g, z):
        return eval_ensemble_icvf_viz(agent.value_learner.model, s, g, z)
    
    s = batch['observations']
    g = batch['icvf_goals']
    z = eval_ensemble(agent.value_learner.model.psi_net, batch['icvf_desired_goals'])[0]
    g = eval_ensemble(agent.value_learner.model.psi_net, batch['icvf_desired_goals'])[0]
    info_szz = get_info(s, z, z)        
    info_sgz = get_info(s, g, z)

    if 'phi' in info_sgz:
        stats = {
            'phi_norm': jnp.linalg.norm(info_sgz['phi'], axis=-1).mean(),
            'psi_norm': jnp.linalg.norm(info_sgz['psi'], axis=-1).mean(),
        }
    else:
        stats = {}

    stats.update({
        'v_szz': info_szz.mean(),
        'v_sgz': info_sgz.mean()
    })
    return stats

@eqx.filter_jit
def get_traj_v(agent, trajectory, seed):
    def get_v(s, g):
        v1, v2 = eval_ensemble_gotil(agent.agent_icvf.value_learner.model, s[None], g[None])
        return (v1 + v2) / 2
    observations = trajectory['observations']
    intents = eqx.filter_vmap(agent.actor_intents_learner.model)(observations).sample(seed=seed)
    all_values = jax.vmap(jax.vmap(get_v, in_axes=(None, 0)), in_axes=(0, None))(observations, intents)
    return {
        'dist_to_beginning': all_values[:, 0],
        'dist_to_end': all_values[:, -1],
        'dist_to_middle': all_values[:, all_values.shape[1] // 2],
    }
    
@eqx.filter_jit
def get_traj_v_icvf(agent, trajectory):
    def get_v(s, g):
        return eval_ensemble_icvf_viz(agent.expert_icvf.value_learner.model, s[None], g[None], g[None]).mean()
    
    observations = trajectory['observations']
    obs_intents = eval_ensemble(agent.expert_icvf.value_learner.model.psi_net, observations)[0]
    all_values = jax.vmap(jax.vmap(get_v, in_axes=(None, 0)), in_axes=(0, None))(observations, obs_intents)
    return {
        'dist_to_beginning': all_values[:, 0],
        'dist_to_end': all_values[:, -1],
        'dist_to_middle': all_values[:, all_values.shape[1] // 2],
    }

@hydra.main(version_base="1.4", config_path=str(ROOT/"configs"), config_name="entry.yaml")
def main(config: DictConfig):
    print_config_tree(config)
    setup_wandb(hyperparam_dict=dict(config),
                project=config.logger.project,
                group=config.logger.group,
                name=None)
    
    env, expert_dataset = setup_expert_dataset(config)
    env.reset()
    
    rng = jax.random.PRNGKey(config.seed)
    gc_dataset = GCSDataset(expert_dataset, **dict(config.GoalDS),
                            discount=config.Env.discount)
    
    if 'antmaze' in env.spec.id.lower():
        example_trajectory = gc_dataset.sample(190, indx=np.arange(10000, 10190))
            
    example_batch = expert_dataset.sample(1)
    
    print("Loading Agent dummy dataset")
    agent_dataset = get_dataset(config.algo.path_to_agent_data)
    mixed_ds = d4rl_utils.get_dataset(env, mixed_ds=False)
    #mixed_ds = combine_ds(expert_dataset, agent_dataset)
    #mixed_ds = d4rl_utils.get_dataset(env, dataset=mixed_ds, mixed_ds=True, normalize_states=False, normalize_rewards=False)
    agent_gc_dataset = GCSDataset(mixed_ds, **dict(config.GoalDS), discount=config.Env.discount)
        
    expert_icvf = icvf.create_eqx_learner(config.seed,
                                    observations=example_batch['observations'],
                                    discount=config.Env.discount,
                                    load_pretrained_icvf=True,
                                    **dict(config.algo))
    
    agent_icvf = icvf.create_eqx_learner(config.seed,
                                    observations=example_batch['observations'],
                                    discount=config.Env.discount,
                                    load_pretrained_icvf=True,
                                    **dict(config.algo))
    
    agent = gotil.create_eqx_learner(config.seed,
                                    expert_icvf=expert_icvf,
                                    agent_icvf=agent_icvf,
                                    batch_size=config.batch_size,
                                    observations=example_batch['observations'],
                                    actions=example_batch['actions'],
                                    **dict(config.algo))
    
    train_metrics = {}
    viz_ant = d4rl_ant.GoalReachingAnt(config.Env.dataset_id.lower())
    sample_traj_img = d4rl_ant.trajectory_image(viz_ant, [example_trajectory])

    train_metrics['sample_traj'] = wandb.Image(sample_traj_img)
    wandb.log(train_metrics)
    
    for i in tqdm(range(1, config.pretrain_steps + 1), smoothing=0.1, dynamic_ncols=True):
        rng, rng1 = jax.random.split(rng, 2)
        pretrain_batch = gc_dataset.sample(config.batch_size) # not needed if expert is pretrained
        
        agent_dataset_batch = agent_gc_dataset.sample(config.batch_size)
        agent, update_info = agent.pretrain_agent(agent_dataset_batch, rng)
        debug_statistics = get_debug_statistics_icvf(agent.agent_icvf, pretrain_batch)
        
        if i % config.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            train_metrics.update({f'pretraining/debug/{k}': v for k, v in debug_statistics.items()})
                    
            traj_metrics = get_traj_v(agent, example_trajectory, seed=rng)
            value_viz = viz_utils.make_visual_no_image(
                traj_metrics,
                [functools.partial(viz_utils.visualize_metric, metric_name=k) for k in traj_metrics.keys()]
            )
            train_metrics['value_traj_viz'] = wandb.Image(value_viz)
            
            traj_metrics = get_traj_v_icvf(agent, example_trajectory)
            value_viz = viz_utils.make_visual_no_image(
                traj_metrics,
                [functools.partial(viz_utils.visualize_metric, metric_name=k) for k in traj_metrics.keys()]
            )
            vzz_image = d4rl_ant.value_image(viz_ant, mixed_ds, functools.partial(get_v, agent=agent))
            train_metrics['value_traj_viz_icvf'] = wandb.Image(value_viz)
            train_metrics['Vzz Heatmap'] = wandb.Image(vzz_image)
            wandb.log(train_metrics, step=i)

        if i % config.eval_interval == 0:
            os.environ['CUDA_VISIBLE_DEVICES']="4"
            base_observation = jax.tree_map(lambda arr: arr[0], gc_dataset.dataset['observations'])
            returns, renders = evaluate_with_trajectories_gotil(env=env, actor=agent, 
                                                        num_episodes=config.eval_episodes, num_video_episodes=config.num_video_episodes, base_observation=base_observation,
                                                        seed=rng)
            if config.num_video_episodes > 0:
                    video = record_video('Video', i, renders=renders)
                    train_metrics['video'] = video
            os.environ['CUDA_VISIBLE_DEVICES']="0,1,2,3,4"
            
            wandb.log({'Eval Returns': returns}, step=i)
            wandb.log(train_metrics, step=i)
        
        
if __name__ == '__main__':
    main()
