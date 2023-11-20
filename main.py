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
from src.agents import hiql, icvf, gotil
from src.agents.gotil import evaluate_with_trajectories_gotil
from src.agents.icvf import update, eval_ensemble_gotil, eval_ensemble_icvf
from src.gc_dataset import GCSDataset
from src.utils import record_video
from src import d4rl_utils, d4rl_ant, ant_diagnostics, viz_utils

from src.generate_antmaze_random import get_dataset, combine_ds
from jaxrl_m.evaluation import supply_rng, evaluate_with_trajectories

# Utilities from root folder
from utils.ds_builder import setup_expert_dataset
from utils.rich_utils import print_config_tree

@jax.jit
def get_gcvalue(agent, s, g):
    v1, v2 = agent.network(s, g, method='value')
    return (v1 + v2) / 2

def get_v(agent, goal, observations):
    goal = jnp.tile(goal, (observations.shape[0], 1))
    return get_gcvalue(agent, observations, goal)

@eqx.filter_jit
def get_debug_statistics_icvf(agent, batch, intents=None):
    def get_info(s, g, z):
        return eval_ensemble_icvf(agent.value_learner.model, s, g, z)
    
    s = batch['observations']
    g = batch['icvf_goals']
    if intents is not None:
        z = intents
        info_szz = None
        info_szg = None
        info_sgg = None
    else:
        z = batch['icvf_desired_goals']
        info_szz = get_info(s, z, z)
        info_szg = get_info(s, z, g)
        info_sgg = get_info(s, g, g)
        
    info_ssz = get_info(s, s, z)
    info_sgz = get_info(s, g, z)

    if 'phi' in info_sgz:
        stats = {
            'phi_norm': jnp.linalg.norm(info_sgz['phi'], axis=-1).mean(),
            'psi_norm': jnp.linalg.norm(info_sgz['psi'], axis=-1).mean(),
        }
    else:
        stats = {}

    stats.update({
        'v_ssz': info_ssz.mean(),
        'v_sgz': info_sgz.mean(),
        #'diff_szz_szg': (info_szz - info_szg).mean(),
        #'diff_sgg_sgz': (info_sgg - info_sgz).mean(),
        # 'v_ssz': info_ssz['v'].mean(),
        # 'v_szz': info_szz['v'].mean(),
        # 'v_sgz': info_sgz['v'].mean(),
        # 'v_sgg': info_sgg['v'].mean(),
        # 'v_szg': info_szg['v'].mean(),
        # 'diff_szz_szg': (info_szz['v'] - info_szg['v']).mean(),
        # 'diff_sgg_sgz': (info_sgg['v'] - info_sgz['v']).mean(),
    })
    if intents is None:
        stats.update({'v_szz': info_szz.mean(),
                      'v_szg': info_szg.mean(),
                      'v_sgg': info_sgg.mean()})
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

@hydra.main(version_base="1.4", config_path=str(ROOT/"configs"), config_name="entry.yaml")
def main(config: DictConfig):
    print_config_tree(config)
    setup_wandb(hyperparam_dict=dict(config),
                project=config.logger.project,
                group=config.logger.group,
                name=None)
    print("Loading Expert data")
    env, expert_dataset = setup_expert_dataset(config)
    env.reset()
    
    rng = jax.random.PRNGKey(config.seed)
    # ICVF is learned on GCRL dataaset
    if config.GoalDS:
        gc_dataset = GCSDataset(expert_dataset, **dict(config.GoalDS),
                                discount=config.Env.discount)
        if 'antmaze' in env.spec.id.lower():
            # take one expert trajectory (where goal is reached)
            example_trajectory = gc_dataset.sample(100, indx=np.arange(700, 800))
            
    total_steps = config.pretrain_steps 
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
                                    load_pretrained_phi=True, # load pretrained ICVF
                                    **dict(config.algo))
    
    agent_icvf = icvf.create_eqx_learner(config.seed,
                                    observations=example_batch['observations'],
                                    discount=config.Env.discount,
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
    
    for i in tqdm(range(1, total_steps + 1), smoothing=0.1, dynamic_ncols=True, desc="Training"):
        rng, rng1 = jax.random.split(rng, 2)
        pretrain_batch = gc_dataset.sample(config.batch_size) # not needed if expert is pretrained
        
        agent_dataset_batch = agent_gc_dataset.sample(config.batch_size)
        agent, update_info, intents = agent.pretrain_agent(agent_dataset_batch, rng)
        debug_statistics = get_debug_statistics_icvf(agent.agent_icvf, pretrain_batch, intents)
        
        if i % config.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            train_metrics.update({f'pretraining/debug/{k}': v for k, v in debug_statistics.items()})
                    
            traj_metrics = get_traj_v(agent, example_trajectory, seed=rng)
            value_viz = viz_utils.make_visual_no_image(
                traj_metrics,
                [functools.partial(viz_utils.visualize_metric, metric_name=k) for k in traj_metrics.keys()]
            )
            train_metrics['value_traj_viz'] = wandb.Image(value_viz)
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
