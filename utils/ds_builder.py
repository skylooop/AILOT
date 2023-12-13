import os
from src import d4rl_utils
from jaxrl_m.evaluation import EpisodeMonitor
import jax.numpy as jnp

def setup_datasets(config):
    
    dataset_expert = d4rl_utils.get_dataset(config.Env.expert_env, expert=True, num_episodes=config.Env.expert_episodes)
    dataset_agent = d4rl_utils.get_dataset(config.Env.agent_env, expert=False)
    
    if 'antmaze' in config.Env.expert_env:
        env_name = config.Env.expert_env
        if 'ultra' in env_name:
            import d4rl_ext
            import gym
            env = gym.make(env_name)
        else:
            import d4rl
            import gym
            env = gym.make(env_name)
            
        env = EpisodeMonitor(env)
        os.environ['CUDA_VISIBLE_DEVICES']="4" # for headless server
        env.render(mode='rgb_array', width=200, height=200)
        os.environ['CUDA_VISIBLE_DEVICES']="0,1,2,3,4"
        
        if 'large' in env_name:
            env.viewer.cam.lookat[0] = 18
            env.viewer.cam.lookat[1] = 12
            env.viewer.cam.distance = 50
            env.viewer.cam.elevation = -90

        elif 'ultra' in env_name:
            env.viewer.cam.lookat[0] = 26
            env.viewer.cam.lookat[1] = 18
            env.viewer.cam.distance = 70
            env.viewer.cam.elevation = -90
        else:
            env.viewer.cam.lookat[0] = 18
            env.viewer.cam.lookat[1] = 12
            env.viewer.cam.distance = 50
            env.viewer.cam.elevation = -90
    else:
        env = gym.make(config.Env.agent_env)     
    print(f"expert data: {dataset_expert['observations'].shape[0]}")
    print(f"agent data: {dataset_agent['observations'].shape[0]}")
    
    # if config.Env.normalize_states:
    #     observations_all = jnp.concatenate([replay_buffer_e.state, replay_buffer_o.state]).astype(np.float32)
    #     state_mean = jnp.mean(observations_all, 0)
    #     state_std = jnp.std(observations_all, 0) + 1e-3
        
    return env, dataset_expert, dataset_agent