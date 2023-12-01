import os
from src import d4rl_utils
from jaxrl_m.evaluation import EpisodeMonitor


def setup_expert_dataset(config):
    
    if 'antmaze' in config.Env.dataset_id.lower():
        env_name = config.Env.dataset_id.lower()
        if 'ultra' in env_name:
            import d4rl_ext
            import gym
            env = gym.make(env_name)
        else:
            import d4rl
            import gym
            env = gym.make(env_name)
        env = EpisodeMonitor(env)
        dataset = d4rl_utils.get_dataset(env, normalize_states=config.Env.normalize_states, normalize_rewards=config.Env.normalize_rewards)
        
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

    return env, dataset