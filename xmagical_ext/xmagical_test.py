import gym
import xmagical
xmagical.register_envs()
import time
import numpy as np
import pygame

# env = gym.make('SweepToTop-Gripper-State-Allo-TestLayout-v0')
# print(xmagical.ALL_REGISTERED_ENVS)

ds = np.load("/home/simmax21/Desktop/GOTIL/d4rl_ext/xmagical/xmagical_replay/gripper_train.npz")

import pygame
from pygame import gfxdraw

pygame.init()
pygame.display.init()
viewer = pygame.display.set_mode((500, 500))

for i in range(1000):
    surf = pygame.surfarray.make_surface(ds['observations'][i])
    viewer.blit(surf, (0, 0))
    pygame.display.flip()
    pygame.display.update()

# env.reset()
# for i in range(1000):
#     print(ds['rewards'][i])
#     env.step(ds['actions'][i])
#     time.sleep(0.028)
#     env.render(mode='human')