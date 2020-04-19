import gym
import pdb
import time
import numpy as np
from payload_manipulation.utils.utils import *
set_seed(41176)
env = gym.make('PayloadEnvG2G-v0')
obs = env.reset(total_chaos=False)
print(np.degrees(env.agent_state[-1]))
# env.agent_state[-1] = -np.pi
env.render()
action = np.array([1, 0, 0, 0, 0])
for i in range(100):
    _, r, d, info = env.step(action)
    env.render()
    print({'r': r, 'd': d, **info})
    time.sleep(0.05)
    pdb.set_trace()
