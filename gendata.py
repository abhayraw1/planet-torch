import gym
import pdb
import pickle
import argparse

from utils import *
from memory import *
from tqdm import trange

BIT_DEPTH = 5

def rollout(memory, env):
    episode = Episode(memory.device, BIT_DEPTH)
    x = env.reset()
    for _ in range(env.env._max_episode_steps):
        u = env.sample_random_action()
        nx, r, d, _ = env.step(u)
        episode.append(x, u, r, d)
        x = nx
    episode.append_last_obs(x)
    memory.append(episode)


def main(env_name, path, size):
    env = TorchImageEnvWrapper('Pendulum-v0', BIT_DEPTH)
    
    memory = Memory(size, None, 50)
    for _ in trange(size):
        rollout(memory, env)
    env.close()

    
    with open(path, 'wb+') as f:
        memory = pickle.dump(memory, f)
    print('DONE!!!')
    print('Thanks. Now move it to Scratch and ping me!! :P')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visual Dataset Generator')
    parser.add_argument('env', type=str, help='Name of Gym Environment.')
    parser.add_argument(
        '--size', type=int, default=1000, help='Size of experience replay'
    )
    parser.add_argument(
        '--output', type=str, default='memory.pth', help='Name of output file'
    )
    args = parser.parse_args()
    main(args.env, args.output, args.size)
