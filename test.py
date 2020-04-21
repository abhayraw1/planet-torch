import gym
import pdb
import tqdm

from utils import *
from memory import *
from models.dssm import *
from models.models import *

from torch.distributions import kl, Normal

BIT_DEPTH = 4
STATE_SIZE = 200
LATENT_SIZE = 30
EMBEDDING_SIZE = 1024


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


def main():
    env = TorchImageEnvWrapper('Pendulum-v0', BIT_DEPTH)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    detssm = DeterministicStateSpaceModel(
        env.action_size, STATE_SIZE, LATENT_SIZE, EMBEDDING_SIZE
    ).to(device)
    memory = Memory(100, device, 50)

    for i in range(2000):
        ready_to_train = memory.size > 5
        rollout(memory, env)
        if ready_to_train:
            (x, u, _, _), lens = memory.sample(64)
            kl_loss, rec_loss = detssm.train_on_batch(u, x, lens)
            print(f'Loss @ Episode [{i+1}]: KL [{kl_loss}] REC [{rec_loss}]')
            if (i + 1) % 25 == 0:
                (x, u, _, _), lens = memory.sample(1)
                detssm.evaluate(u, x, i+1, 'results/eps_')
    env.close()
    pdb.set_trace()


if __name__ == '__main__':
    main()