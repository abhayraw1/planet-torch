import gym
import pdb
import tqdm
import pickle

from utils import *
from memory import *
from models.dssm import *
from models.models import *

from torch.distributions import kl, Normal

BIT_DEPTH = 5
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


def train(memory, model, i):
    for _ in tqdm.tqdm(range(10)):
        (x, u, _, _), lens = memory.sample(32)
        kl_loss, rec_loss = model.train_on_batch(u, x, lens)
    print(f'Loss @ Episode [{i+1}]: KL [{kl_loss}] REC [{rec_loss}]')


def main():
    # env = TorchImageEnvWrapper('Pendulum-v0', BIT_DEPTH)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    detssm = DeterministicStateSpaceModel(
        1, STATE_SIZE, LATENT_SIZE, EMBEDDING_SIZE
    ).to(device)
    # memory = Memory(100, device, 50)
    
    with open('memory.pth', 'rb') as f:
        memory = pickle.load(f)
        memory.device = device
        for e in memory.data:
            e.device = device

    for i in tqdm.tqdm(range(1000)):
        ready_to_train = memory.size > 50
        # rollout(memory, env)
        if ready_to_train:
            train(memory, detssm, i)
            if (i + 1) % 25 == 0:
                (x, u, _, _), lens = memory.sample(1)
                detssm.evaluate(u, x, i+1, 'results/eps_')
            
    # env.close()
    pdb.set_trace()


if __name__ == '__main__':
    main()