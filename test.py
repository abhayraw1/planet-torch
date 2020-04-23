import gym
import pdb
import tqdm
import pickle

from utils import *
from memory import *
from models.dssm import *
from models.models import *

from torch.distributions import kl, Normal
from torch.utils.tensorboard import SummaryWriter

BIT_DEPTH = 5
STATE_SIZE = 7
LATENT_SIZE = 3
EMBEDDING_SIZE = 11


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

    writer = SummaryWriter('results/test')
    device = torch.device("cpu")
    detssm = DeterministicStateSpaceModel(
        1, STATE_SIZE, LATENT_SIZE, EMBEDDING_SIZE
    ).to(device)
    obs = torch.rand(1, 5, 3, 64, 64)
    actions = torch.rand(1, 4, 1)
    e0 = [detssm.encoder(o) for o in torch.unbind(obs[:, :4], dim=1)]
    c0 = torch.cat(e0, dim=-1)
    s0 = detssm.init_state_model(c0)
    sn = detssm.transition_model(actions, s0, unroll_dim=1)
    pr = detssm.latent_prior(torch.cat([sn, actions], dim=-1))
    writer.add_graph(detssm, [actions, obs])
    # writer.add_graph(detssm.transition_model, [actions, s0])
    # writer.add_graph(detssm.latent_prior, torch.cat([sn, actions], dim=-1))
    pdb.set_trace()


if __name__ == '__main__':
    main()