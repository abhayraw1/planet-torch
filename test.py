import gym
import pdb
import tqdm

from utils import *
from models import *
from memory import *


BIT_DEPTH = 4
STATE_SIZE = 200
EMBEDDING_SIZE = 1024


def rollout(memory, env):
    episode = Episode(memory.device, BIT_DEPTH)
    x = env.reset()
    for _ in range(env.env._max_episode_steps//4):
        u = env.sample_random_action()
        nx, r, d, _ = env.step(u)
        episode.append(x, u, r, d)
        x = nx
    episode.append_last_obs(x)
    memory.append(episode)


def main():
    env = TorchImageEnvWrapper('Pendulum-v0', BIT_DEPTH)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(EMBEDDING_SIZE).to(device)
    decoder = Decoder(STATE_SIZE, EMBEDDING_SIZE).to(device)
    prior = PriorModel(EMBEDDING_SIZE, STATE_SIZE).to(device)
    posterior = PosteriorModel(STATE_SIZE, EMBEDDING_SIZE).to(device)
    transition = TransitionModel(STATE_SIZE, env.action_size).to(device)
    params = get_combined_params(encoder, decoder, prior, posterior)
    optimizer = torch.optim.Adam(params, lr=1e-3)
    memory = Memory(100, device, 50)

    for i in range(2000):
        ready_to_train = memory.size > 5
        rollout(memory, env)
        if ready_to_train:
            x, u, r, d = memory.sample(16)
            # Every thing is TIME FIRST
            e0 = encoder(x[0])
            h0 = prior(e0)
            ## After this???
            pdb.set_trace()
            ht = transition(h0.sample(), u, unroll_dim=0)
            # the posterior dist on the first state.
            s0 = posterior(prior().rsample(), e[0:1]).rsample()
            # Prior transitions on the rest of the states. 
            pr = prior(s0, u)
            ps = posterior(pr.rsample(), e[1:])
            d1 = decoder(pr.mean)
            d2 = decoder(ps.mean)
            loss = torch.distributions.kl.kl_divergence(ps, pr).sum(dim=-1)
            # Is this to stop gradients when KL < 3?
            loss = torch.max(torch.tensor(3.).to(device), loss).mean()
            loss = loss + ((x[1:] - d2).abs()).sum((1, 2, 3)).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1e3, norm_type=2)
            optimizer.step()
            print(f'Loss @ Episode [{i+1}]: {loss.item()}')
            # pdb.set_trace()
            if (i + 1) % 50 == 0:   
                save_frames(x[1::8], d1[::8], d2[::8], f'results_eps_{i+1}', 5)
    env.close()
    pdb.set_trace()


if __name__ == '__main__':
    main()