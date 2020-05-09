import pdb
import torch
from tqdm import trange
from functools import partial
from collections import defaultdict


from torch.distributions import Normal, kl
from torch.distributions.kl import kl_divergence

from utils import *
from memory import *
from rssm_model import *
from rssm_policy import *
from rollout_generator import RolloutGenerator

def train(memory, rssm, optimizer, device, N=16, H=50, beta=0.7):
    free_nats = torch.ones(1, device=device)*3.0
    batch = memory.sample(N, H, time_first=True)
    x, u, r, t  = [torch.tensor(x).float().to(device) for x in batch]
    preprocess_img(x, depth=5)
    h_t = torch.zeros(N, rssm.state_size).to(device)
    s_t = torch.zeros(N, rssm.latent_size).to(device)
    a_t = torch.zeros(N, rssm.action_size).to(device)
    e_t = bottle(rssm.encoder, x)
    h_t, s_t = rssm.get_init_state(e_t[0], h_t, s_t, a_t)
    kl_loss, rc_loss, re_loss = 0, 0, 0
    states, priors, posteriors, posterior_samples = [], [], [], []
    for i, a_t in enumerate(torch.unbind(u, dim=0)):
        h_t = rssm.deterministic_state_fwd(h_t, s_t, a_t)
        states.append(h_t)
        priors.append(rssm.state_prior(h_t))
        posteriors.append(rssm.state_posterior(h_t, e_t[i + 1]))
        posterior_samples.append(Normal(*posteriors[-1]).rsample())
        s_t = posterior_samples[-1]
    prior_dist = Normal(*map(torch.stack, zip(*priors)))
    posterior_dist = Normal(*map(torch.stack, zip(*posteriors)))
    states, posterior_samples = map(torch.stack, (states, posterior_samples))
    rec_loss = F.mse_loss(
        bottle(rssm.decoder, states, posterior_samples), x[1:],
        reduction='none'
    ).sum((2, 3, 4)).mean()
    kld_loss = torch.max(
        kl_divergence(posterior_dist, prior_dist).sum(-1).mean(),
        free_nats
    )
    rew_loss = F.mse_loss(
        bottle(rssm.pred_reward, states, posterior_samples), r
    )
    optimizer.zero_grad()
    nn.utils.clip_grad_norm_(rssm.parameters(), 100., norm_type=2)
    (beta*kld_loss + rec_loss + rew_loss).backward()
    optimizer.step()
    metrics = {
        'losses': {
            'kl': kld_loss.item(),
            'rc': rec_loss.item(),
            're': rew_loss.item()
        },
        'grad_norms': {
            k: 0 if v.grad is None else v.grad.norm().item()
            for k, v in rssm.named_parameters()
        }
    }
    return metrics


def main():
    env = TorchImageEnvWrapper('Pendulum-v0', bit_depth=5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rssm_model = RecurrentStateSpaceModel(env.action_size).to(device)
    optimizer = torch.optim.Adam(rssm_model.parameters(), lr=1e-3)
    policy = RSSMPolicy(
        rssm_model, 
        planning_horizon=20,
        num_candidates=1000,
        num_iterations=10,
        top_candidates=20,
        device=device
    )
    rollout_gen = RolloutGenerator(
        env,
        device,
        policy=policy,
        episode_gen=lambda : Episode(partial(postprocess_img, depth=5)),
        max_episode_steps=500,
    )
    mem = Memory(100)
    mem.append(rollout_gen.rollout_n(15, random_policy=True))
    metrics = Metrics()
    for i in trange(100, desc='Epoch', leave=False):
        for _ in trange(150, desc='Iter ', leave=False):
            train_metrics = train(mem, rssm_model.train(), optimizer, device)
            metrics.update(train_metrics)
        mem.append(rollout_gen.rollout_once(explore=True))
        eval_episode, eval_frames, eval_metrics = rollout_gen.rollout_eval()
        mem.append(eval_episode)
        save_video(eval_frames, 'results', f'vid_{i+1}')
        metrics.update(eval_metrics)

        for k, v in metrics.data.items():
            lineplot(np.arange(len(v)), v, k, path='results')

        if (i + 1) % 25 == 0:
            torch.save(rssm_model.state_dict(), f'results/ckpt_{i+1}.pth')

    pdb.set_trace()

if __name__ == '__main__':
    main()
