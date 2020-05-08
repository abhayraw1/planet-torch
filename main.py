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

def train(memory, rssm, optimizer, device, N=16, H=40):
    free_nats = torch.ones(1, device=device)*3.0
    batch = memory.sample(N, H, time_first=True)
    x, u, r, t  = [torch.tensor(x).float().to(device) for x in batch]
    preprocess_img(x, depth=5)
    h_t = torch.zeros(N, rssm.state_size).to(device)
    s_t = torch.zeros(N, rssm.latent_size).to(device)
    a_t = torch.zeros(N, rssm.action_size).to(device)
    h_t, s_t = rssm.get_init_state(rssm.encoder(x[0]), h_t, s_t, a_t)
    kl_loss, rc_loss, re_loss = 0, 0, 0
    for i, a_t in enumerate(torch.unbind(u, dim=0)):
        h_t = rssm.deterministic_state_fwd(h_t, s_t, a_t)
        sm_t, ss_t = rssm.state_prior(h_t)
        pm_t, ps_t = rssm.state_posterior(h_t, rssm.encoder(x[i + 1]))
        s_t = pm_t + torch.randn_like(pm_t)*ps_t
        rec = rssm.decoder(h_t, s_t)
        kl_div = kl_divergence(Normal(pm_t, ps_t), Normal(sm_t, ss_t))
        kl_loss += kl_div.sum(-1).mean()
        rc_loss += ((rec - x[i + 1]).abs()).sum((1, 2, 3)).mean()
        re_loss += F.mse_loss(rssm.pred_reward(h_t, s_t), r[i])

    kl_loss, rc_loss, re_loss = [x/H for x in (kl_loss, rc_loss, re_loss)]
    optimizer.zero_grad()
    nn.utils.clip_grad_norm_(rssm.parameters(), 100., norm_type=2)
    (torch.max(kl_loss, free_nats) + rc_loss + re_loss).backward()
    optimizer.step()
    return {'kl': kl_loss.item(), 'rc': rc_loss.item(), 're': re_loss.item()}


def main():
    env = TorchImageEnvWrapper('Pendulum-v0', bit_depth=5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rssm_model = RecurrentStateSpaceModel(env.action_size).to(device)
    optimizer = torch.optim.Adam(rssm_model.parameters(), lr=1e-3)
    policy = RSSMPolicy(
        rssm_model, 
        planning_horizon=12,
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
        max_episode_steps=100,
    )
    mem = Memory(100)
    mem.append(rollout_gen.rollout_n(15, random_policy=True))
    metrics = defaultdict(list)
    for i in trange(100, desc='Epoch', leave=False):
        for _ in trange(150, desc='Iter ', leave=False):
            losses = train(mem, rssm_model.train(), optimizer, device)
            for k, v in losses.items():
                metrics[k].append(v)
        mem.append(rollout_gen.rollout_once(explore=True))
        # pdb.set_trace()
        print(losses)
        eval_episode, eval_frames, eval_rec_loss = rollout_gen.rollout_eval()
        mem.append(eval_episode)
        save_video(eval_frames, 'results', f'vid_{i+1}')
        metrics['eval_rec_loss'].append(eval_rec_loss)

        for k, v in metrics.items():
            lineplot(np.arange(len(v)), v, k, path='results')

        if (i + 1) % 25 == 0:
            torch.save(rssm_model.state_dict(), f'results/ckpt_{i+1}.pth')

    pdb.set_trace()

if __name__ == '__main__':
    main()
