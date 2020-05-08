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

def train(memory, rssm, optimizer, device, N=16, H=50):
    batch = memory.sample(N, H, time_first=True)
    x, u, r, t  = [torch.tensor(x).float().to(device) for x in batch]
    preprocess_img(x, depth=5)
    a_t = torch.zeros(N, rssm.action_size).to(device)
    free_nats = torch.ones_like(a_t.flatten())*3.0
    h_t, s_t = rssm.get_init_state(rssm.encoder(x[0]))
    kl_loss, rc_loss, re_loss = 0, 0, 0
    for i, a_t in enumerate(torch.unbind(u, dim=0)):
        h_t = rssm.deterministic_state_fwd(h_t, s_t, a_t)
        sm_t, ss_t = rssm.state_prior(h_t)
        pm_t, ps_t = rssm.state_posterior(h_t, rssm.encoder(x[i + 1]))
        s_t = pm_t + torch.randn_like(pm_t)*ps_t
        rec = rssm.decoder(h_t, s_t)
        kl_div = kl_divergence(Normal(pm_t, ps_t), Normal(sm_t, ss_t))
        kl_loss += torch.max(free_nats, kl_div.sum(-1)).mean()
        rc_loss += ((rec - x[i + 1]).abs()).sum((1, 2, 3)).mean()
        re_loss += F.mse_loss(rssm.pred_reward(h_t, s_t), r[i])

    kl_loss, rc_loss, re_loss = [x/H for x in (kl_loss, rc_loss, re_loss)]
    optimizer.zero_grad()
    nn.utils.clip_grad_norm_(rssm.parameters(), 100., norm_type=2)
    (kl_loss + rc_loss + re_loss).backward()
    optimizer.step()
    return {'kl': kl_loss.item(), 'rc': rc_loss.item(), 're': re_loss.item()}


def train_with_latent_overshoot(memory, rssm, opt, device, N=16, H=50, D=5):
    batch = memory.sample(N, H, time_first=True)
    x, u, r, t  = [torch.tensor(x).float().to(device) for x in batch]
    preprocess_img(x, depth=5)
    a_t = torch.zeros(N, rssm.action_size).to(device)
    free_nats = torch.ones_like(a_t.flatten())*2.0
    h_t, s_t = rssm.get_init_state(rssm.encoder(x[0]))
    kl_loss, rc_loss, re_loss = 0, 0, 0
    for i, a_t in enumerate(torch.unbind(u, dim=0)):
        h_t = rssm.deterministic_state_fwd(h_t, s_t, a_t)
        sm_t, ss_t = rssm.state_prior(h_t)
        pm_t, ps_t = rssm.state_posterior(h_t, rssm.encoder(x[i + 1]))
        s_t = pm_t + torch.randn_like(pm_t)*ps_t
        rec = rssm.decoder(h_t, s_t)
        kl_div = kl_divergence(Normal(pm_t, ps_t), Normal(sm_t, ss_t))
        kl_loss += torch.max(free_nats, kl_div.sum(-1)).mean()
        rc_loss += ((rec - x[i + 1])**2).sum((1, 2, 3)).mean()
        re_loss += F.mse_loss(rssm.pred_reward(h_t, s_t), r[i])

    kl_loss, rc_loss, re_loss = [x/H for x in (kl_loss, rc_loss, re_loss)]
    opt.zero_grad()
    nn.utils.clip_grad_norm_(rssm.parameters(), 100., norm_type=2)
    (kl_loss + rc_loss + re_loss).backward()
    opt.step()
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
    # rssm_model.load_state_dict(torch.load('results/ckpt_100.pth'))
    pdb.set_trace()
        # episode_to_videos
        # evaluate(mem, rssm_model.eval(), device, save_video=True)


    """
    Do training here !!
    1. Sample a batch of experience from the memory
    2. Do as fwd pass like policy wala
    3. compute duniya bhar ka loss
    4. backprop
    5. do an evaluation
    """  
    

if __name__ == '__main__':
    main()
