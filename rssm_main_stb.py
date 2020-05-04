import gym
import pdb
import pickle

from utils import *
from policy import *
from memory import *
from models.rssm import *
from tqdm import trange
from collections import defaultdict
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence as kl_div

from pprint import pprint


BIT_DEPTH = 5
FREE_NATS = 2
STATE_SIZE = 200
LATENT_SIZE = 30
EMBEDDING_SIZE = 1024

################### POLICY PARAMETERS ###################
NUM_OPTIM_ITERS = 10
PLANNING_HORIZON = 8
TOP_OPTIM_CANDIDATES = 20
NUM_OPTIM_CANDIDATES = 200


"""
Training implementation as indicated in:
  Learning Latent Dynamics for Planning from Pixels
  arXiv:1811.04551

  (a.) The Standard Varioational Bound Method
     using only single step predictions.
"""

def train(memory, model, optimizer, record_grads=False):
  """
  Trained using the Standard Variational Bound method indicated in Fig. 3a
  """
  model.train()
  metrics = defaultdict(list)
  if record_grads:
    metrics['grads'] = defaultdict(list)
  for _ in trange(50, desc='# Epoch: ', leave=False):
    (x, u, r, _), lens = memory.sample(32)
    n, t = x.shape[:2]

    states, priors, posteriors = model(x, u)
    prior_dists = Normal(priors['means'], priors['stddevs'])
    posterior_dists = Normal(posteriors['means'], posteriors['stddevs'])
    posterior_samples = posteriors['samples']
    # Get Mask for masking out timesteps not to be considered.
    mask = get_mask(u[..., 0], lens)
    mask = torch.cat([torch.ones_like(mask[:, :1]), mask], dim=-1)
    # Reconstruction Loss
    rx = model.decoder(states.view(n*t, -1), posterior_samples.view(n*t, -1))
    rec_loss = (((x - rx.view(n, t, 3, 64, 64))**2).sum((2, 3, 4)))
    rec_loss = (rec_loss*mask).mean()
    # KL Divergence
    kl = kl_div(prior_dists, posterior_dists).sum(-1)*mask
    kl_loss = torch.max(FREE_NATS, kl).mean()
    # Reward Loss
    rw = model.pred_reward(states, posterior_samples)*mask
    rw_loss = ((rw[:, 1:] - r)**2).mean()
    optimizer.zero_grad()
    (rec_loss + kl_loss + rw_loss).backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1000, norm_type=2)
    if record_grads:
      pprint({
        k: 0 if x.grad is None else x.grad.mean().item()
        for k, x in dict(model.named_parameters()).items()
      })
    metrics['kl_losses'].append(kl_loss.item())
    metrics['rec_losses'].append(rec_loss.item())
    optimizer.step()
  return metrics


def rollout(memory, env, policy):
  episode = Episode(memory.device, BIT_DEPTH)
  x = env.reset()
  eps_reward = 0
  for _ in trange(
      env.env._max_episode_steps,
      desc='Episode Rollout',
      leave=False
    ):
    u = policy(x.to(memory.device))
    nx, r, d, _ = env.step(u)
    episode.append(x, u, r, d)
    eps_reward += r
    x = nx
  episode.append_last_obs(x)
  memory.append(episode)
  return episode, eps_reward


def evaluate(memory, policy, model, env, path, eps):
  model.eval()
  eps_rewards = []
  for i in trange(10, desc='Eval Episodes', leave=False):
    episode, eps_reward = rollout(memory, env, policy)
    eps_rewards.append(eps_reward)
  (x, u, _, _) = episode.prepare()
  states, priors, posteriors = model(x[None], u[None])
  states = states.squeeze()
  pred1 = model.decoder(states, priors['means'].squeeze())
  pred2 = model.decoder(states, posteriors['means'].squeeze())
  save_frames(x, pred1, pred2, f'{path}_{eps}')
  return {'eps_reward': eps_rewards}


def main():
  global FREE_NATS
  env = TorchImageEnvWrapper('Pendulum-v0', BIT_DEPTH)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  FREE_NATS = torch.full((1, ), FREE_NATS).to(device)
  rssm = RecurrentStateSpaceModel(1, STATE_SIZE, LATENT_SIZE, EMBEDDING_SIZE)
  rssm = rssm.to(device)
  optimizer = torch.optim.Adam(get_combined_params(rssm), lr=1e-3)

  policy = CrossEntropyBasedOptimizer(
    rssm, PLANNING_HORIZON, NUM_OPTIM_CANDIDATES, env.action_size,
    NUM_OPTIM_ITERS, TOP_OPTIM_CANDIDATES
  ).to(device)

  memory = load_memory('test_exp_replay.pth', device)
  # pdb.set_trace()  
  global_metrics = defaultdict(list)
  for i in trange(1000, desc='# Episode: ', leave=False):
    # Train on memory
    metrics = train(memory, rssm, optimizer, record_grads=False)

    # checkpoint
    if (i + 1) % 25 == 0:
      torch.save(rssm.state_dict(), f'results/test_rssm/ckpt_{i+1}.pth')

    # Evaluate and Save Predictions
    if (i + 1) % 10 == 0:
      mx = evaluate(memory, policy, rssm, env, 'results/test_rssm/eps', i + 1)
      metrics.update(mx)
    # Plot Data
    for k, v in metrics.items():
      global_metrics[k].extend(metrics[k])
    plot_metrics(global_metrics, path='results/test_rssm', prefix='TRAIN_')
  pdb.set_trace()


if __name__ == '__main__':
  main()