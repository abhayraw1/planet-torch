# import gym
import pdb
import pickle

from utils import *
from memory import *
from rssm_model import *
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

"""
Training implementation as indicated in:
    Learning Latent Dynamics for Planning from Pixels
    arXiv:1811.04551
    
    (c.) The Latent Overshooting Method
         using only single step predictions.
"""

def train(memory, model, optimizer, record_grads=True):
    """
    Trained using the Standard Variational Bound method indicated in Fig. 3a
    """
    model.train()
    metrics = defaultdict(list)
    if record_grads:
        metrics['grads'] = defaultdict(list)
    for _ in trange(10, desc='# Epoch: ', leave=False):
        (x, u, _, _), lens = memory.sample(32)
        states, priors, posteriors = model(x, u)
        prior_dists = [Normal(*p) for p in priors]
        posterior_dists = [Normal(*p) for p in posteriors]
        posterior_samples = [d.rsample() for d in posterior_dists]
        # Reconstruction Loss
        rx = model.decoder(states[0], posterior_samples[0])
        iloss = (((x[:, 0] - rx)**2).sum((1, 2, 3))).mean()
        # KL Divergence
        kl = kl_div(prior_dists[0], posterior_dists[0]).sum(-1)
        kloss = torch.max(FREE_NATS, kl).mean()
        mask = get_mask(u[..., 0], lens).T
        for i in range(1, len(states)):
            rx = model.decoder(states[i], posterior_samples[i])
            iloss += (((x[:, i] - rx)**2).sum((1, 2, 3))*mask[i-1]).mean()
            kl = kl_div(prior_dists[i], posterior_dists[i]).sum(-1)
            kloss += torch.max(FREE_NATS, (kl*mask[i-1])).mean()
        kloss /= len(states)
        iloss /= len(states)
        optimizer.zero_grad()
        (iloss + kloss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 100, norm_type=2)
        if record_grads:
            pprint({
                k: 0 if x.grad is None else x.grad.mean().item()
                for k, x in dict(model.named_parameters()).items()
            })
        metrics['kl_losses'].append(kloss.item())
        metrics['rec_losses'].append(iloss.item())
        optimizer.step()
    return metrics


def evaluate(memory, model, path, eps):
    model.eval()
    (x, u, _, _), lens = memory.sample(1)
    states, priors, posteriors = model(x, u)
    states = torch.stack(states).squeeze()
    priors =  Normal(*map(lambda x: torch.stack(x).squeeze(), zip(*priors)))
    posts =  Normal(*map(lambda x: torch.stack(x).squeeze(), zip(*posteriors)))
    pred1 = model.decoder(states, priors.mean)
    pred2 = model.decoder(states, posts.mean)
    save_frames(x[0], pred1, pred2, f'{path}_{eps}')


def main():
    global FREE_NATS
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    FREE_NATS = torch.full((1, ), FREE_NATS).to(device)
    rssm = RecurrentStateSpaceModel(1, STATE_SIZE, LATENT_SIZE, EMBEDDING_SIZE)
    rssm = rssm.to(device)
    optimizer = torch.optim.Adam(get_combined_params(rssm), lr=1e-3)

    test_data = load_memory('test_exp_replay.pth', device)
    train_data = load_memory('train_exp_replay.pth', device)
    
    global_metrics = defaultdict(list)
    for i in trange(1000, desc='# Episode: ', leave=False):
        metrics = train(train_data, rssm, optimizer, record_grads=False)
        for k, v in metrics.items():
            global_metrics[k].extend(metrics[k])
        plot_metrics(global_metrics, path='results/test_rssm', prefix='TRAIN_')
        if (i + 1) % 10 == 0:
            evaluate(test_data, rssm, 'results/test_rssm/eps', i + 1)
        if (i + 1) % 25 == 0:
            torch.save(rssm.state_dict(), f'results/test_rssm/ckpt_{i+1}.pth')
    pdb.set_trace()


if __name__ == '__main__':
    main()