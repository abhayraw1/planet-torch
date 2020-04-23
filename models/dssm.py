import pdb

import torch
import numpy as np

from torch import nn
from torch.nn import GRUCell
from torch.nn import functional as F
from torch.distributions import Normal, kl

from utils import *
from models.models import *

class DeterministicStateSpaceModel(nn.Module):
    def __init__(self, action_size, state_size, latent_size, encoding_size):
        super().__init__()
        prior_in_size = state_size + action_size
        posterior_in_size = 2*latent_size + encoding_size
        self.transition_model = TransitionModel(action_size, state_size)
        self.latent_prior = StochasticModel(prior_in_size, latent_size)
        self.latent_posterior = StochasticModel(posterior_in_size, latent_size)
        self.encoder = EncoderModel(encoding_size)
        self.decoder = DecoderModel(state_size, latent_size)
        self.init_state_model = DeterministicModel(4*encoding_size, state_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def prior(self, actions, observation):
        """
        Should return the latents sampled from the prior dist
        inputs:
            actions: [N x T x D]
            observation: [N x C x H x W]
        """
        e0 = [self.encoder(o) for o in torch.unbind(observation, dim=1)]
        s0 = self.init_state_model(torch.cat(e0, dim=-1))
        sn = self.transition_model(actions, s0, unroll_dim=1)
        pr = self.latent_prior(torch.cat([sn, actions], dim=-1))
        return pr, sn

    def posterior(self, actions, observations, N, T, C, H, W):
        """
        Should return the latents sampled from the posterior dist
        inputs:
            actions: [N x T x D]
            observations: [N x T + 1 x C x H x W]
        """
        pr, sn = self.prior(actions[:, 3:], observations[:, :4])
        en = self.encoder(observations[:, 4:].reshape(N*(T-4), C, H, W))
        ps = self.latent_posterior(torch.cat([*pr, en.view(N, T-4, -1)], -1))
        return pr, ps, sn

    def forward(self, actions, observations):
        N, T, C, H, W = observations.shape
        pr, ps, sn = None, None, None
        # pdb.set_trace()
        if actions.shape[:2] != (N, T - 1):
            raise ValueError('Check time dim in actions and obs.')
        if observations.ndimension() == 4:
            raise NotImplementedError
            pr, sn = self.prior(actions, observations)
        if observations.ndimension() == 5:
            pr, ps, sn = self.posterior(actions, observations, N, T, C, H, W)
        return (*pr, *ps, sn)
        return {
            'prior': pr,
            'states': sn,
            'posterior': ps,
        }

    def train_on_batch(self, actions, observations, seq_lens):
        self.train()
        rets = self(actions, observations)
        prior_dist = Normal(*rets['prior'])
        posterior_dist = Normal(*rets['posterior'])
        preds = self.decoder(rets['states'], posterior_dist.rsample())

        kmask = get_mask(prior_dist.mean, seq_lens)
        rmask = get_mask(observations[:, 4:], seq_lens)
        rloss = F.mse_loss(preds, observations[:, 4:], reduction='none')*rmask
        rloss = rloss.sum((2, 3, 4)).mean()
        kloss = kl.kl_divergence(posterior_dist, prior_dist)*kmask
        kloss = kloss.sum((1, 2))
        kloss = torch.max(kloss - 2., torch.tensor(0.).to(kloss.device)).mean()
        # pdb.set_trace()

        self.optimizer.zero_grad()
        (rloss + kloss).backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1e2, norm_type=2)
        # print([x if x is None else x.grad.mean() for x in self.parameters()])
        self.optimizer.step()
        return kloss.item(), rloss.item()

    def evaluate(self, actions, observations, eps, prefix='eval_'):
        self.eval()
        rets = self(actions, observations)
        prior_dist = Normal(*rets['prior'])
        pred1 = self.decoder(rets['states'], prior_dist.mean)
        posterior_dist = Normal(*rets['posterior'])
        pred2 = self.decoder(rets['states'], posterior_dist.mean)
        save_frames(observations[0, 4:], pred1[0], pred2[0], f'{prefix}_{eps}')
        # rloss = ((pred2 - observations[:, 1:])**2).sum((2, 3, 4)).mean()
        # kloss = kl.kl_divergence(prior_dist, posterior_dist).sum((1, 2)).mean()
        # return kloss.item(), rloss.item()