import pdb

import torch
import numpy as np

from torch import nn
from torch.nn import GRUCell
from torch.nn import functional as F
from torch.distributions import Normal, kl

from utils import *
from models.models import *


class RecurrentStateSpaceModel(nn.Module):
  def __init__(self, action_size, state_size, latent_size, encoding_size):
    super().__init__()
    ip_latent_act = latent_size + action_size
    ip_posteriors = encoding_size + state_size
    # add later? not training it right now =--------
    ip_init_state = encoding_size# + state_size <--|
    self.encoder = EncoderModel(encoding_size)
    self.decoder = DecoderModel(state_size, latent_size)
    self.prior_model = StochasticModel(state_size, latent_size)
    self.reward_model = DeterministicModel(state_size + latent_size, 1)
    self.posterior_model = StochasticModel(ip_posteriors, latent_size)
    self.transition_model = GRUCell(state_size, state_size)
    self.init_state_model = StochasticModel(ip_init_state, state_size)
    self.concat_latent_act = DeterministicModel(ip_latent_act, state_size)


  def get_priors_and_states(self, init_state, actions):
    states = [init_state]
    pm, ps = self.prior_model(states[-1])
    priors = {
      'means': [pm],
      'stddevs': [ps],
      'samples': [pm + torch.randn_like(pm)*ps]
    }
    for i in range(1, actions.size(1) + 1):
      cat_latent_act = self.concat_latent_act(
        torch.cat([priors['samples'][-1], actions[:, i - 1]], dim=-1)
      )
      states.append(self.transition_model(cat_latent_act, states[-1]))
      pm, ps = self.prior_model(states[-1])
      priors['means'].append(pm)
      priors['stddevs'].append(ps)
      priors['samples'].append(pm + torch.randn_like(pm)*ps)
    states = torch.stack(states, dim=1)
    priors = {k: torch.stack(v, dim=1) for k, v in priors.items()}
    return states, priors

  def forward(self, observations, actions, init_state=None):
    """
    return time unfolded batches of prior/ posterior latents and states
    conditioned on the actions and observations
    if t in observation.size(1) == 1 then return only priors
    else return posteriors too!!
    inputs:
      observations [n x t x 3 x 64 x 64]
      actions [n x t x action_size]
      init_state not_implemented
    outputs: 
      states [n x t x d]
      priors {'means', 'stddevs', 'samples': [n x t x l]}
      posteriors {'means', 'stddevs', 'samples': [n x t x l]}
    """
    n, t, c, h, w = observations.shape
    en = self.encoder(observations.view(n*t, c, h, w)).view(n, t, -1)
    sm, sd = self.init_state_model(en[:, 0])
    s0 = sm + torch.randn_like(sm)*sd
    states, priors = self.get_priors_and_states(s0, actions)

    if t == 1:
      return states, priors
    # pdb.set_trace()
    pm, ps = self.posterior_model(torch.cat([states[:, 0], en[:, 0]], dim=-1))
    posteriors = {
      'means': [pm],
      'stddevs': [ps],
      'samples': [pm + torch.randn_like(pm)*ps]
    }
    for i in range(1, actions.size(1) + 1):
      pm, ps = self.posterior_model(torch.cat([states[:, i], en[:, i]], dim=-1))
      posteriors['means'].append(pm)
      posteriors['stddevs'].append(ps)
      posteriors['samples'].append(pm + torch.randn_like(pm)*ps)
    
    posteriors = {k: torch.stack(v, dim=1) for k, v in posteriors.items()}
    return states, priors, posteriors

  def pred_reward(self, states, latents):
    return self.reward_model(torch.cat([states, latents], dim=-1)).squeeze()
