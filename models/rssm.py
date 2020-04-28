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

    def forward(self, observations, actions, init_state=None):
        n, t, c, h, w = observations.shape
        if actions.size(1) != (t - 1):
            raise ValueError('action obs shape prob...')
        en = self.encoder(observations.view(n*t, c, h, w)).view(n, t, -1)
        states = [Normal(*self.init_state_model(en[:, 0])).rsample()]
        prior_states = [self.prior_model(states[-1])]
        posterior_states = [
            self.posterior_model(torch.cat([states[-1], en[:, 0]], dim=-1))
        ]
        for i in range(1, t):
            cat_latent_act = self.concat_latent_act(
                torch.cat([
                    Normal(*prior_states[-1]).sample(),
                    actions[:, i - 1]
                ], dim=-1)
            )
            states.append(self.transition_model(cat_latent_act, states[-1]))
            prior_states.append(self.prior_model(states[-1]))
            posterior_states.append(
                self.posterior_model(torch.cat([states[-1], en[:, i]], dim=-1))
            )
        return states, prior_states, posterior_states

    def pred_reward(self, states, latents):
        return self.reward_model(torch.cat([states, latents], dim=-1))
