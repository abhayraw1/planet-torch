import pdb

import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

def to_time_independent_batches(x):
    pass


class EncoderModel(nn.Module):
    def __init__(self, embedding_size, activation_function='elu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        if embedding_size == 1024:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(1024, embedding_size)

    def forward(self, observation):
        hidden = self.act_fn(self.conv1(observation))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        hidden = hidden.view(-1, 1024)
        hidden = self.fc(hidden)
        return hidden


class DecoderModel(nn.Module):
    def __init__(self, state_size, latent_size, activation_function='elu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(state_size + latent_size, 1024)
        self.conv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)
        # self.embedding_size = embedding_size

    def forward(self, state, latent):
        # hidden = self.act_fn(self.fc1(state))
        n, _ = state.shape
        hidden = self.act_fn(self.fc1(torch.cat([state, latent], dim=-1)))
        hidden = hidden.view(n, 1024, 1, 1)
        hidden = self.act_fn(self.conv1(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        observation = self.conv4(hidden)
        return observation.view(n, *observation.shape[1:])


class StochasticModel(nn.Module):
    def __init__(self, input_size, output_size, activation_function='elu'):
        super().__init__()
        hidden_size = 128
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, output_size)
        self.std = nn.Linear(hidden_size, output_size)
    
    def forward(self, embed):
        hidden = self.act_fn(self.fc1(embed))
        return (self.mean(hidden), F.softplus(self.std(hidden)))
        # hidden = self.act_fn(self.fc2(hidden))


class DeterministicModel(nn.Module):
    def __init__(self, input_size, output_size, activation_function='elu'):
        super().__init__()
        hidden_size = 128
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, hidden):
        hidden = self.act_fn(self.fc1(hidden))
        return self.act_fn(self.fc3(hidden))
        # hidden = self.act_fn(self.fc2(hidden))


class PosteriorModel(nn.Module):
    def __init__(self, state_size, embed_size, activation_function='elu'):
        super().__init__()
        hidden_size = 128
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, state_size)
        self.std = nn.Linear(hidden_size, state_size)

    def forward(self, embed):
        hidden = self.act_fn(self.fc1(embed))
        hidden = self.act_fn(self.fc2(hidden))
        return Normal(self.mean(hidden), F.softplus(self.std(hidden)))


class TransitionModel(nn.Module):
    def __init__(self, action_size, state_size, activation_function='elu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.rnn = nn.GRUCell(action_size, state_size)
        self.fc1 = nn.Linear(state_size, state_size)
        self.fc2 = nn.Linear(state_size, state_size)

    def forward(self, actions, state, unroll_dim=1):
        states = []
        for action in torch.unbind(actions, dim=unroll_dim):
            state = self.act_fn(self.rnn(action, state))
            state = self.act_fn(self.fc1(state))
            state = self.act_fn(self.fc2(state))
            states.append(state.clone())
        return torch.stack(states).transpose(0, 1)


class TransitionModelv2(nn.Module):
    def __init__(self,
            input_size, latent_size, activation_function='elu'
        ):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.rnn = nn.GRUCell(action_size, latent_size)
        self.fc1 = nn.Linear(latent_size, latent_size)
        self.fc2 = nn.Linear(latent_size, latent_size)

    def forward(self, state, prev_state, unroll_dim=1):
        states = []
        for action in torch.unbind(actions, dim=unroll_dim):
            state = self.act_fn(self.rnn(action, state))
            state = self.act_fn(self.fc1(state))
            state = self.act_fn(self.fc2(state))
            states.append(state.clone())
        return torch.stack(states).transpose(0, 1)


class PriorTransitionModel(nn.Module):
    def __init__(self, state_size, action_size, activation_function='elu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.rnn = nn.GRUCell(action_size, state_size)
        self.fc1 = nn.Linear(state_size, state_size)
        self.fc2 = nn.Linear(state_size, state_size)
        self.mean = nn.Linear(state_size, state_size)
        self.std = nn.Linear(state_size, state_size)
        self.register_buffer('zero_state', torch.zeros(1, state_size))
        self.register_buffer('zero_action', torch.zeros(1, action_size))

    def forward(self, state=None, actions=None):
        state = self.zero_state if state is None else state
        actions = self.zero_action if actions is None else actions
        means, stds = [], []
        for action in torch.unbind(actions):
            state = self.act_fn(self.rnn(action[None], state))
            state = self.act_fn(self.fc1(state))
            state = self.act_fn(self.fc2(state))
            means.append(self.mean(state).flatten())
            stds.append(F.softplus(self.std(state)).flatten())
        return Normal(torch.stack(means), torch.stack(stds))


class PosteriorTransitionModel(nn.Module):
    def __init__(self, state_size, embedding_size, activation_function='elu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(state_size + embedding_size, state_size)
        self.fc2 = nn.Linear(state_size, state_size)
        self.mean = nn.Linear(state_size, state_size)
        self.std = nn.Linear(state_size, state_size)

    def forward(self, state, embedding):
        hidden = self.act_fn(self.fc1(torch.cat([state, embedding], dim=-1)))
        hidden = self.act_fn(self.fc2(hidden))
        return Normal(self.mean(hidden), F.softplus(self.std(hidden)))
