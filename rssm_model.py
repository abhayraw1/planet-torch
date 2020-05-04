import torch

from torch import nn
from torch.nn import functional as F


class VisualEncoder(nn.Module):
    def __init__(self, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
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


class VisualObservationModel(nn.Module):
    def __init__(self,
            state_size,
            latent_size,
            embedding_size,
            activation_function='relu'
        ):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(latent_size + state_size, embedding_size)
        self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

    def forward(self, state, latent):
        hidden = self.fc1(torch.cat([latent, state], dim=1))
        hidden = hidden.view(-1, self.embedding_size, 1, 1)
        hidden = self.act_fn(self.conv1(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        observation = self.conv4(hidden)
        return observation


class RecurrentStateSpaceModel(nn.Module):
    def __init__(self,
            action_size,
            state_size=200,
            latent_size=30,
            hidden_size=128,
            embed_size=1024,
            activation_function='relu'
        ):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.latent_size = latent_size
        self.act_fn = getattr(F, activation_function)
        self.encoder = VisualEncoder(embed_size)
        self.decoder = VisualObservationModel(
            state_size, latent_size, embed_size
        )
        self.grucell = nn.GRUCell(state_size, state_size)
        self.lat_act_layer = nn.Linear(latent_size + action_size, state_size)
        self.fc_prior_1 = nn.Linear(state_size, hidden_size)
        self.fc_prior_m = nn.Linear(hidden_size, latent_size)
        self.fc_prior_s = nn.Linear(hidden_size, latent_size)
        self.fc_posterior_1 = nn.Linear(state_size + embed_size, hidden_size)
        self.fc_posterior_m = nn.Linear(hidden_size, latent_size)
        self.fc_posterior_s = nn.Linear(hidden_size, latent_size)
        self.fc_reward_1 = nn.Linear(state_size + latent_size, hidden_size)
        self.fc_reward_2 = nn.Linear(hidden_size, 1)


    def get_init_state(self, enc, h_t, s_t, a_t):
        h_tp1 = self.deterministic_state_fwd(h_t, s_t, a_t)
        s_tp1 = self.state_posterior(h_t, enc, sample=True)
        return h_tp1, s_tp1

    def deterministic_state_fwd(self, h_t, s_t, a_t):
        """Returns h_tp1 = f(h_t, s_t, a_t)"""
        h = torch.cat([s_t, a_t], dim=-1)
        h = self.act_fn(self.lat_act_layer(h))
        return self.grucell(h, h_t)

    def state_prior(self, h_t, sample=False):
        z = self.act_fn(self.fc_prior_1(h_t))
        m = self.fc_prior_m(z)
        s = F.softplus(self.fc_prior_s(z))# + 1e-1
        if sample:
            return m + torch.randn_like(m) * s
        return m, s

    def state_posterior(self, h_t, e_t, sample=False):
        z = torch.cat([h_t, e_t], dim=-1)
        z = self.act_fn(self.fc_posterior_1(z))
        m = self.fc_posterior_m(z)
        s = F.softplus(self.fc_posterior_s(z))# + 1e-1
        if sample:
            return m + torch.randn_like(m) * s
        return m, s

    def pred_reward(self, h_t, s_t):
        r = self.act_fn(self.fc_reward_1(torch.cat([h_t, s_t], dim=-1)))
        return self.fc_reward_2(r).squeeze()