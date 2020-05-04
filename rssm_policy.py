import pdb
import torch

from base_policy import Policy
from torch.distributions import Normal


class RSSMPolicy:
    def __init__(self,
            model,
            planning_horizon,
            num_candidates,
            num_iterations,
            top_candidates,
            device
        ):
        super().__init__()
        self.rssm = model
        self.N = num_candidates
        self.K = top_candidates
        self.T = num_iterations
        self.H = planning_horizon
        self.d = self.rssm.action_size
        self.device = device
        self.state_size = self.rssm.state_size
        self.latent_size = self.rssm.latent_size

    def reset(self):
        self.prev_state = torch.zeros(1, self.state_size).to(self.device)
        self.prev_latent = torch.zeros(1, self.latent_size).to(self.device)
        self.prev_action = torch.zeros(1, self.d).to(self.device)

    def poll(self, observation):
        self.mu = torch.zeros(self.H, self.d).to(self.device)
        self.stddev = torch.ones(self.H, self.d).to(self.device)
        # observation could be of shape [CHW] but only 1 timestep
        assert len(observation.shape) == 3, 'obs should be [CHW]'
        # pdb.set_trace()
        self.prev_state, self.prev_latent = self.rssm.get_init_state(
            self.rssm.encoder(observation[None]),
            self.prev_state, self.prev_latent, self.prev_action
        )
        h_t = self.prev_state.expand(self.N, -1).clone()
        s_t = self.prev_latent.expand(self.N, -1).clone()
        for _ in range(self.T):
            actions = Normal(self.mu, self.stddev).sample((self.N,))
            rewards = torch.zeros(self.N).to(self.device)
            for a_t in torch.unbind(actions, dim=1):
                h_t = self.rssm.deterministic_state_fwd(h_t, s_t, a_t)
                s_t = self.rssm.state_prior(h_t, sample=True)
                rewards += self.rssm.pred_reward(h_t, s_t)
            _, k = torch.topk(rewards, self.K, 0, largest=True, sorted=False)
            self.mu = actions[k].mean(dim=0)
            self.stddev = actions[k].std(dim=0)
        # print(self.mu.cpu().numpy())
        self.prev_action = self.mu[0:1]
        return self.prev_action
