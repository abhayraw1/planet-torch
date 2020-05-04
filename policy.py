import torch

from tqdm import trange
from torch import nn
from torch.distributions import Normal


class CrossEntropyBasedOptimizer(nn.Module):
  def __init__(self,
      model,
      planning_horizon,
      num_candidates,
      action_size,
      num_iterations,
      top_candidates
    ):
    super().__init__()
    self.model = model
    self.d = action_size
    self.N = num_candidates
    self.K = top_candidates
    self.T = num_iterations
    self.H = planning_horizon
    self.register_buffer('mu', torch.zeros((self.H, self.d)))
    self.register_buffer('stddev', torch.ones((self.H, self.d)))

  def clear(self):
    self.mu.data *= 0 
    self.stddev.data *= 0 
    self.stddev.data += 1 

  def forward(self, observation):
    """
    observation shape should be [1, 1, 3, 64, 64]
    """
    self.clear()
    observation = observation.expand(self.N, 1, *observation.shape)
    with torch.no_grad():
      for _ in range(self.T):
        actions = Normal(self.mu, self.stddev).sample((self.N,))
        state, priors = self.model(observation, actions)
        r = self.model.pred_reward(state, priors['means']).sum(-1)
        _, k = torch.topk(r, self.K, 0, largest=True, sorted=False)
        self.mu = actions[k].mean(dim=0)
        self.stddev = actions[k].std(dim=0)
      return self.mu[0].cpu()
