import numpy as np
import torch
from tqdm import trange
from memory import Episode # this needs modification!
from torchvision.utils import make_grid

class RolloutGenerator:
    """Rollout generator class."""
    def __init__(self,
        env,
        device,
        policy=None,
        max_episode_steps=None,
        episode_gen=None,
        name=None,
    ):
        self.env = env
        self.device = device
        self.policy = policy
        self.episode_gen = episode_gen or Episode
        self.name = name or 'Rollout Generator'
        self.max_episode_steps = max_episode_steps
        if self.max_episode_steps is None:
            self.max_episode_steps = self.env.max_episode_steps

    def rollout_once(self, random_policy=False, explore=False) -> Episode:
        """Performs a single rollout of an environment given a policy
        and returns and episode instance.
        """
        if self.policy is None and not random_policy:
            random_policy = True
            print('Policy is None. Using random policy instead!!')
        if not random_policy:
            self.policy.reset()
        eps = self.episode_gen()
        obs = self.env.reset()
        des = f'{self.name} Ts'
        for _ in trange(self.max_episode_steps, desc=des, leave=False):
            if random_policy:
                act = self.env.sample_random_action()
            else:
                act = self.policy.poll(obs.to(self.device)).flatten()
                if explore:
                    act += torch.randn_like(act)*0.3
            nobs, reward, terminal, _ = self.env.step(act)
            eps.append(obs, act, reward, terminal)
            obs = nobs
        eps.terminate(nobs)
        return eps 

    def rollout_n(self, n=1, random_policy=False) -> [Episode]:
        """
        Performs n rollouts.
        """
        if self.policy is None and not random_policy:
            random_policy = True
            print('Policy is None. Using random policy instead!!')
        des = f'{self.name} EPS'
        ret = []
        for _ in trange(n, desc=des, leave=False):
            ret.append(self.rollout_once(random_policy=random_policy))
        return ret

    def rollout_eval(self):
        assert self.policy is not None, 'Policy is None!!'
        self.policy.reset()
        eps = self.episode_gen()
        obs = self.env.reset()
        des = f'{self.name} Eval Ts'
        frames = []
        metrics = {}
        rec_losses = []
        rew_losses = []
        eps_reward = 0
        for _ in trange(self.max_episode_steps, desc=des, leave=False):
            with torch.no_grad():
                act = self.policy.poll(obs.to(self.device)).flatten()
                dec = self.policy.rssm.decoder(
                    self.policy.prev_state,
                    self.policy.prev_latent
                ).squeeze().detach().cpu()
                rec_losses.append(((obs - dec)**2).sum())
                p_reward = self.policy.rssm.pred_reward(
                    self.policy.prev_state,
                    self.policy.prev_latent
                ).cpu().numpy()
                frames.append(make_grid([obs + 0.5, dec + 0.5], nrow=2).numpy())
            nobs, reward, terminal, _ = self.env.step(act)
            eps.append(obs, act, reward, terminal)
            eps_reward += reward
            obs = nobs
            rew_losses.append(abs(reward - p_reward.item()))
        eps.terminate(nobs)
        metrics['eval_reward'] = eps_reward
        metrics['eval_rc_loss'] = rec_losses
        metrics['eval_reward_loss'] = rew_losses
        return eps, np.stack(frames), metrics
