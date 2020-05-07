import pdb
import torch
import numpy as np

from utils import *
from collections import deque
from numpy.random import choice
from torch import float32 as F32
from torch.nn.utils.rnn import pad_sequence

class Episode:
    """Records the agent's interaction with the environment for a single
    episode. At termination, it converts all the data to Numpy arrays.
    """
    def __init__(self, postprocess_fn=lambda x: x):
        self.x = []
        self.u = []
        self.t = []
        self.r = []
        self.postprocess_fn = postprocess_fn
        self._size = 0

    @property
    def size(self):
        return self._size

    def append(self, obs, act, reward, terminal):
        self._size += 1
        self.x.append(self.postprocess_fn(obs.numpy()))
        self.u.append(act.cpu().numpy())
        self.r.append(reward)
        self.t.append(terminal)

    def terminate(self, obs):
        self.x.append(self.postprocess_fn(obs.numpy()))
        self.x = np.stack(self.x)
        self.u = np.stack(self.u)
        self.r = np.stack(self.r)
        self.t = np.stack(self.t)


class Memory(deque):
    def __init__(self, size):
        """Maintains a FIFO list of `size` number of episodes.
        """
        self.episodes = deque(maxlen=size)
        self.eps_lengths = deque(maxlen=size)
        print(f'Creating memory with len {size} episodes.')

    @property
    def size(self):
        return sum(self.eps_lengths)

    def _append(self, episode: Episode):
        if isinstance(episode, Episode):
            self.episodes.append(episode)
            self.eps_lengths.append(episode.size)
        else:
            raise ValueError('can only append <Episode> or list of <Episode>')

    def append(self, episodes: [Episode]):
        if isinstance(episodes, Episode):
            episodes = [episodes]
        if isinstance(episodes, list):
            for e in episodes:
                self._append(e)
        else:
            raise ValueError('can only append <Episode> or list of <Episode>')

    def sample(self, batch_size, tracelen=1, time_first=False):
        episode_idx = choice(len(self.episodes), batch_size)
        init_st_idx = [
            choice(self.eps_lengths[i] - tracelen + 1)
            for i in episode_idx
        ]
        x, u, r, t = [], [], [], []
        for n, (i, s) in enumerate(zip(episode_idx, init_st_idx)):
            x.append(self.episodes[i].x[s: s + tracelen + 1])
            u.append(self.episodes[i].u[s: s + tracelen])
            r.append(self.episodes[i].r[s: s + tracelen])
            t.append(self.episodes[i].t[s: s + tracelen])
        if tracelen == 1:
            rets = [np.stack(x)] + [np.stack(i)[:, 0] for i in (u, r, t)]
        else:
            rets = [np.stack(i) for i in (x, u, r, t)]
        if time_first:
            rets = [a.swapaxes(1, 0) for a in rets]
        return rets
