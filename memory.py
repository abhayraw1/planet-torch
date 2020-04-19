import pdb
import torch
import numpy as np

from utils import *
from collections import deque
from numpy.random import choice
from torch import float32 as F32

class Episode:
    def __init__(self, device, bit_depth):
        self.device = device
        self.bit_depth = bit_depth
        self.clear()

    @property
    def size(self):
        return self._size

    def clear(self):
        self.x = []
        self.u = []
        self.d = []
        self.r = []
        self._size = 0

    def append(self, x, u, r, d):
        self._size += 1
        self.x.append(postprocess_img(x.numpy(), self.bit_depth))
        self.u.append(u.numpy())
        self.r.append(r)
        self.d.append(d)

    def append_last_obs(self, x):
        self.x.append(postprocess_img(x.numpy(), self.bit_depth))

    def prepare(self, s=0, e=None):
        e = e or self.size
        prossx = torch.tensor(self.x[s:e+1], dtype=F32, device=self.device)
        preprocess_img(prossx, self.bit_depth),
        return (
            prossx,
            torch.tensor(self.u[s:e], dtype=F32, device=self.device),
            torch.tensor(self.r[s:e], dtype=F32, device=self.device),
            torch.tensor(self.d[s:e], dtype=F32, device=self.device),
        )


class Memory:
    def __init__(self, size, device, tracelen):
        self.device = device
        self._shapes = None
        self.tracelen = tracelen
        self.data = deque(maxlen=size)

    @property
    def size(self):
        return len(self.data)

    @property
    def shapes(self):
        return self._shapes

    def get_empty_batch(self, batch_size):
        data = []
        for i, s in enumerate(self.shapes):
            h = self.tracelen + 1 if not i else self.tracelen
            data.append(torch.zeros(batch_size, h, *s).to(self.device))
        return data
    
    def append(self, episode: Episode):
        self.data.append(episode)
        if self.shapes is None:
            self._shapes = [a.shape[1:] for a in episode.prepare(e=1)]

    def sample(self, batch_size):
        episode_idx = choice(self.size, batch_size)
        init_st_idx = [choice(self.data[i].size) for i in episode_idx]
        data = self.get_empty_batch(batch_size)
        xx, uu, rr, dd = [], [], [], []
        for n, (i, s) in enumerate(zip(episode_idx, init_st_idx)):
            x, u, r, d = self.data[i].prepare(s, s + self.tracelen)
            xx.append(x)
            uu.append(u)
            rr.append(r)
            dd.append(d)
        return [torch.nn.utils.rnn.pad_sequence(i) for i in [xx, uu, rr, dd]]