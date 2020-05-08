import os
import sys
import pdb
import cv2
import gym
import torch
import pickle
import plotly
import pathlib
import numpy as np


from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line

from torchvision.utils import make_grid, save_image


def to_tensor_obs(image):
    """
    Converts the input np img to channel first 64x64 dim torch img.
    """
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    return image


def postprocess_img(image, depth):
    """
    Postprocess an image observation for storage.
    From float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
    """
    image = np.floor((image + 0.5) * 2 ** depth)
    return np.clip(image * 2**(8 - depth), 0, 2**8 - 1).astype(np.uint8)


def preprocess_img(image, depth):
    """
    Preprocesses an observation inplace.
    From float32 Tensor [0, 255] to [-0.5, 0.5]
    Also adds some noise to the observations !!
    """
    image.div_(2 ** (8 - depth)).floor_().div_(2 ** depth).sub_(0.5)
    image.add_(torch.randn_like(image).div_(2 ** depth)).clamp_(-0.5, 0.5)
    

def get_combined_params(*models):
    """
    Returns the combine parameter list of all the models given as input.
    """
    params = []
    for model in models:
        params.extend(list(model.parameters()))
    return params


def save_video(frames, path, name):
    """
    Saves a video containing frames.
    """
    frames = (frames*255).astype('uint8').transpose(0, 2, 3, 1)[..., ::-1]
    _, H, W, _ = frames.shape
    writer = cv2.VideoWriter(
        str(pathlib.Path(path)/f'{name}.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'), 25., (W, H), True
    )
    for frame in frames:
        writer.write(frame)
    writer.release()


def save_frames(target, pred_prior, pred_posterior, name, n_rows=5):
    """
    Saves the target images with the generated prior and posterior predictions. 
    """
    image = torch.cat([target, pred_prior, pred_posterior], dim=3)
    save_image(make_grid(image + 0.5, nrow=n_rows), f'{name}.png')


def get_mask(tensor, lengths):
    """
    Generates the masks for batches of sequences.
    Time should be the first axis.
    input:
        tensor: the tensor for which to generate the mask [N x T x ...]
        lengths: lengths of the seq. [N] 
    """
    mask = torch.zeros_like(tensor)
    for i in range(len(lengths)):
        mask[i, :lengths[i]] = 1.
    return mask


def load_memory(path, device):
    """
    Loads an experience replay buffer.
    """
    with open(path, 'rb') as f:
        memory = pickle.load(f)
        memory.device = device
        for e in memory.data:
            e.device = device
    return memory

def apply_model(model, inputs, ignore_dim=None):
    pass

def plot_metrics(metrics, path, prefix):
    for key, val in metrics.items():
        lineplot(np.arange(len(val)), val, f'{prefix}{key}', path)

def lineplot(xs, ys, title, path='', xaxis='episode'):
    MAX_LINE = Line(color='rgb(0, 132, 180)', dash='dash')
    MIN_LINE = Line(color='rgb(0, 132, 180)', dash='dash')
    NO_LINE = Line(color='rgba(0, 0, 0, 0)')
    MEAN_LINE = Line(color='rgb(0, 172, 237)')
    std_colour = 'rgba(29, 202, 255, 0.2)'
    if isinstance(ys, dict):
        data = []
        for key, val in ys.items():
            xs = np.arange(len(val))
            data.append(Scatter(x=xs, y=np.array(val), name=key))
    elif np.asarray(ys, dtype=np.float32).ndim == 2:
        ys = np.asarray(ys, dtype=np.float32)
        ys_mean, ys_std = ys.mean(-1), ys.std(-1)
        ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std
        l_max = Scatter(x=xs, y=ys.max(-1), line=MAX_LINE, name='Max')
        l_min = Scatter(x=xs, y=ys.min(-1), line=MIN_LINE, name='Min')
        l_stu = Scatter(x=xs, y=ys_upper, line=NO_LINE, showlegend=False)
        l_mean = Scatter(
            x=xs, y=ys_mean, fill='tonexty', fillcolor=std_colour,
            line=MEAN_LINE, name='Mean'
        )
        l_stl = Scatter(
            x=xs, y=ys_lower, fill='tonexty', fillcolor=std_colour,
            line=NO_LINE, name='-1 Std. Dev.', showlegend=False
        )
        data = [l_stu, l_mean, l_stl, l_min, l_max]
    else:
        data = [Scatter(x=xs, y=ys, line=MEAN_LINE)]
    plotly.offline.plot({
        'data': data,
        'layout': dict(
            title=title,
            xaxis={'title': xaxis},
            yaxis={'title': title}
            )
        }, filename=os.path.join(path, title + '.html'), auto_open=False
    )



class TorchImageEnvWrapper:
    """
    Torch Env Wrapper that wraps a gym env and makes interactions using Tensors.
    Also returns observations in image form.
    """
    def __init__(self, env, bit_depth, observation_shape=None, act_rep=2):
        self.env = gym.make(env)
        self.bit_depth = bit_depth
        self.action_repeats = act_rep

    def reset(self):
        self.env.reset()
        x = to_tensor_obs(self.env.render(mode='rgb_array'))
        preprocess_img(x, self.bit_depth)
        return x

    def step(self, u):
        u = u.cpu().detach().numpy()
        for _ in range(self.action_repeats - 1):
            self.env.step(u)
        _, r, d, i = self.env.step(u)
        x = to_tensor_obs(self.env.render(mode='rgb_array'))
        preprocess_img(x, self.bit_depth)
        return x, r, d, i

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    @property
    def observation_size(self):
        return (3, 64, 64)

    @property
    def action_size(self):
        return self.env.action_space.shape[0]

    def sample_random_action(self):
        return torch.tensor(self.env.action_space.sample())
