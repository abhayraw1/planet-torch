import sys
import pdb
import cv2
import gym
import torch
import numpy as np

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
    """
    image.div_(2 ** (8 - depth)).floor_().div_(2 ** depth).sub_(0.5)
    image.add_(torch.rand_like(image).div_(2 ** depth))


def get_combined_params(*models):
    """
    Returns the combine parameter list of all the models given as input.
    """
    params = []
    for model in models:
        params.extend(list(model.parameters()))
    return params


def save_frames(target, pred_prior, pred_posterior, name, n_rows=5):
    """
    Saves the target images with the generated prior and posterior predictions. 
    """
    image = torch.cat([target, pred_prior, pred_posterior], dim=3)
    save_image(make_grid(image + 0.5, nrow=n_rows), f'{name}.png')


def apply_model(model, inputs, ignore_dim=None):
    pass



class TorchImageEnvWrapper:
    """
    Torch Env Wrapper that wraps a gym env and makes interactions using Tensors.
    Also returns observations in image form.
    """
    def __init__(self, env, bit_depth, observation_shape=None):
        self.env = gym.make(env)
        self.bit_depth = bit_depth

    def reset(self):
        self.env.reset()
        x = to_tensor_obs(self.env.render(mode='rgb_array'))
        preprocess_img(x, self.bit_depth)
        return x

    def step(self, u):
        _, r, d, i = self.env.step(u.detach().numpy())
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
