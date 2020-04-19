import os
import cv2
import sys
import gym
import pdb
import yaml
import time
import enum
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from numpy import pi
from copy import deepcopy
from gym.spaces import Dict, Box
from numpy.linalg import norm, inv as inverse
from shapely.geometry import MultiPoint, Point
from matplotlib.patches import Polygon, Circle, FancyArrowPatch

from payload_manipulation.utils.utils import *
from payload_manipulation.utils.visualizer import Visualizer
from payload_manipulation.utils.payload_boundaries import *
from payload_manipulation.utils.transformations import compose_matrix
from payload_manipulation.utils.payload_transform import PayloadTransform


# matplotlib.use('Agg')
matplotlib.rcParams['toolbar'] = 'None'

def cossin(x):
    return np.cos(x), np.sin(x)

class StateType(enum.Enum):
    RADIANS = 1
    COS_SIN = 2
    SIX_DIM = 3


class ColorScheme:
    def __init__(self, payload=None, obstacles=None, freespace=None, goal=None):
        goal = goal if goal is not None else (231, 76, 60)
        payload = payload if payload is not None else (231, 76, 60)
        obstacles = obstacles if obstacles is not None else (52, 152, 219)
        freespace = freespace if freespace is not None else (96, 174, 39)
        self.payload = self.pld = tuple(map(int, payload))
        self.obstacles = self.obs = tuple(map(int, obstacles))
        self.freespace = self.fsp = tuple(map(int, freespace))
        self.goal = self.gol = tuple(map(int, goal))


OBS_STATE_SPACE = {
    StateType.RADIANS: 3, StateType.COS_SIN: 8, StateType.SIX_DIM: 8
}

class PayloadEnv(gym.Env):
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.config = config
        self.__dict__.update(config)
        self.dt = config.get('dt', 0.1)
        self.n_obs = config.get('n_obs', 15)
        self.goal_state = np.zeros(2)
        self.payload_dims = config.get('payload_dims', np.array([4., 3., 0.]))
        self.obstacle_radius = config.get('obstacle_radius', 0.3)
        self.payload_b = np.array([200, 200, np.pi/4, np.pi/4, np.pi, np.pi])
        self.transform = PayloadTransform(dims=self.payload_dims)
        self.mean_height = 1.0
        self.epsr = 7.5
        self.arena_dims = config.get('arena_dims', np.array([self.epsr]*2))
        self.twoD = config.get('twoD', True)
        self.color_scheme = config.get('color_scheme', ColorScheme())
        # for spinup algos... should have no effect
        self.init_vx()
        self.set_image_size(config.get('image_size', np.array([192, 192])))
        self.state_type = config.get('state_type', StateType.RADIANS)
        assert isinstance(self.state_type, StateType)
        self.to_render = config.get('to_render', True)
        self.fig = None
        self.prev_state = None
        self.curr_img = None
        self.alpha = 0.7
        self.a, self.b, _ = self.payload_dims/2 + self.obstacle_radius*1.35
        self.action_space = Box(
            np.array([-0.3, -0.3, -np.pi/6, -np.pi/6, -np.pi/4]),
            np.array([+0.3, +0.3, +np.pi/6, +np.pi/6, +np.pi/4])
        )
        self.observation_space = Box(
            -np.ones(OBS_STATE_SPACE[self.state_type]),
            np.ones(OBS_STATE_SPACE[self.state_type])
        )


    def init_plot(self):
        self.fig = plt.figure(figsize=self.image_size/100)
        self.axs = plt.axes([0,0,1,1], frameon=False)
        plt.ion()
        self.im = plt.imshow(np.zeros(self.image_size), interpolation='none')
        self.axs.set_xticks([])
        self.axs.set_yticks([])
        plt.show()


    def render(self, mode='human', forced=False):
        if not self.to_render:
            return
        if self.fig is None:
            self.init_plot()
        if self.curr_img is None:
            raise ValueError('Can\'t render without reset()')
        if forced:
            self.get_observation()
        self.im.set_array(self.curr_img/255)
        plt.gcf().canvas.draw_idle()
        plt.gcf().canvas.start_event_loop(0.0001)
        if mode == 'human':
            return
        return self.curr_img


    def get_random_state(self, no_collide=[], r=1):
        while True:
            x = (2*np.random.random(2) - 1.0)*(self.arena_dims + 2**.5)
            aex = abs(self.to_ego_frame(x))
            # check if obstacle is inside payload (don't want that)
            if all(aex < self.payload_dims[:2]/2 + self.obstacle_radius*2):
                # print('Fail 1')
                continue
            # check if obstacle is outside the eps neighbourhood
            # Can go super strict by using any() in place of all()
            if all(aex > self.epsr):
                # print('Fail 2')
                continue
            break
        return x


    def reset(self, total_chaos=False):
        self.agent_state = np.zeros(5)
        self.agent_state[-1] = (2*np.random.random() - 1)*np.pi
        self.obstacle_state = np.zeros((self.n_obs, 2))
        elems = []
        for i in range(self.n_obs):
            self.obstacle_state[i] = self.get_random_state()
            elems.append(self.obstacle_state[i])
        r = 0.1 if total_chaos else max(self.payload_dims)
        self.goal_state = self.get_random_state()
        self.prev_action = np.zeros_like(self.action_space.sample())
        self.success = False
        return self.get_observation()


    def init_vx(self):
        self.vxs = []
        pseudo_vxs = np.array([
            [1, 1, 1], [1, 1, -1], [-1, 1, -1], [-1, 1, 1],
            [-1, -1, 1], [-1, -1, -1], [1, -1, -1], [1, -1, 1]
        ])
        if self.twoD:
            pseudo_vxs = pseudo_vxs[range(0, 8, 2)]
        for vx in pseudo_vxs:
            self.vxs.append(self.payload_dims/2*vx)
        self.vxs = np.stack(self.vxs).transpose(0, 1)

    @property    
    def vertices(self):
        x, r = np.split(self.agent_state, [2])
        x = np.concatenate([x, np.ones(1)*self.mean_height])
        return np.matmul(
            compose_matrix(translate=x, angles=r),
            np.concatenate([self.vxs, np.ones((4, 1))], axis=-1).T
        )[:-1].T

    def to_ego_frame(self, pts):
        if len(pts.shape) == 1:
            return self.to_ego_frame(pts[None]).flatten()
        pts = np.concatenate([pts.T, OIS(pts.shape[0])])
        x, y, _, _, r = self.agent_state
        tf = compose_matrix(translate=[x, y, 0], angles=[0, 0, r])
        return np.matmul(np.linalg.inv(tf), pts)[:-2].T

    def from_ego_frame(self, pts):
        if len(pts.shape) == 1:
            return self.from_ego_frame(pts[None]).flatten()
        pts = np.concatenate([pts.T, OIS(pts.shape[0])])
        x, y, _, _, r = self.agent_state
        tf = compose_matrix(translate=[x, y, 0], angles=[0, 0, r])
        return np.matmul(tf, pts)[:-2].T

    def get_observation(self):
        obs = np.ones((*self.image_size, 3))
        obs = (obs*self.color_scheme.fsp).astype(np.uint8)
        for obstacle_xy in self.obstacle_state:
            center = self.to_ego_frame(obstacle_xy)
            if (abs(center) > self.epsr).all():
                continue
            center = ((center*[1, -1])*self.scale + self.image_size/2)
            cv2.circle(
                obs, tuple(map(int, center)), self.radius_px,
                self.color_scheme.obs, -1
            )
        center = self.to_ego_frame(self.goal_state)
        center = ((center*[1, -1])*self.scale + self.image_size/2).astype('i')
        cv2.circle(
            obs, tuple(center), self.radius_px, self.color_scheme.gol, -1
        )
        vxs = self.to_ego_frame(self.vertices[:, :-1])*[[1, -1]]
        vxs = (vxs*self.scale + self.image_size/2).astype('i')
        pld = np.copy(obs)
        cv2.fillConvexPoly(pld, vxs, self.color_scheme.pld)
        # alpha = 0.5
        cv2.addWeighted(pld, self.alpha, obs, 1 - self.alpha, 0, obs)
        self.curr_img = np.copy(obs)
        state = np.copy(self.agent_state)
        state[:2] = self.to_ego_frame(self.goal_state)
        if self.state_type == StateType.COS_SIN:
            state = np.concatenate(
                [state[:2]] + list(zip(*cossin(self.agent_state[2:])))
            )
        if self.state_type == StateType.SIX_DIM:
            state = np.concatenate([
                state[:2],
                compose_matrix(angles=self.agent_state[2:])[:3, :2].flatten()
            ])
        return self.curr_img, state


    def get_reward_and_done(self, action, info):
        pld = MultiPoint(self.to_ego_frame(self.vertices[:, :-1])).convex_hull
        reward = 0
        done, success = False, False
        d2g = self.to_ego_frame(self.goal_state)

        if any(abs(d2g) > self.epsr):
            return -1, False, True

        for obstacle_xy in self.to_ego_frame(self.obstacle_state):
            if any(obstacle_xy > self.epsr):
                continue
            obstacle = Point(*obstacle_xy).buffer(self.obstacle_radius)
            collision = obstacle.intersects(pld)
            if collision:
                print('COLLISION!')
                return -1, False, True
        # r1 = -norm(self.agent_state[2:4])
        r2 = norm(self.prev_state[:2] - self.goal_state)\
            - norm(self.agent_state[:2] - self.goal_state)
        # r3 = -norm(self.prev_action - action)
        # reward = 0.05*r1 + r2 + 0.1*r3 + (-0.1*self.dt) # step_penalty
        reward = r2 - 0.1*self.dt
        if norm(d2g) < 0.3:
            reward = 1
            success = True
        info['d2g'] = norm(d2g)
        return reward, success, done or success

    def step(self, controls):
        self.prev_state = np.copy(self.agent_state)
        if len(controls.shape) == 1:
            controls = controls[None]
        controls = controls.clip(self.action_space.low, self.action_space.high)
        for control in controls:
            self.agent_state[0] += control[0]*np.cos(self.agent_state[-1])
            self.agent_state[0] -= control[1]*np.sin(self.agent_state[-1])
            self.agent_state[1] += control[0]*np.sin(self.agent_state[-1])
            self.agent_state[1] += control[1]*np.cos(self.agent_state[-1])
            self.agent_state[2:] += control[2:]*self.dt
            self.agent_state[2:4] = self.agent_state[2:4].clip(-pi/4, pi/4)
            self.agent_state[-1] = mod_angle(self.agent_state[-1])
        obs = self.get_observation()
        info = {}
        reward, success, done = self.get_reward_and_done(controls, info)
        self.prev_action = np.copy(controls)
        self.success = success
        info['success'] = success
        return obs, reward, done, info

    def set_rendering(self, enable):
        self.to_render = enable

    def set_image_size(self, image_size):
        assert image_size.size == 2
        self.image_size = image_size
        self.scale = self.image_size/(2*self.epsr)
        self.radius_px = int((self.obstacle_radius*self.scale).mean()*1.5)

    def close(self):
        plt.close(self.fig)
        super().close()



def register_with_config(env_name, config=None):
    config = config or {}
    gym.envs.registration.register(
        id=env_name,
        entry_point=PayloadEnv,
        kwargs={'config': config}
    )

env_config = {
    'n_obs': 8,
    'state_type': StateType.SIX_DIM,
    'color_scheme': ColorScheme(*(np.eye(3)*255), 255*np.ones(3)),
}
register_with_config('PayloadEnvG2G-v0', env_config)

if __name__ == '__main__':
    set_seed(55723)
    env = gym.make('PayloadEnvG2G-v0')

    obs = env.reset(total_chaos=False)
    env.goal_state = env.agent_state[:2]
    # print('Goal RN: ', env.goal_state)
    env.to_ego_frame(env.goal_state)
    t = time.time()
    d = False
    for i in range(300):
        env.render(forced=True)
        pdb.set_trace()
        if d:
            break
        # u = np.array([1, 0, 0, 0, 0])
        u = env.action_space.sample()
        u = env.action_space.high
        print(env.agent_state)
        obs, r, d, ii = env.step(u)
        time.sleep(.1)
        # print(obs[0].shape)
        # input()
    print(f'Time taken: {(time.time() - t)/300}')
