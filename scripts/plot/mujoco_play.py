import os, sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'externals/pytorch-a2c-ppo-acktr-gail'))
sys.path.append(os.path.join(base_dir, 'externals/baselines/'))

import environments
from a2c_ppo_acktr.model import Policy
import torch
import gym
from gym import wrappers
import numpy as np
import argparse
import os
import pickle

from time import time

# define argparser
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='MO-HalfCheetah-v2')
parser.add_argument('--model', type = str)
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--fps', type = float, default = 60.0)
parser.add_argument('--record', action = 'store_true')
parser.add_argument('--video_filename', type = str)
parser.add_argument('--disable-render', default=False, action='store_true')
parser.add_argument('--layernorm', action = 'store_true', default=False)
parser.add_argument('--env-params', type=str, default=None)

# parse arguments
args = parser.parse_args()
if args.model:
    state_dict_path = args.model
else:
    print("[Error] Please indicate the trained model file")
    quit()
record_video = args.record
if record_video:
    if args.video_filename:
        record_video_filename = args.video_filename
    else:
        record_video_filename = 'video'

# main
env_name = args.env
save_path = os.path.dirname(state_dict_path)
device = 'cpu'
torch.set_default_dtype(torch.float64)

env = gym.make(env_name)
env.seed(args.seed)
if record_video:
    env = wrappers.Monitor(env, os.path.join(save_path, 'videos', record_video_filename), force = True)

policy = Policy(
    env.observation_space.shape,
    env.action_space,
    base_kwargs={'recurrent': False, 'layernorm' : args.layernorm},
    obj_num=env.obj_dim)

state_dict = torch.load(state_dict_path)
policy.load_state_dict(state_dict)
policy = policy.to(device)
policy.double()
policy.eval()

ob_rms = None
if args.env_params is not None and os.path.exists(args.env_params):
    with open(args.env_params, 'rb') as fp:
        env_params = pickle.load(fp)
    ob_rms = env_params['ob_rms']

while True:
    obs = env.reset()
    obj = np.zeros(env.obj_dim)
    t = time()
    done = False
    iter = 0
    while not done:
        if ob_rms is not None:
            obs = np.clip((obs - ob_rms.mean) / np.sqrt(ob_rms.var + 1e-8), -10.0, 10.0)
        _, action, _, _ = policy.act(torch.Tensor(obs).unsqueeze(0), None, None, deterministic=True)
        obs, _, done, info = env.step(action.detach().numpy())
        obj += info['obj']
        while time() - t < 1 / args.fps:
            pass
        if not args.disable_render:
            env.render()
        t = time()
        iter += 1
    print('iter = ', iter)
    print('objective = ', obj)

env.close()


