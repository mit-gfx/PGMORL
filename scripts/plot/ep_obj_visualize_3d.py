'''
Visualize the computed Pareto policies in the performance space. 
By double-click the point in the plot, it will automatically open a new window and render the simulation for the selected policy.
'''
import os, sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'morl/'))

import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backend_bases import MouseButton
import numpy as np
import os
from re import split
from copy import deepcopy
from multiprocessing import Process
np.set_printoptions(precision=1)
from hypervolume import InnerHyperVolume

objs = []
model_file_list = []
env_params_file_list = []
fig = plt.figure()
ax = Axes3D(fig, azim=45)

def find_nearest_point(arr_points, point):
    idx = np.square(np.array(arr_points) - np.array(point)).sum(axis=-1).argmin()
    return idx

def on_click(event):
    if event.inaxes != ax: return
    if event.button != MouseButton.LEFT or not event.dblclick: return
    b = ax.button_pressed
    ax.button_pressed = -1
    xyz = ax.format_coord(event.xdata, event.ydata)
    ax.button_pressed = b
    xyz = list(map(lambda x: float(x), split(',|=', xyz.replace(' ', ''))[1::2]))
    idx = find_nearest_point(objs, xyz)
    model_file = model_file_list[idx]
    env_params_file = env_params_file_list[idx]
    play_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mujoco_play.py')
    play_cmd = 'python {} --env {} --model {} --env-params {}'.format(play_script, args.env, model_file, env_params_file)
    print('Objectives:', objs[idx])
    Process(target=os.system, args=(play_cmd,)).start()

fig.canvas.mpl_connect('button_press_event', on_click)

def compute_hypervolume_sparsity_3d(obj_batch, ref_point):
    HV = InnerHyperVolume(ref_point)
    hv = HV.compute(obj_batch)

    sparsity = 0.0
    m = len(obj_batch[0])
    for dim in range(m):
        objs_i = np.sort(deepcopy(obj_batch.T[dim]))
        for i in range(1, len(objs_i)):
            sparsity += np.square(objs_i[i] - objs_i[i - 1])
    sparsity /= (len(obj_batch) - 1)
    
    return hv, sparsity

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='MO-Hopper-v3')
parser.add_argument('--log-dir', type=str, required=True)
parser.add_argument('--fig-path', type=str, default=None)
parser.add_argument('--title', type=str, default=None)
parser.add_argument('--obj', type=str, nargs='+', default=None)
parser.add_argument('--ref-point', type=float, nargs='+', default=[0., 0., 0.])

args = parser.parse_args()

if args.env == 'MO-Hopper-v3':
    args.obj = ['Running Speed', 'Jumping Height', 'Energy Efficiency']

args.log_path = os.path.join(args.log_dir, 'final', 'objs.txt')
with open(args.log_path, 'r') as fp:
    data = fp.readlines()
    rew_data = []
    for j, line_data in enumerate(data):
        line_data = line_data.split(',')
        line_data = list(map(lambda x: float(x), line_data))
        rew_data.append(line_data)
        model_file_list.append(os.path.join(os.path.dirname(args.log_path), 'EP_policy_' + str(j) + '.pt'))
        env_params_file = os.path.join(os.path.dirname(args.log_path), 'EP_env_params_' + str(j) + '.pkl')
        env_params_file_list.append(env_params_file)
    objs.extend(rew_data)
    rew_data = np.array(rew_data).T
    ax.scatter(*rew_data)

hypervolume, sparsity = compute_hypervolume_sparsity_3d(np.array(objs), args.ref_point)

print('hypervolume = {:.0f}, sparsity = {:.0f}'.format(hypervolume, sparsity))
legend = 'H = {:.0f}, S = {:.0f}'.format(hypervolume, sparsity)
ax.legend([legend], loc='lower left')

if args.obj is not None:
    ax.set_xlabel(args.obj[0])
    ax.set_ylabel(args.obj[1])
    ax.set_zlabel(args.obj[2])

if args.title is not None:
    plt.title(args.title)

if args.fig_path is None:
    plt.show()
else:
    os.makedirs(os.path.dirname(args.fig_path), exist_ok=True)
    plt.savefig(args.fig_path)
    plt.show()