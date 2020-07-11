'''
Visualize the computed Pareto policies in the performance space. 
By double-click the point in the plot, it will automatically open a new window and render the simulation for the selected policy.
'''
import argparse
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import numpy as np
import os
from multiprocessing import Process
np.set_printoptions(precision=1)

objs = []
model_file_list = []
env_params_file_list = []
fig, ax = plt.subplots(1, 1)

def find_nearest_point(arr_points, point):
    idx = np.square(np.array(arr_points) - np.array(point)).sum(axis=-1).argmin()
    return idx

def on_click(event):
    if event.inaxes != ax: return
    if event.button != MouseButton.LEFT or not event.dblclick: return
    ix, iy = event.xdata, event.ydata
    idx = find_nearest_point(objs, [ix, iy])
    model_file = model_file_list[idx]
    env_params_file = env_params_file_list[idx]
    play_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mujoco_play.py')
    play_cmd = 'python {} --env {} --model {} --env-params {}'.format(play_script, args.env, model_file, env_params_file)
    print('Objectives:', objs[idx])
    Process(target=os.system, args=(play_cmd,)).start()

fig.canvas.mpl_connect('button_press_event', on_click)

def get_ep_indices(obj_batch):
    # return sorted indices of undominated objs
    if len(obj_batch) == 0: return np.array([])
    sort_indices = np.lexsort((obj_batch.T[1], obj_batch.T[0]))
    ep_indices = []
    max_val = -np.inf
    for idx in sort_indices[::-1]:
        if obj_batch[idx][1] > max_val:
            max_val = obj_batch[idx][1]
            ep_indices.append(idx)
    return ep_indices[::-1]

# compute the hypervolume given the pareto points, only for 2-dim objectives now
def compute_hypervolume_sparsity(obj_batch, ref_point):
    if obj_batch.shape[1] != 2:
        return 0
    objs = obj_batch[get_ep_indices(obj_batch)]

    objs = np.array(objs)
    print('input size : {}, pareto size : {}'.format(len(obj_batch), len(objs)))

    x, hypervolume = ref_point[0], 0.0
    sparsity = 0.0
    for i in range(len(objs)):
        hypervolume += (max(ref_point[0], objs[i][0]) - x) * (max(ref_point[1], objs[i][1]) - ref_point[1])
        x = max(ref_point[0], objs[i][0])
        if i > 0:
            sparsity += np.sum(np.square(objs[i] - objs[i - 1]))
    if len(objs) > 1:
        sparsity /= len(objs) - 1
        
    return hypervolume, sparsity

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='MO-HalfCheetah-v2')
parser.add_argument('--log-dir', type=str, required=True)
parser.add_argument('--fig-path', type=str, default=None)
parser.add_argument('--title', type=str, default=None)
parser.add_argument('--obj', type=str, nargs='+', default=None)
parser.add_argument('--ref-point', type=float, nargs='+', default=[0., 0.])

args = parser.parse_args()

if args.env in ['MO-HalfCheetah-v2', 'MO-Walker2d-v2', 'MO-Swimmer-v2', 'MO-Humanoid-v2']:
    args.obj = ['Forward Speed', 'Energy Efficiency']
elif args.env == 'MO-Ant-v2':
    args.obj = ['X-Axis Speed', 'Y-Axis Speed']
elif args.env == 'MO-Hopper-v2':
    args.obj = ['Running Speed', 'Jumping Height']

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

hypervolume, sparsity = compute_hypervolume_sparsity(np.array(objs), args.ref_point)

print('hypervolume = {:.0f}, sparsity = {:.0f}'.format(hypervolume, sparsity))

legend = 'H = {:.0f}, S = {:.0f}'.format(hypervolume, sparsity)
ax.legend([legend], loc='lower left')

if args.obj is not None:
    ax.set_xlabel(args.obj[0])
    ax.set_ylabel(args.obj[1])

if args.title is not None:
    plt.title(args.title)

if args.fig_path is None:
    plt.show()
else:
    os.makedirs(os.path.dirname(args.fig_path), exist_ok=True)
    plt.savefig(args.fig_path)
    plt.show()