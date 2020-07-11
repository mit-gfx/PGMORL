'''
Visualize the pareto fronts of all runs (different seeds) of one algorithm for one problem.
'''

import os, sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'morl/'))

import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from copy import deepcopy
np.set_printoptions(precision=1)
from hypervolume import InnerHyperVolume

def compute_hypervolume_sparsity_3d(obj_batch, ref_point):
    HV = InnerHyperVolume(ref_point)
    hv = HV.compute(obj_batch)

    sparsity = 0.0
    m = len(obj_batch[0])
    for dim in range(m):
        objs_i = np.sort(deepcopy(obj_batch.T[dim]))
        for i in range(1, len(objs_i)):
            sparsity += np.square(objs_i[i] - objs_i[i - 1])
    if len(obj_batch) == 1:
        sparsity = 0.0
    else:
        sparsity /= (len(obj_batch) - 1)
    
    return hv, sparsity

parser = argparse.ArgumentParser()
parser.add_argument('--log-dir', type=str, required=True)
parser.add_argument('--save-fig', default=False, action='store_true')
parser.add_argument('--title', type=str, default=None)
parser.add_argument('--obj', type=str, nargs='+', default=None)
parser.add_argument('--num-seeds', type=int, default=6)
parser.add_argument('--ref-point', type=float, nargs='+', default=[0., 0., 0.])
args = parser.parse_args()

fig = plt.figure()
ax = Axes3D(fig, azim=45)
legends = []
hv_mean, sp_mean = 0, 0

for ii in range(args.num_seeds):
    objs_file = 'objs.txt'
    log_path = os.path.join(args.log_dir, str(ii), 'final', objs_file)
    if not os.path.exists(log_path): log_path = os.path.join(args.log_dir, str(ii), 'test', objs_file)
    with open(log_path, 'r') as fp:
        data = fp.readlines()
        rew_data = []
        for j, line_data in enumerate(data):
            line_data = line_data.split(',')
            line_data = list(map(lambda x: float(x), line_data))
            rew_data.append(line_data)
        objs = rew_data.copy()
        rew_data = np.array(rew_data).T
        ax.scatter(*rew_data)

    hypervolume, sparsity = compute_hypervolume_sparsity_3d(np.array(objs), args.ref_point)

    print('hypervolume = {:.0f}, sparsity = {:.0f}'.format(hypervolume, sparsity))

    legends.append('{}, H = {:.0f}, S = {:.0f}'.format(ii, hypervolume, sparsity))

    hv_mean += hypervolume / args.num_seeds
    sp_mean += sparsity / args.num_seeds

print('mean hypervolume = {:.0f}, mean sparsity = {:.0f}'.format(hv_mean, sp_mean))

if args.obj is not None:
    ax.set_xlabel(args.obj[0])
    ax.set_ylabel(args.obj[1])
    ax.set_zlabel(args.obj[2])

if args.title is not None:
    plt.title(args.title)

plt.legend(legends, loc='lower left')

if not args.save_fig:
    plt.show()
else:
    fig_path = os.path.join(args.log_dir, 'ep_fig.png')
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path)
    plt.show()