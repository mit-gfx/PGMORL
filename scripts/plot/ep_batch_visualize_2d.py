'''
Visualize the pareto fronts of all runs (different seeds) of one algorithm for one problem.
'''
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
np.set_printoptions(precision=1)

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

    ref_x, ref_y = ref_point # set referent point as (0, 0)
    x, hypervolume = ref_x, 0.0
    sparsity = 0.0
    for i in range(len(objs)):
        hypervolume += (max(ref_x, objs[i][0]) - x) * (max(ref_y, objs[i][1]) - ref_y)
        x = max(ref_x, objs[i][0])
        if i > 0:
            sparsity += np.sum(np.square(objs[i] - objs[i - 1]))
    
    if len(objs) == 1:
        sparsity = 0.0
    else:
        sparsity = sparsity / (len(objs) - 1)

    return hypervolume, sparsity

parser = argparse.ArgumentParser()
parser.add_argument('--log-dir', type=str, required=True)
parser.add_argument('--save-fig', default=False, action='store_true')
parser.add_argument('--title', type=str, default=None)
parser.add_argument('--obj', type=str, nargs='+', default=None)
parser.add_argument('--num-seeds', type=int, default=6)
parser.add_argument('--ref-point', type=float, nargs='+', default=[0., 0.])
args = parser.parse_args()

fig, ax = plt.subplots(1, 1)
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

    hypervolume, sparsity = compute_hypervolume_sparsity(np.array(objs), args.ref_point)

    print('hypervolume = {:.0f}, sparsity = {:.0f}'.format(hypervolume, sparsity))
    legends.append('{}, H = {:.0f}, S = {:.0f}'.format(ii, hypervolume, sparsity))

    hv_mean += hypervolume / args.num_seeds
    sp_mean += sparsity / args.num_seeds

print('mean hypervolume = {:.0f}, mean sparsity = {:.0f}'.format(hv_mean, sp_mean))

if args.obj is not None:
    ax.set_xlabel(args.obj[0])
    ax.set_ylabel(args.obj[1])

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