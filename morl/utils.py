import os, sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'externals/baselines'))
sys.path.append(os.path.join(base_dir, 'externals/pytorch-a2c-ppo-acktr-gail'))

import numpy as np
from copy import deepcopy
from hypervolume import InnerHyperVolume

def print_error(*message):
    print('\033[91m', 'ERROR ', *message, '\033[0m')
    raise RuntimeError

def print_ok(*message):
    print('\033[92m', *message, '\033[0m')

def print_warning(*message):
    print('\033[93m', *message, '\033[0m')

def print_info(*message):
    print('\033[96m', *message, '\033[0m')
    
def check_dominated(obj_batch, obj):
    return (np.logical_and(
                (obj_batch >= obj).all(axis=1), 
                (obj_batch > obj).any(axis=1))
            ).any()
            
# return sorted indices of nondominated objs
def get_ep_indices(obj_batch_input):
    if len(obj_batch_input) == 0: return np.array([])
    obj_batch = np.array(obj_batch_input)
    sorted_indices = np.argsort(obj_batch.T[0])
    ep_indices = []
    for idx in sorted_indices:
        if (obj_batch[idx] >= 0).all() and not check_dominated(obj_batch, obj_batch[idx]):
            ep_indices.append(idx)
    return ep_indices

# update ep with a new point
def update_ep(ep_objs_batch, new_objs):
    if (new_objs < 0).any():
        return deepcopy(ep_objs_batch)
    new_ep_objs_batch = []
    on_ep = True
    for i in range(len(ep_objs_batch)):
        dominated = False
        if (new_objs >= ep_objs_batch[i]).all():
            dominated = True
        if (ep_objs_batch[i] >= new_objs - 1e-5).all() and (ep_objs_batch[i] > new_objs + 1e-5).any():
            on_ep = False
        if not dominated:
            new_ep_objs_batch.append(deepcopy(ep_objs_batch[i]))
    if on_ep:
        inserted = False
        for i in range(len(new_ep_objs_batch)): # gaurantee the new ep objs is still in order with first objective
            if new_objs[0] < new_ep_objs_batch[i][0]:
                new_ep_objs_batch.insert(i, deepcopy(new_objs))
                inserted = True
                break
        if not inserted:
            new_ep_objs_batch.append(deepcopy(new_objs))
        
    return new_ep_objs_batch

def generate_weights_batch_dfs(i, obj_num, min_weight, max_weight, delta_weight, weight, weights_batch):
    if i == obj_num - 1:
        weight.append(1.0 - np.sum(weight[0:i]))
        weights_batch.append(deepcopy(weight))
        weight = weight[0:i]
        return
    w = min_weight
    while w < max_weight + 0.5 * delta_weight and np.sum(weight[0:i]) + w < 1.0 + 0.5 * delta_weight:
        weight.append(w)
        generate_weights_batch_dfs(i + 1, obj_num, min_weight, max_weight, delta_weight, weight, weights_batch)
        weight = weight[0:i]
        w += delta_weight

# compute the hypervolume of a given pareto front
def compute_hypervolume(ep_objs_batch):
    n = len(ep_objs_batch[0])
    HV = InnerHyperVolume(np.zeros(n))
    return HV.compute(ep_objs_batch)

# compute the sparsity of a given pareto front
def compute_sparsity(ep_objs_batch):
    if len(ep_objs_batch) < 2:
        return 0.0

    sparsity = 0.0
    m = len(ep_objs_batch[0])
    ep_objs_batch_np = np.array(ep_objs_batch)
    for dim in range(m):
        objs_i = np.sort(deepcopy(ep_objs_batch_np.T[dim]))
        for i in range(1, len(objs_i)):
            sparsity += np.square(objs_i[i] - objs_i[i - 1])
    sparsity /= (len(ep_objs_batch) - 1)
    
    return sparsity

def update_ep_and_compute_hypervolume_sparsity(task_id, ep_objs_batch, new_objs, queue):
    new_ep_objs_batch = update_ep(ep_objs_batch, new_objs)
    hv = compute_hypervolume(new_ep_objs_batch)
    sparsity = compute_sparsity(new_ep_objs_batch)
    queue.put([task_id, hv, sparsity])

