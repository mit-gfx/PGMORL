import numpy as np
import torch
import torch.optim as optim
from copy import deepcopy
from sample import Sample
from utils import get_ep_indices, generate_weights_batch_dfs, update_ep, compute_hypervolume, compute_sparsity, update_ep_and_compute_hypervolume_sparsity
from scipy.optimize import least_squares
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.multiprocessing import Process, Queue, Event
from hypervolume import InnerHyperVolume

def collect_nearest_data(opt_graph, index, threshold = 0.1):
    objs_data, weights_data, delta_objs_data = [], [], []
    for i in range(len(opt_graph.objs)):
        diff = np.abs(opt_graph.objs[index] - opt_graph.objs[i])
        if np.all(diff < np.abs(opt_graph.objs[index]) * threshold):
            for next_index in opt_graph.succ[i]:
                objs_data.append(opt_graph.objs[i])
                weights_data.append(opt_graph.weights[next_index] / np.sum(opt_graph.weights[next_index]))
                delta_objs_data.append(opt_graph.delta_objs[next_index])
    return objs_data, weights_data, delta_objs_data

def predict_hyperbolic(args, opt_graph, index, test_weights):
    test_weights = np.array(test_weights)

    for test_weight in test_weights:
        test_weight /= np.sum(test_weight)
    
    threshold = 0.1
    sigma = 0.03
    # gradually enlarging the searching range so that get enough data point to fit the model
    while True:
        objs_data, weights_data, delta_objs_data = collect_nearest_data(opt_graph, index, threshold)
        cnt_data = 0
        for i in range(len(weights_data)):
            flag = True
            for j in range(i):
                if np.linalg.norm(weights_data[i] - weights_data[j]) < 1e-5:
                    flag = False
                    break
            if flag:
                cnt_data += 1
                if cnt_data > 3:
                    break
        if cnt_data > 3 or threshold >= 1.0:
            break
        else:
            threshold *= 2.0
            sigma *= 2.0

    def f(x, A, a, b, c):
        return A * (np.exp(a * (x - b)) - 1) / (np.exp(a * (x - b)) + 1) + c

    def fun(params, x, y):
        # f = A * (exp(a(x - b)) - 1) / (exp(a(x - b)) + 1) + c
        return (params[0] * (np.exp(params[1] * (x - params[2])) - 1.) / (np.exp(params[1] * (x - params[2])) + 1) + params[3] - y) * w

    def jac(params, x, y):
        A, a, b, c = params[0], params[1], params[2], params[3]

        J = np.zeros([len(params), len(x)])

        # df_dA = (exp(a(x - b)) - 1) / (exp(a(x - b)) + 1)
        J[0] = ((np.exp(a * (x - b)) - 1) / (np.exp(a * (x - b)) + 1)) * w

        # df_da = A(x - b)(2exp(a(x-b)))/(exp(a(x-b)) + 1)^2
        J[1] = (A * (x - b) * (2. * np.exp(a * (x - b))) / ((np.exp(a * (x - b)) + 1) ** 2)) * w

        # df_db = A(-a)(2exp(a(x-b)))/(exp(a(x-b)) + 1)^2
        J[2] = (A * (-a) * (2. * np.exp(a * (x - b))) / ((np.exp(a * (x - b)) + 1) ** 2)) * w

        # df_dc = 1
        J[3] = w

        return np.transpose(J)

    M = args.obj_num
    delta_predictions = []
    for dim in range(M):
        train_x = []
        train_y = []
        w = []
        for i in range(len(objs_data)):
            train_x.append(weights_data[i][dim])
            train_y.append(delta_objs_data[i][dim])
            diff = np.abs(objs_data[i] - opt_graph.objs[index])
            dist = np.linalg.norm(diff / np.abs(opt_graph.objs[index]))
            coef = np.exp(-((dist  / sigma) ** 2) / 2.0)
            w.append(coef)
        
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        w = np.array(w)

        A_upperbound = np.clip(np.max(train_y) - np.min(train_y), 1.0, 500.0)
        params0 = np.ones(4)
        
        f_scale = 20.

        res_robust = least_squares(fun, params0, loss='soft_l1', f_scale = f_scale, args = (train_x, train_y), jac = jac, bounds = ([0, 0.1, -5., -500.], [A_upperbound, 20., 5., 500.]))
        
        delta_predictions.append(f(test_weights.T[dim], *res_robust.x))

    predictions = []
    delta_predictions = np.transpose(np.array(delta_predictions))
    original_objs = opt_graph.objs[index]
    for i in range(len(test_weights)):
        predictions.append(original_objs + delta_predictions[i])

    results = {'sample_index': index, 'predictions': predictions}

    return results

class Population:
    def __init__(self, args):
        self.sample_batch = [] # all samples in population
        self.pbuffer_size = args.pbuffer_size
        self.obj_num = args.obj_num
        self.z_min = np.zeros(args.obj_num) # left-lower reference point
        self.pbuffer_vec = []
        generate_weights_batch_dfs(0, args.obj_num, 0.0, 1.0, 1.0 / (args.pbuffer_num - 1), [], self.pbuffer_vec)
        for i in range(len(self.pbuffer_vec)):
            self.pbuffer_vec[i] = self.pbuffer_vec[i] / np.linalg.norm(self.pbuffer_vec[i])
        self.pbuffer_num = len(self.pbuffer_vec)
        self.pbuffers = [[] for _ in range(self.pbuffer_num)] # store the sample indices in each pbuffer
        self.pbuffer_dist = [[] for _ in range(self.pbuffer_num)] # store the sample distance in each pbuffer

    def find_buffer_id(self, f):
        max_dot, buffer_id = -np.inf, -1
        for i in range(self.pbuffer_num):
            dot = np.dot(self.pbuffer_vec[i], f)
            if dot > max_dot:
                max_dot, buffer_id = dot, i
        return buffer_id

    def insert_pbuffer(self, index, objs, enforce):
        f = objs - self.z_min
        if np.min(f) < 1e-7:
            return False

        dist = np.linalg.norm(f)
        buffer_id = self.find_buffer_id(f)

        inserted = False
        if enforce:
            for i in range(len(self.pbuffers[buffer_id])):
                if self.pbuffer_dist[buffer_id][i] < dist:
                    self.pbuffers[buffer_id].insert(i, index)
                    self.pbuffer_dist[buffer_id].insert(i, dist)
                    inserted = True
                    break
            if not inserted:
                self.pbuffers[buffer_id].append(index)
                self.pbuffer_dist[buffer_id].append(dist)
            inserted = True
        else:
            for i in range(len(self.pbuffers[buffer_id])):
                if self.pbuffer_dist[buffer_id][i] < dist:
                    self.pbuffers[buffer_id].insert(i, index)
                    self.pbuffer_dist[buffer_id].insert(i, dist)
                    inserted = True
                    break
            if inserted and len(self.pbuffers[buffer_id]) > self.pbuffer_size:
                self.pbuffers[buffer_id] = self.pbuffers[buffer_id][:self.pbuffer_size]
                self.pbuffer_dist[buffer_id] = self.pbuffer_dist[buffer_id][:self.pbuffer_size]
            elif (not inserted) and len(self.pbuffers[buffer_id]) < self.pbuffer_size:
                self.pbuffers[buffer_id].append(index)
                self.pbuffer_dist[buffer_id].append(dist)
                inserted = True
        
        return inserted
                        
    def update(self, sample_batch):
        ### population = Union(population, offspring) ###
        all_sample_batch = self.sample_batch + sample_batch

        self.sample_batch = []
        self.pbuffers = [[] for _ in range(self.pbuffer_num)]
        self.pbuffer_dist = [[] for _ in range(self.pbuffer_num)]
        
        ### select population by performance buffer ###
        for i, sample in enumerate(all_sample_batch):
            self.insert_pbuffer(i, sample.objs, False)
        
        for pbuffer in self.pbuffers:
            for index in pbuffer:
                self.sample_batch.append(all_sample_batch[index])
    
    def evaluate_hv(self, candidates, mask, virtual_ep_objs_batch):
        hv = [0.0 for _ in range(len(candidates))]
        for i in range(len(candidates)):
            if mask[i]:
                new_ep_objs_batch = update_ep(virtual_ep_objs_batch, candidates[i]['prediction'])
                hv[i] = compute_hypervolume(new_ep_objs_batch)
        return hv

    def evaluate_sparsity(self, candidates, mask, virtual_ep_objs_batch):
        sparsity = [0.0 for _ in range(len(candidates))]
        for i in range(len(candidates)):
            if mask[i]:
                new_ep_objs_batch = update_ep(virtual_ep_objs_batch, candidates[i]['prediction'])
                sparsity[i] = compute_sparsity(new_ep_objs_batch)          
        return sparsity

    def evaluate_hypervolume_sparsity(self, candidates, mask, virtual_ep_objs_batch):
        hv = [0.0 for _ in range(len(candidates))]
        sparsity = [0.0 for _ in range(len(candidates))]
        for i in range(len(candidates)):
            if mask[i]:
                new_ep_objs_batch = update_ep(virtual_ep_objs_batch, candidates[i]['prediction'])
                hv[i] = compute_hypervolume(new_ep_objs_batch)
                sparsity[i] = compute_sparsity(new_ep_objs_batch)
        return hv, sparsity

    def evaluate_hypervolume_sparsity_parallel(self, args, candidates, mask, virtual_ep_objs_batch):
        hv = [0.0 for _ in range(len(candidates))]
        sparsity = [0.0 for _ in range(len(candidates))]
        processes = []
        max_process_num = args.num_tasks * args.num_processes
        queue = Queue()
        for i in range(len(candidates)):
            if mask[i]:
                p = Process(target=update_ep_and_compute_hypervolume_sparsity, args=(i, virtual_ep_objs_batch, candidates[i]['prediction'], queue))
                p.start()
                processes.append(p)
                if len(processes) >= max_process_num:
                    for _ in processes:
                        task_id, hv_res, sparsity_res = queue.get()
                        hv[task_id] = hv_res
                        sparsity[task_id] = sparsity_res
                    processes = []
        for _ in processes:
            task_id, hv_res, sparsity_res = queue.get()
            hv[task_id] = hv_res
            sparsity[task_id] = sparsity_res
        return hv, sparsity

    def prediction_guided_selection(self, args, iteration, ep, opt_graph, scalarization_template):
        N = args.num_tasks # number of (sample, weight) to be selected
        num_weights = args.num_weight_candidates # for each sample, we have num_weights optimization directions to be candidates, they distribute evenly around the last weight direction

        ### Prediction ###

        candidates = [] # list of candidate, each candidate is a (sample, weight) pair associated with their predicted future point
        for sample in self.sample_batch:
            # get weights evenly distributed around the last weight direction and only discard the weight in the left bottom region
            weight_center = opt_graph.weights[sample.optgraph_id]
            weight_center = weight_center / np.sum(weight_center)
            weight_candidates = []
            generate_weights_batch_dfs(0, args.obj_num, 0.0, 1.0, args.delta_weight / 2.0, [], weight_candidates)
            test_weights = []

            # add weight center
            duplicated = False
            for succ in opt_graph.succ[sample.optgraph_id]:
                w = deepcopy(opt_graph.weights[succ])
                w = w / np.sum(w)
                if np.linalg.norm(w - weight_center) < 1e-3:
                    duplicated = True
                    break
            if not duplicated:
                test_weights.append(weight_center)
            
            # randomly add other weights
            weight_indices = np.array([i for i in range(len(weight_candidates))])
            np.random.shuffle(weight_indices)
            for i in range(len(weight_indices)):
                if len(test_weights) >= num_weights: # if enough weights have been selected, then stop
                    break
                weight = weight_candidates[weight_indices[i]]
                if np.linalg.norm(weight - weight_center) < 1e-3:
                    continue
                angle = np.arccos(np.clip(np.dot(weight_center, weight) / np.linalg.norm(weight_center) / np.linalg.norm(weight), -1.0, 1.0))
                if angle < np.pi / 4.0: # check if weight is close to the previous weight
                    duplicated = False
                    for succ in opt_graph.succ[sample.optgraph_id]:
                        w = deepcopy(opt_graph.weights[succ])
                        w = w / np.sum(w)
                        if np.linalg.norm(w - weight) < 1e-3:
                            duplicated = True
                            break
                    if not duplicated:
                        test_weights.append(weight)

            if len(test_weights) > 0:
                results = predict_hyperbolic(args, opt_graph, sample.optgraph_id, test_weights)
                for i in range(len(test_weights)):
                    candidates.append({'sample': sample, 'weight': test_weights[i], \
                        'prediction': results['predictions'][i]})

        ### Optimization ###
            
        # initialize virtual ep as current ep
        virtual_ep_objs_batch = []
        for i in range(len(ep.sample_batch)):
            virtual_ep_objs_batch.append(deepcopy(ep.sample_batch[i].objs))

        mask = np.ones(len(candidates), dtype = bool)

        predicted_offspring_objs = []
        elite_batch, scalarization_batch = [], []

        # greedy algorithm for knapsack problem
        alpha = args.sparsity

        for _ in range(N):
            hv, sparsity = self.evaluate_hypervolume_sparsity_parallel(args, candidates, mask, virtual_ep_objs_batch)

            # select the one with max dhv - alpha * dsparsity
            max_metrics, best_id = -np.inf, -1
            for i in range(len(candidates)):
                if mask[i]:
                    if hv[i] - alpha * sparsity[i] > max_metrics:
                        max_metrics, best_id = hv[i] - alpha * sparsity[i], i

            if best_id == -1:
                print('Too few candidates')
                break

            elite_batch.append(candidates[best_id]['sample'])
            scalarization = deepcopy(scalarization_template)
            scalarization.update_weights(candidates[best_id]['weight'] / np.sum(candidates[best_id]['weight']))
            scalarization_batch.append(scalarization)
            mask[best_id] = False

            # update virtual_ep_objs_batch
            predicted_new_objs = [deepcopy(candidates[best_id]['prediction'])]
            virtual_ep_objs_batch = update_ep(virtual_ep_objs_batch, predicted_new_objs[0])
            
            predicted_offspring_objs.extend(predicted_new_objs)

        return elite_batch, scalarization_batch, predicted_offspring_objs

    def random_selection(self, args, scalarization_template):
        elite_batch, scalarization_batch = [], []
        for _ in range(args.num_tasks):
            elite_idx = np.random.choice(len(self.sample_batch))
            elite_batch.append(self.sample_batch[elite_idx])
            weights = np.random.uniform(args.min_weight, args.max_weight, args.obj_num)
            weights = weights / np.sum(weights)
            scalarization = deepcopy(scalarization_template)
            scalarization.update_weights(weights)
            scalarization_batch.append(scalarization)
        return elite_batch, scalarization_batch
