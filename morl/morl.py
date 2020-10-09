import os, sys

import environments

# import python packages
import time
from copy import deepcopy

# import third-party packages
import numpy as np
import torch
import torch.optim as optim
from multiprocessing import Process, Queue, Event
import pickle

# import our packages
from scalarization_methods import WeightedSumScalarization
from sample import Sample
from task import Task
from ep import EP
from population_2d import Population as Population2d
from population_3d import Population as Population3d
from opt_graph import OptGraph
from utils import generate_weights_batch_dfs, print_info
from warm_up import initialize_warm_up_batch
from mopg import MOPG_worker

def run(args):

    # --------------------> Preparation <-------------------- #
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    device = torch.device("cpu")

    # build a scalarization template
    scalarization_template = WeightedSumScalarization(num_objs = args.obj_num, weights = np.ones(args.obj_num) / args.obj_num)

    total_num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    
    start_time = time.time()

    # initialize ep and population and opt_graph
    ep = EP()
    if args.obj_num == 2:
        population = Population2d(args)
    elif args.obj_num > 2:
        population = Population3d(args)
    else:
        raise NotImplementedError
    opt_graph = OptGraph()
    
    # Construct tasks for warm up
    elite_batch, scalarization_batch = initialize_warm_up_batch(args, device)
    rl_num_updates = args.warmup_iter
    for sample, scalarization in zip(elite_batch, scalarization_batch):
        sample.optgraph_id = opt_graph.insert(deepcopy(scalarization.weights), deepcopy(sample.objs), -1)
    
    episode = 0
    iteration = 0
    while iteration < total_num_updates:
        if episode == 0:
            print_info('\n------------------------------- Warm-up Stage -------------------------------')    
        else:
            print_info('\n-------------------- Evolutionary Stage: Generation {:3} --------------------'.format(episode))

        episode += 1
        
        offspring_batch = np.array([])

        # --------------------> RL Optimization <-------------------- #
        # compose task for each elite
        task_batch = []
        for elite, scalarization in \
                zip(elite_batch, scalarization_batch):
            task_batch.append(Task(elite, scalarization)) # each task is a (policy, weight) pair

        # run MOPG for each task in parallel
        processes = []
        results_queue = Queue()
        done_event = Event()
        
        for task_id, task in enumerate(task_batch):
            p = Process(target = MOPG_worker, \
                args = (args, task_id, task, device, iteration, rl_num_updates, start_time, results_queue, done_event))
            p.start()
            processes.append(p)

        # collect MOPG results for offsprings and insert objs into objs buffer
        all_offspring_batch = [[] for _ in range(len(processes))]
        cnt_done_workers = 0
        while cnt_done_workers < len(processes):
            rl_results = results_queue.get()
            task_id, offsprings = rl_results['task_id'], rl_results['offspring_batch']
            for sample in offsprings:
                all_offspring_batch[task_id].append(Sample.copy_from(sample))
            if rl_results['done']:
                cnt_done_workers += 1
        
        # put all intermidiate policies into all_sample_batch for EP update
        all_sample_batch = [] 
        # store the last policy for each optimization weight for RA
        last_offspring_batch = [None] * len(processes) 
        # only the policies with iteration % update_iter = 0 are inserted into offspring_batch for population update
        # after warm-up stage, it's equivalent to the last_offspring_batch
        offspring_batch = [] 
        for task_id in range(len(processes)):
            offsprings = all_offspring_batch[task_id]
            prev_node_id = task_batch[task_id].sample.optgraph_id
            opt_weights = deepcopy(task_batch[task_id].scalarization.weights).detach().numpy()
            for i, sample in enumerate(offsprings):
                all_sample_batch.append(sample)
                if (i + 1) % args.update_iter == 0:
                    prev_node_id = opt_graph.insert(opt_weights, deepcopy(sample.objs), prev_node_id)
                    sample.optgraph_id = prev_node_id
                    offspring_batch.append(sample)
            last_offspring_batch[task_id] = offsprings[-1]

        done_event.set()

        # -----------------------> Update EP <----------------------- #
        # update EP and population
        ep.update(all_sample_batch)
        population.update(offspring_batch)

        # ------------------- > Task Selection <--------------------- #
        if args.selection_method == 'moead':
            elite_batch, scalarization_batch = [], []
            weights_batch = []
            generate_weights_batch_dfs(0, args.obj_num, args.min_weight, args.max_weight, args.delta_weight, [], weights_batch)
            for weights in weights_batch:
                scalarization = deepcopy(scalarization_template)
                scalarization.update_weights(weights)
                scalarization_batch.append(scalarization)
                best_sample, best_value = None, -np.inf
                for sample in population.sample_batch:
                    value = scalarization.evaluate(torch.Tensor(sample.objs))
                    if value > best_value:
                        best_sample, best_value = sample, value
                elite_batch.append(best_sample)
        elif args.selection_method == 'prediction-guided':
            elite_batch, scalarization_batch, predicted_offspring_objs = population.prediction_guided_selection(args, iteration, ep, opt_graph, scalarization_template)
        elif args.selection_method == 'random':
            elite_batch, scalarization_batch = population.random_selection(args, scalarization_template)
        elif args.selection_method == 'ra':
            elite_batch = last_offspring_batch
            scalarization_batch = []
            weights_batch = []
            generate_weights_batch_dfs(0, args.obj_num, args.min_weight, args.max_weight, args.delta_weight, [], weights_batch)
            for weights in weights_batch:
                scalarization = deepcopy(scalarization_template)
                scalarization.update_weights(weights)
                scalarization_batch.append(scalarization)
        elif args.selection_method == 'pfa':
            if args.obj_num > 2:
                raise NotImplementedError
            elite_batch = last_offspring_batch
            scalarization_batch = []
            delta_ratio = (iteration + rl_num_updates + args.update_iter - args.warmup_iter) / (total_num_updates - args.warmup_iter)
            delta_ratio = np.clip(delta_ratio, 0.0, 1.0)
            for i in np.arange(args.min_weight, args.max_weight + 0.5 * args.delta_weight, args.delta_weight):
                w = np.clip(i + delta_ratio * args.delta_weight, args.min_weight, args.max_weight)
                weights = np.array([abs(w), abs(1.0 - w)])
                scalarization = deepcopy(scalarization_template)
                scalarization.update_weights(weights)
                scalarization_batch.append(scalarization)
        else:
            raise NotImplementedError
        
        print_info('Selected Tasks:')
        for i in range(len(elite_batch)):
            print_info('objs = {}, weight = {}'.format(elite_batch[i].objs, scalarization_batch[i].weights))

        iteration = min(iteration + rl_num_updates, total_num_updates)

        rl_num_updates = args.update_iter

        # ----------------------> Save Results <---------------------- #
        # save ep
        ep_dir = os.path.join(args.save_dir, str(iteration), 'ep')
        os.makedirs(ep_dir, exist_ok = True)
        with open(os.path.join(ep_dir, 'objs.txt'), 'w') as fp:
            for obj in ep.obj_batch:
                fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*obj))

        # save population
        population_dir = os.path.join(args.save_dir, str(iteration), 'population')
        os.makedirs(population_dir, exist_ok = True)
        with open(os.path.join(population_dir, 'objs.txt'), 'w') as fp:
            for sample in population.sample_batch:
                fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*(sample.objs)))
        # save optgraph and node id for each sample in population
        with open(os.path.join(population_dir, 'optgraph.txt'), 'w') as fp:
            fp.write('{}\n'.format(len(opt_graph.objs)))
            for i in range(len(opt_graph.objs)):
                fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + ';{:5f}' + (args.obj_num - 1) * ',{:5f}' + ';{}\n').format(*(opt_graph.weights[i]), *(opt_graph.objs[i]), opt_graph.prev[i]))
            fp.write('{}\n'.format(len(population.sample_batch)))
            for sample in population.sample_batch:
                fp.write('{}\n'.format(sample.optgraph_id))

        # save elites
        elite_dir = os.path.join(args.save_dir, str(iteration), 'elites')
        os.makedirs(elite_dir, exist_ok = True)
        with open(os.path.join(elite_dir, 'elites.txt'), 'w') as fp:
            for elite in elite_batch:
                fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*(elite.objs)))
        with open(os.path.join(elite_dir, 'weights.txt'), 'w') as fp:
            for scalarization in scalarization_batch:
                fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*(scalarization.weights)))
        if args.selection_method == 'prediction-guided':
            with open(os.path.join(elite_dir, 'predictions.txt'), 'w') as fp:
                for objs in predicted_offspring_objs:
                    fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*(objs)))
        with open(os.path.join(elite_dir, 'offsprings.txt'), 'w') as fp:
            for i in range(len(all_offspring_batch)):
                for j in range(len(all_offspring_batch[i])):
                    fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*(all_offspring_batch[i][j].objs)))

    # ----------------------> Save Final Model <---------------------- 

    os.makedirs(os.path.join(args.save_dir, 'final'), exist_ok = True)

    # save ep policies & env_params
    for i, sample in enumerate(ep.sample_batch):
        torch.save(sample.actor_critic.state_dict(), os.path.join(args.save_dir, 'final', 'EP_policy_{}.pt'.format(i)))
        with open(os.path.join(args.save_dir, 'final', 'EP_env_params_{}.pkl'.format(i)), 'wb') as fp:
            pickle.dump(sample.env_params, fp)
    
    # save all ep objectives
    with open(os.path.join(args.save_dir, 'final', 'objs.txt'), 'w') as fp:
        for i, obj in enumerate(ep.obj_batch):
            fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*(obj)))

    # save all ep env_params
    if args.obj_rms:
        with open(os.path.join(args.save_dir, 'final', 'env_params.txt'), 'w') as fp:
            for sample in ep.sample_batch:
                fp.write('obj_rms: mean: {} var: {}\n'.format(sample.env_params['obj_rms'].mean, sample.env_params['obj_rms'].var))
