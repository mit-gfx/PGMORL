'''
Visualize the training process.
'''
import argparse
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import os
from copy import deepcopy
from multiprocessing import Process
import sys
# np.set_printoptions(precision=1)

iterations_str = []
iterations = []
ep_objs = []
population_objs = []
elites_objs = []
elites_weights = []
predictions = []
offsprings = []

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

# compute the hypervolume and sparsity given the pareto points, only for 2-dim objectives now
def compute_metrics(obj_batch):
    if obj_batch.shape[1] != 2:
        return 0
    
    objs = obj_batch

    ref_x, ref_y = 0.0, 0.0 # set referent point as (0, 0)
    x, hypervolume = ref_x, 0.0
    sqdist = 0.0
    for i in range(len(objs)):
        hypervolume += (max(ref_x, objs[i][0]) - x) * (max(ref_y, objs[i][1]) - ref_y)
        x = max(ref_x, objs[i][0])
        if i > 0:
            sqdist += np.sum(np.square(objs[i] - objs[i - 1]))

    if len(objs) == 1:
        sparsity = 0.0
    else:
        sparsity = sqdist / (len(objs) - 1)

    print('Pareto size : {}, Hypervolume : {:.0f}, Sparsity : {:.2f}'.format(len(objs), hypervolume, sparsity))

    return hypervolume, sparsity

def get_objs(objs_path):
    objs = []
    if os.path.exists(objs_path):
        with open(objs_path, 'r') as fp:
            data = fp.readlines()
            for j, line_data in enumerate(data):
                line_data = line_data.split(',')
                objs.append([float(line_data[0]), float(line_data[1])])
    return objs

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='MO-HalfCheetah-v2')
parser.add_argument('--log-dir', type=str, required=True)
parser.add_argument('--save-fig', default=False, action='store_true')
parser.add_argument('--title', type=str, default=None)
parser.add_argument('--obj', type=str, nargs='+', default=None)

args = parser.parse_args()

if args.env in ['MO-HalfCheetah-v2', 'MO-Walker2d-v2', 'MO-Swimmer-v2', 'MO-Humanoid-v2']:
    args.obj = ['Forward Speed', 'Energy Efficiency']
elif args.env == 'MO-Ant-v2':
    args.obj = ['X-Axis Speed', 'Y-Axis Speed']
elif args.env == 'MO-Hopper-v2':
    args.obj = ['Running Speed', 'Jumping Height']

all_iteration_folders = os.listdir(args.log_dir)
for folder in all_iteration_folders:
    if os.path.isdir(os.path.join(args.log_dir, folder)) and folder != 'final':
        iterations.append(int(folder))
        population_log_dir = os.path.join(args.log_dir, folder, 'population')
        # load population objs
        population_objs.append(get_objs(os.path.join(population_log_dir, 'objs.txt')))
        # load ep
        ep_log_dir = os.path.join(args.log_dir, folder, 'ep')
        ep_objs.append(get_objs(os.path.join(ep_log_dir, 'objs.txt')))
        # load elites
        elites_log_dir = os.path.join(args.log_dir, folder, 'elites')
        elites_objs.append(get_objs(os.path.join(elites_log_dir, 'elites.txt')))
        elites_weights.append(get_objs(os.path.join(elites_log_dir, 'weights.txt')))
        predictions.append(get_objs(os.path.join(elites_log_dir, 'predictions.txt')))
        offsprings.append(get_objs(os.path.join(elites_log_dir, 'offsprings.txt')))

for weights in elites_weights:
    for weight in weights:
        norm = np.sqrt(weight[0] ** 2 + weight[1] ** 2)
        weight[0] /= norm
        weight[1] /= norm
        
iterations = np.array(iterations)
ep_objs = np.array(ep_objs)
population_objs = np.array(population_objs)
elites_objs = np.array(elites_objs)
elites_weights = np.array(elites_weights)
predictions = np.array(predictions)
offsprings = np.array(offsprings)

have_pred = (predictions.size > 0)
have_offspring = (offsprings.size > 0)

sorted_index = np.argsort(iterations)
sorted_ep_objs = []
sorted_population_objs = []
sorted_elites_objs = []
sorted_elites_weights = []
sorted_predictions = []
sorted_offsprings = []
utopians = []
for i in range(len(sorted_index)):
    index = sorted_index[i]
    sorted_ep_objs.append(deepcopy(ep_objs[index]))
    sorted_population_objs.append(deepcopy(population_objs[index]))
    sorted_elites_objs.append(deepcopy(elites_objs[index]))
    sorted_elites_weights.append(deepcopy(elites_weights[index]))
    if have_pred:
        sorted_predictions.append(deepcopy(predictions[index]))
    if have_offspring:
        if i < len(sorted_index) - 1:
            sorted_offsprings.append(deepcopy(offsprings[sorted_index[i + 1]]))
        else:
            sorted_offsprings.append([])
    utopian = np.max(sorted_ep_objs[i], axis=0)
    utopians.append(utopian)

all_elites_objs = []
all_elites_weights = []
for i in range(len(sorted_elites_objs)):
    for j in range(len(sorted_elites_objs[i])):
        all_elites_objs.append(sorted_elites_objs[i][j])
        all_elites_weights.append(sorted_elites_weights[i][j])
all_elites_objs = np.array(all_elites_objs)
all_elites_weights = np.array(all_elites_weights)

hypervolumes, sparsities = [], []
for i in range(len(sorted_ep_objs)):
    hypervolume, sparsity = compute_metrics(np.array(sorted_ep_objs[i]))
    hypervolumes.append(hypervolume)
    sparsities.append(sparsity)

print('Pareto size : {}, Hypervolume : {:.0f}, Sparsity : {:.2f}'.format(len(sorted_ep_objs[-1]), hypervolumes[-1], sparsities[-1]))

fig, ax = plt.subplots(1, 3, figsize = (24, 8))

ax[0].axis('equal')

N_elites = len(sorted_elites_objs[0])
N_prediction_step = len(sorted_predictions[0]) // N_elites if have_pred else 0
N_offsprings = len(sorted_offsprings[0]) // N_elites if have_offspring else 0
epoch_pareto_drawing = []
epoch_population_drawing = []
epoch_elites_drawing = []
epoch_weights_drawing = []
epoch_predictions_drawing = []
epoch_offsprings_drawing = []
utopians_drawing = []
for i in range(len(sorted_ep_objs)):
    if sorted_population_objs != []:
        cur_population_objs = np.transpose(sorted_population_objs[i].copy())
        epoch_population_drawing.append(ax[0].scatter(cur_population_objs[0], cur_population_objs[1], s = 15, color = 'grey'))
    cur_ep_objs = np.transpose(sorted_ep_objs[i].copy())
    epoch_pareto_drawing.append(ax[0].scatter(cur_ep_objs[0], cur_ep_objs[1], s = 15, color = 'black'))
    cur_elites_objs = np.transpose(sorted_elites_objs[i].copy())
    if have_pred:
        cur_predictions = np.transpose(sorted_predictions[i].copy())
    if have_offspring:
        cur_offsprings = np.transpose(sorted_offsprings[i].copy())
    cur_elites_drawing = []
    cur_weights_drawing = []
    cur_predictions_drawing = []
    cur_offsprings_drawing = []
    N_offsprings = len(sorted_offsprings[i]) // N_elites
    for j in range(N_elites):
        cur_elites_drawing.append(ax[0].scatter(cur_elites_objs[0][j], cur_elites_objs[1][j], s = 120, facecolors = 'none', edgecolors = 'g'))
        cur_weights_drawing.append(ax[0].arrow(cur_elites_objs[0][j], cur_elites_objs[1][j], sorted_elites_weights[i][j][0] * 30, sorted_elites_weights[i][j][1] * 30, width = 0.1, head_width = 3, head_length = 5))
        if have_pred:
            cur_predictions_drawing.append(ax[0].scatter(cur_predictions[0][j * N_prediction_step:(j + 1) * N_prediction_step], cur_predictions[1][j * N_prediction_step:(j + 1) * N_prediction_step], s = 15, color = 'green'))
        if have_offspring and i < len(sorted_ep_objs) - 1:
            cur_offsprings_drawing.append(ax[0].scatter(cur_offsprings[0][(j + 1) * N_offsprings - 1], cur_offsprings[1][(j + 1) * N_offsprings - 1], s = 15, color = 'red'))
    epoch_elites_drawing.append(cur_elites_drawing)
    epoch_weights_drawing.append(cur_weights_drawing)
    epoch_predictions_drawing.append(cur_predictions_drawing)
    epoch_offsprings_drawing.append(cur_offsprings_drawing)

    utopians_drawing.append(ax[0].scatter(utopians[i][0], utopians[i][1], s = 80, marker = (5, 1), color = 'r'))

ax[0].scatter([0], [0])
ax[1].plot(hypervolumes)
ax[2].plot(sparsities)

epoch_hypervolume_drawing = []
epoch_sparsity_drawing = []
for i in range(len(hypervolumes)):
    epoch_hypervolume_drawing.append(ax[1].scatter(i, hypervolumes[i], color = 'r'))
    epoch_sparsity_drawing.append(ax[2].scatter(i, sparsities[i], color = 'r'))

for i in range(1, len(epoch_pareto_drawing)):
    epoch_pareto_drawing[i].set_visible(False)
    epoch_hypervolume_drawing[i].set_visible(False)
    epoch_sparsity_drawing[i].set_visible(False)
    if epoch_population_drawing != []:
        epoch_population_drawing[i].set_visible(False)
    for j in range(N_elites):
        epoch_elites_drawing[i][j].set_visible(False)
        epoch_weights_drawing[i][j].set_visible(False)
        if have_pred:
            epoch_predictions_drawing[i][j].set_visible(False)
        if have_offspring and i < len(epoch_pareto_drawing) - 1:
            epoch_offsprings_drawing[i][j].set_visible(False)
    utopians_drawing[i].set_visible(False)
    
global cur, cur_elite, visualize_elites, visualize_population, visualize_pareto, visualize_prediction, visualize_offsprings
cur = 0
cur_elite = -1
visualize_elites = True
visualize_population = True
visualize_pareto = True
visualize_prediction = True
visualize_offsprings = True
N = len(epoch_pareto_drawing)

if args.obj is not None:
    ax[0].set_xlabel(args.obj[0])
    ax[0].set_ylabel(args.obj[1])
ax[1].set_xlabel('generation')
ax[2].set_xlabel('generation')

ax[0].set_title('Pareto')
ax[1].set_title('Hypervolume')
ax[2].set_title('Sparsity')

def press(event):
    global cur, cur_elite, visualize_elites, visualize_population, visualize_pareto, visualize_prediction, visualize_offsprings
    epoch_pareto_drawing[cur].set_visible(False)
    epoch_hypervolume_drawing[cur].set_visible(False)
    epoch_sparsity_drawing[cur].set_visible(False)
    if epoch_population_drawing != []:
        epoch_population_drawing[cur].set_visible(False)
    for j in range(N_elites):
        epoch_elites_drawing[cur][j].set_visible(False)
        epoch_weights_drawing[cur][j].set_visible(False)
        if have_pred:
            epoch_predictions_drawing[cur][j].set_visible(False)
        if have_offspring and cur < N - 1:
            epoch_offsprings_drawing[cur][j].set_visible(False)
    utopians_drawing[cur].set_visible(False)

    if event.key == 'right':
        cur = cur + 1
        if cur >= N:
            cur = N - 1
    elif event.key == 'left':
        cur = cur - 1
        if cur < 0:
            cur = 0
    elif event.key == 'up':
        cur_elite += 1
        if cur_elite == N_elites:
            cur_elite = -1
    elif event.key == 'down':
        cur_elite -= 1
        if cur_elite < -1:
            cur_elite = N_elites - 1
    elif event.key == '1':
        visualize_elites = not visualize_elites
    elif event.key == 'p':
        visualize_population = not visualize_population
    elif event.key == 'P':
        visualize_pareto = not visualize_pareto
    elif event.key == 'n':
        visualize_prediction = not visualize_prediction
    elif event.key == 'r':
        visualize_offsprings = not visualize_offsprings

    epoch_pareto_drawing[cur].set_visible(visualize_pareto)
    epoch_hypervolume_drawing[cur].set_visible(True)
    epoch_sparsity_drawing[cur].set_visible(True)
    if epoch_population_drawing != []:
        epoch_population_drawing[cur].set_visible(visualize_population)
    if cur_elite == -1:
        for j in range(N_elites):
            epoch_elites_drawing[cur][j].set_visible(visualize_elites)
            epoch_weights_drawing[cur][j].set_visible(visualize_elites)
            if have_pred:
                epoch_predictions_drawing[cur][j].set_visible(visualize_prediction)
            if have_offspring and cur < N - 1:
                epoch_offsprings_drawing[cur][j].set_visible(visualize_offsprings)
    else:
        epoch_elites_drawing[cur][cur_elite].set_visible(visualize_elites)
        epoch_weights_drawing[cur][cur_elite].set_visible(visualize_elites)
        if have_pred:
            epoch_predictions_drawing[cur][cur_elite].set_visible(visualize_prediction)
        if have_offspring and cur < N - 1:
            epoch_offsprings_drawing[cur][cur_elite].set_visible(visualize_offsprings)
    utopians_drawing[cur].set_visible(True)

    if cur_elite == -1:
        sys.stdout.write('\rcur = {}, utopian = {}'.format(cur, utopians[cur]))
    else:
        sys.stdout.write('\rcur = {}, cur_elite = {}, weight = {}, utopian = {}'.format(cur, cur_elite, sorted_elites_weights[cur][cur_elite], utopians[cur]))
    sys.stdout.flush()
    
    fig.canvas.draw()

fig.canvas.mpl_connect('key_press_event', press)

print('-----------------------------------------------')
print('> help')
print("- change epoch: 'left', 'right'")
print("- change tasks: 'up', 'down'")
print("- turn on/off visualization of tasks: '1'")
print("- turn on/off visualization of population: 'p'")
print("- turn on/off visualization of pareto: 'P'")
print("- turn on/off visualization of prediction: 'n'")
print("- turn on/off visualization of offsprings: 'r'")
print('-----------------------------------------------')

if not args.save_fig:
    plt.show()
else:
    plt.savefig(os.path.join(args.log_dir, 'EP_his.png'))
    plt.show()
