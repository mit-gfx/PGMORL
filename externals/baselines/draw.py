import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import argparse
import csv
import pandas
import os
import sys
import pickle
import numpy as np
from os.path import join

# matplotlib
titlesize = 33
xsize = 30
ysize = 30
ticksize = 25
legendsize = 25
error_region_alpha = 0.25


def smoothed(x, w):
    """Smooth x by averaging over sliding windows of w, assuming sufficient length.
    """
    if len(x) <= w:
        return x
    smooth = []
    for i in range(1, w):
        smooth.append( np.mean(x[0:i]) )
    for i in range(w, len(x)+1):
        smooth.append( np.mean(x[i-w:i]) )
    assert len(x) == len(smooth), "lengths: {}, {}".format(len(x), len(smooth))
    return np.array(smooth)


def _get_stuff_from_monitor(mon):
    """Get stuff from `monitor` log files.

    Monitor files are named `0.envidx.monitor.csv` and have one line for each
    episode that finished in that CPU 'core', with the reward, length (number
    of steps) and the time (in seconds). The lengths are not cumulative, but
    time is cumulative.
    """
    scores = []
    steps  = []
    times  = []
    with open(mon, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for csv_row in csv_reader:
            # First two lines don't contain interesting stuff.
            if line_count == 0 or line_count == 1:
                line_count += 1
                continue
            scores.append(float(csv_row[0]))
            steps.append(int(csv_row[1]))
            times.append(float(csv_row[2]))
            line_count += 1
    print("finished: {}".format(mon))
    return scores, steps, times


def plot(args):
    """Load monitor curves and the progress csv file. And plot from those.
    """
    nrows, ncols = 1, 2
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, sharey=True, figsize=(11*ncols,7*nrows))
    title = args.title

    # Global statistics across all monitors
    scores_all = []
    steps_all = []
    times_all = []
    total_train_steps = 0
    train_hours = 0

    monitors = sorted(
        [x for x in os.listdir(args.path) if 'monitor.csv' in x and '.swp' not in x]
    )
    progfile = join(args.path,'progress.csv')

    # First row, info from all the monitors, i.e., number of CPUs.
    for env_idx,mon in enumerate(monitors):
        monitor_path = join(args.path, mon)
        scores, steps, times = _get_stuff_from_monitor(monitor_path)

        # Now process to see as a function of episodes and training steps, etc.
        num_episodes = len(scores)
        tr_episodes = np.arange(num_episodes)
        tr_steps = np.cumsum(steps)
        tr_times = np.array(times) / 60.0 # get it in minutes

        # Plot for individual monitors.
        envlabel = 'env {}'.format(env_idx)
        sm_10 = smoothed(scores, w=10)
        ax[0,0].plot(tr_steps, sm_10, label=envlabel+'; avg {:.1f} last {:.1f}'.format(
                np.mean(sm_10), sm_10[-1]))
        sm_100 = smoothed(scores, w=100)
        ax[0,1].plot(tr_times, sm_100, label=envlabel+'; avg {:.1f} last {:.1f}'.format(
                np.mean(sm_100), sm_100[-1]))

        # Handle global stuff.
        total_train_steps += tr_steps[-1]
        train_hours = max(train_hours, tr_times[-1] / 60.0)

    # Bells and whistles
    for row in range(nrows):
        for col in range(ncols):
            ax[row,col].set_ylabel("Scores", fontsize=30)
            ax[row,col].tick_params(axis='x', labelsize=25)
            ax[row,col].tick_params(axis='y', labelsize=25)
            leg = ax[row,col].legend(loc="best", ncol=1, prop={'size':25})
            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)
    ax[0,0].set_title(title+', Smoothed (w=10)', fontsize=titlesize)
    ax[0,0].set_xlabel("Train Steps (total {})".format(total_train_steps), fontsize=xsize)
    ax[0,1].set_title(title+', Smoothed (w=100)', fontsize=titlesize)
    ax[0,1].set_xlabel("Train Time (in Hours {:.2f})".format(train_hours), fontsize=xsize)
    plt.tight_layout()
    figname = '{}.png'.format(title)
    plt.savefig(figname)
    print("\nJust saved: {}".format(figname))


if __name__ == "__main__":
    pp = argparse.ArgumentParser()
    pp.add_argument('--path', type=str)
    pp.add_argument('--title', type=str)
    args = pp.parse_args()
    plot(args)