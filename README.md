# PG-MORL

This repository contains the implementation for the paper [Prediction-Guided Multi-Objective Reinforcement Learning for Continuous Robot Control](http://people.csail.mit.edu/jiex/papers/PGMORL/) (**ICML 2020**). 

In this paper, we propose an evolutionary learning algorithm to compute a high-quality and dense Pareto solutions for multi-objective continuous robot control problems. We also design seven multi-objective continuous control benchmark problems based on [Mujoco](http://www.mujoco.org/), which are also included in this repository. This repository also contains the code for the baseline algorithms in the paper.

![teaser](images/teaser.gif)



## Installation

#### Prerequisites

- **Operating System**: tested on Ubuntu 16.04 and Ubuntu 18.04.
- **Python Version**: >= 3.7.4.
- **PyTorch Version**: >= 1.3.0.
- **MuJoCo** : install mujoco and mujoco-py of version 2.0 by following the instructions in [mujoco-py](<https://github.com/openai/mujoco-py>).

#### Install Dependencies

You can either install the dependencies in a conda virtual env (recomended) or manually. 

For conda virtual env installation, simply create a virtual env named **pgmorl** by:

```
conda env create -f environment.yml
```

If you prefer to install all the dependencies by yourself, you could open `environment.yml` in editor to see which packages need to be installed by `pip`.



## Run the Code

The training related code are in the folder `morl`. We provide the scripts in `scrips` folder to run our algorithm/baseline algorithms on each problem described in the paper, and also provide several visualization scripts in `scripts/plot` folder for you to visualize the computed Pareto policies and the training process. 

#### Precomputed Pareto Results

While you can run the training code the compute the Pareto policies from scratch by following the training steps below, we also provide the precomputed Pareto results for each problem. You can download them for each problem separately in [this google drive link](https://drive.google.com/drive/folders/15toW4SjF2b4PPvU2ZFA6kTweWfh7CqQr?usp=sharing) and directly visualize them with the visualization instructions to play with the results. After downloading the precomputed results, you can unzip it, create a `results` folder under the project root directory, and put the downloaded file inside.

#### Benchmark Problems

We design seven multi-objective continuous control benchmark problems based on Mujoco simulation, including *Walker2d-v2*, *HalfCheetah-v2*, *Hopper-v2*, *Ant-v2*, *Swimmer-v2*, *Humanoid-v2*, and *Hopper-v3*. A suffix of *-v3* indicates a three-objective problem. The reward (i.e. objective) functions in each problem are designed to have similar scales. All environments code can be found in `environments/mujoco` folder. To avoid conflicting to the original mujoco environment names, we add a `MO-` prefix to the name of each environment. For example, the environment name for *Walker2d-v2* is *MO-Walker2d-v2*.

#### Train

The main entrance of the training code is at  `morl/run.py`. We provide a training script in `scripts` folder for each problem for you to easily start with. You can just follow the following steps to see how to run the training for each problem by each algorithm (our algorithm and baseline algorithms).

- Enter the project folder

  ```
  cd PGMORL
  ```

- Activate the conda env:

  ```
  conda activate pgmorl
  ```

- To run our algorithm on *Walker2d-v2* for a single run:

  ```
  python scripts/walker2d-v2.py --pgmorl --num-seeds 1 --num-processes 1
  ```

  You can also set other flags as arguments to run the baseline algorithms (e.g. --ra, --moead, --pfa, --random). Please refer to the python scripts for more details about the arguments.

- By default, the results are stored in `results/[problem name]/[algorithm name]/[seed idx]`.

#### Visualization

- We provide a script to visualize the computed/downloaded Pareto results.

  ```
  python scripts/plot/ep_obj_visualize_2d.py --env MO-Walker2d-v2 --log-dir ./results/Walker2d-v2/pgmorl/0/
  ```

  You can replace `MO-Walker2d-v2` to your problem name, and replace the `./results/Walker2d-v2/pgmorl/0` by the path to your stored results.

  It will show a plot of the computed Pareto policies in the performance space. By double-click the point in the plot, it will automatically open a new window and render the simulation for the selected policy. 

- We also provide a script to help you visualize the evolution process of the policy population.

  ```
  python scripts/plot/training_visualize_2d.py --env MO-Walker2d-v2 --log-dir ./results/Walker2d-v2/pgmorl/0/
  ```
  
  It will plot the policy population (gray points) in each generation with some other useful information. The black points are the policies on the Pareto front, the green circles are the selected policies to be optimized in next generation, the red points are the predicted offsprings and the green points are the real offsprings. You can interact with the plot with the keyboard. For example, be pressing left/right, you can evolve the policy population by generation. You can refer to the plot scripts for the full description of the allowable operations.

#### Reproducibility

We run all our experiments on VM instances with 96 Intel Skylake vCPUs and 86.4G memory on Google Cloud Platform without GPU. 



## Acknowledgement

We use the implementation of [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) as the underlying PPO implementation and modify it into our Multi-Objective Policy Gradient algorithm.



## Citation

If you find our paper or code is useful, please consider citing: 

```
@inproceedings{xu2020prediction,
  title={Prediction-Guided Multi-Objective Reinforcement Learning for Continuous Robot Control},
  author={Xu, Jie and Tian, Yunsheng and Ma, Pingchuan and Rus, Daniela and Sueda, Shinjiro and Matusik, Wojciech},
  booktitle={Proceedings of the 37th International Conference on Machine Learning},
  year={2020}
}
```



