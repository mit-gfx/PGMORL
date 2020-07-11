from copy import deepcopy
import numpy as np

'''
OptGraph is a data structure to store the optimization history.
The optimization history is a rooted forest, and is organized in a tree structure.
'''
class OptGraph:
    def __init__(self):
        self.weights = []
        self.objs = []
        self.delta_objs = []
        self.prev = []
        self.succ = []

    def insert(self, weights, objs, prev):
        self.weights.append(deepcopy(weights) / np.linalg.norm(weights))
        self.objs.append(deepcopy(objs))
        self.prev.append(prev)
        if prev == -1:
            self.delta_objs.append(np.zeros_like(objs))
        else:
            self.delta_objs.append(objs - self.objs[prev])
        if prev != -1:
            self.succ[prev].append(len(self.objs) - 1)
        self.succ.append([])
        return len(self.objs) - 1