import numpy as np
from abc import abstractmethod
import torch

class ScalarizationFunction():
    def __init__(self, num_objs, weights = None):
        self.num_objs = num_objs
        if weights is not None:
            self.weights = torch.Tensor(weights)
        else:
            self.weights = None
    
    def update_weights(self, weights):
        if weights is not None:
            self.weights = torch.Tensor(weights)

    @abstractmethod
    def evaluate(self, objs):
        pass

class WeightedSumScalarization(ScalarizationFunction):
    def __init__(self, num_objs, weights = None):
        super(WeightedSumScalarization, self).__init__(num_objs, weights)
    
    def update_z(self, z):
        pass

    def evaluate(self, objs):
        return (objs * self.weights).sum(axis = -1)