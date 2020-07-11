from copy import deepcopy
from sample import Sample

'''
Define a MOPG task, which is a pair of a policy and a scalarization weight.
'''
class Task:
    def __init__(self, sample, scalarization):
        self.sample = Sample.copy_from(sample)
        self.scalarization = deepcopy(scalarization)