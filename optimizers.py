import torch
from torch.optim import Optimizer

class SGD(Optimizer):
    def __init__(self, params, lr):
        super(SGD, self).__init__(params, defaults={'lr': lr})

    def step(self):
        """
        Performs a single optimization step.
        """
        for group in self.param_groups:
            for param in group['params']:
                param.data -= group['lr'] * param.grad.data
