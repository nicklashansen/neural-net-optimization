import torch
from torch.optim import Optimizer

class SGD(Optimizer):
    def __init__(self, params, lr, mu=0, nesterov=False, weight_decay=0):
        defaults = {'lr': lr, 'mu': mu, 'nesterov': nesterov, 'weight_decay': weight_decay}
        super(SGD, self).__init__(params, defaults)

    def step(self):
        """
        Performs a single optimization step.
        """
        for group in self.param_groups:

            lr = group['lr']
            mu = group['mu']
            nesterov = group['nesterov']
            weight_decay = group['weight_decay']

            if mu != 0 and 'v' not in group:
                group['v'] = []
                if nesterov:
                    group['theta'] = []
                for param in group['params']:
                    group['v'].append(torch.zeros_like(param))
                    if nesterov:
                        group['theta'].append(param.data)

            for idx, param in enumerate(group['params']):
                param.grad.data += weight_decay * param.data

                if mu != 0:
                    v = group['v'][idx]
                    v = mu * v - lr * param.grad.data
                    group['v'][idx] = v

                    if nesterov:
                        group['theta'][idx] += v
                        param.data = group['theta'][idx] + mu * v

                    else:
                        param.data += v        

                else:
                    param.data -= lr * param.grad.data
