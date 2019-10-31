import torch
from torch.optim import Optimizer

class SGD(Optimizer):
    def __init__(self, params, lr, mu=0, nesterov=False):
        defaults = {'lr': lr, 'mu': mu, 'nesterov': nesterov}
        super(SGD, self).__init__(params, defaults)

    def step(self):
        """
        Performs a single optimization step.
        """
        for group in self.param_groups:

            lr = group['lr']
            mu = group['mu']
            nesterov = group['nesterov']

            if mu != 0 and 'v' not in group:
                group['v'] = []
                if nesterov:
                    group['theta'] = []
                for param in group['params']:
                    group['v'].append(torch.zeros_like(param))
                    if nesterov:
                        group['theta'].append(param.data)

            for idx, param in enumerate(group['params']):
                if mu != 0:
                    if nesterov:
                        v = group['v'][idx]
                        v = mu * v - lr * param.grad.data
                        group['v'][idx] = v
                        group['theta'][idx] += v
                        param.data = group['theta'][idx] + mu * v

                    else:
                        v = group['v'][idx]
                        v = mu * v - lr * param.grad.data
                        param.data += v
                        group['v'][idx] = v

                    #look_a_head = param.data + mu * v
                    #v = mu * v - lr * look_a_head.grad.data

                else:
                    param.data -= lr * param.grad.data
