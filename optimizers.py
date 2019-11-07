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

                else:
                    param.data -= lr * param.grad.data


class Adam(Optimizer):
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, nesterov=False):
        defaults = {'lr': lr, 'beta1': beta1, 'beta2': beta2, 'nesterov': nesterov}
        super(Adam, self).__init__(params, defaults)

    def step(self):
        """
        Performs a single optimization step.
        """
        for group in self.param_groups:

            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            nesterov = group['nesterov']

            if 'm' not in group and 'v' not in group:
                group['m'] = []
                group['v'] = []
                group['t'] = 1
                if nesterov:
                    group['theta'] = []
                for param in group['params']:
                    group['m'].append(torch.zeros_like(param))
                    group['v'].append(torch.zeros_like(param))
                    if nesterov:
                        group['theta'].append(param.data)

            for idx, param in enumerate(group['params']):
                m = group['m'][idx]
                v = group['v'][idx]
                t = group['t']
                m = beta1 * m + (1 - beta1) * param.grad.data
                v = beta2 * v + (1 - beta2) * param.grad.data**2
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)
                param.data += - lr * m_hat / (torch.sqrt(v_hat) + 1e-8)
                group['m'][idx] = m
                group['v'][idx] = v

            group['t'] += 1