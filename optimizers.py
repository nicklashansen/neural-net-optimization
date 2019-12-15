import torch
import numpy as np
from torch.optim import Optimizer
from torch.distributions import Bernoulli, Normal


class SGD(Optimizer):
    """
    Stochastic gradient descent. Also includes implementations of momentum,
    Nesterov's momentum, L2 regularization, SGDW and Learning Rate Dropout.
    """
    def __init__(self, params, lr, mu=0, nesterov=False, weight_decay=0, lrd=1):
        defaults = {'lr': lr, 'mu': mu, 'nesterov': nesterov, 'weight_decay': weight_decay, 'lrd': lrd}
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
            lrd_bernoulli = Bernoulli(probs=group['lrd'])

            if mu != 0 and 'v' not in group:
                group['v'] = []
                if nesterov:
                    group['theta'] = []
                for param in group['params']:
                    group['v'].append(torch.zeros_like(param))
                    if nesterov:
                        theta_param = torch.ones_like(param).mul_(param.data)
                        group['theta'].append(theta_param)

            for idx, param in enumerate(group['params']):
                param.grad.data -= weight_decay * param.data
                lrd_mask = lrd_bernoulli.sample(param.size()).to(param.device)

                if mu != 0:
                    v = group['v'][idx]
                    v = mu * v - lr * param.grad.data
                    group['v'][idx] = v

                    if nesterov:
                        group['theta'][idx] += lrd_mask * v
                        param.data = group['theta'][idx] + mu * v

                    else:
                        param.data += lrd_mask * v

                else:
                    param.data -= lrd_mask * lr * param.grad.data


class Adam(Optimizer):
    """
    Adam as proposed by https://arxiv.org/abs/1412.6980.
    Also includes a number of proposed extensions to the the Adam algorithm,
    such as Nadam, L2 regularization, AdamW, RAdam and Learning Rate Dropout.
    """
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, nesterov=False, l2_reg=0, weight_decay=0, rectified=False, lrd=1, eps=1e-8):
        defaults = {'lr': lr, 'beta1': beta1, 'beta2': beta2, 'nesterov': nesterov, 'l2_reg': l2_reg,
                    'weight_decay': weight_decay, 'rectified': rectified, 'lrd': lrd, 'eps': eps}
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
            l2_reg = group['l2_reg']
            weight_decay = group['weight_decay']
            rectified = group['rectified']
            lrd_bernoulli = Bernoulli(probs=group['lrd'])
            eps = group['eps']

            if 'm' not in group and 'v' not in group:
                group['m'] = []
                group['v'] = []
                group['t'] = 1
                if nesterov:
                    group['prev_grad'] = []
                for param in group['params']:
                    group['m'].append(torch.zeros_like(param))
                    group['v'].append(torch.zeros_like(param))
                    if nesterov:
                        group['prev_grad'].append(torch.zeros_like(param))

            for idx, param in enumerate(group['params']):
                if l2_reg:
                    param.grad.data += l2_reg * param.data

                if nesterov:
                    grad = group['prev_grad'][idx]
                else:
                    grad = param.grad.data

                lrd_mask = lrd_bernoulli.sample(param.size()).to(param.device)

                m = group['m'][idx]
                v = group['v'][idx]
                t = group['t']
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)

                if nesterov:
                    group['prev_grad'][idx] = param.grad.data

                if rectified:
                    rho_inf = 2 / (1 - beta2) - 1
                    rho = rho_inf - 2 * t * beta2**t / (1 - beta2**t)
                    if rho >= 5:
                        numerator = (1 - beta2**t) * (rho - 4) * (rho - 2) * rho_inf
                        denominator = (rho_inf - 4) * (rho_inf - 2) * rho
                        r = np.sqrt(numerator / denominator)
                        param.data += - lrd_mask * lr * r * m_hat / (torch.sqrt(v) + eps)
                    else:
                        param.data += - lrd_mask * lr * m_hat
                else:
                    param.data += - lrd_mask * lr * m_hat / (torch.sqrt(v_hat) + eps)

                if weight_decay:
                    param.data -= weight_decay * param.data

                group['m'][idx] = m
                group['v'][idx] = v

            group['t'] += 1


class RMSProp(Adam):
    """
    RMSprop as proposed by http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf.
    Note that this implementation, unlike the original RMSprop, uses bias-corrected moments.
    """
    def __init__(self, params, lr, beta2):
        super(RMSProp, self).__init__(params, lr, beta2=beta2, beta1=0)


class Lookahead(Optimizer):
    """
    Lookahead Optimization as proposed by https://arxiv.org/abs/1907.08610.
    This is a wrapper class that can be applied to an instantiated optimizer.
    """
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = optimizer.param_groups

        self.counter = 0
        for group in optimizer.param_groups:
            group['phi'] = []
            for param in group['params']:
                phi_param = torch.ones_like(param).mul_(param.data)
                group['phi'].append(phi_param)

    def step(self):
        if self.counter == self.k:
            for group_idx, group in enumerate(self.param_groups):
                for idx, _ in enumerate(group['phi']):
                    theta = self.optimizer.param_groups[group_idx]['params'][idx].data
                    group['phi'][idx] = group['phi'][idx] + self.alpha * (theta - group['phi'][idx])
            self.counter = 0
        else:
            self.counter += 1
            self.optimizer.step()


class GradientNoise(Optimizer):
    """
    Gradient Noise as proposed by https://arxiv.org/abs/1511.06807.
    This is a wrapper class that can be applied to an instantiated optimizer.
    """
    def __init__(self, optimizer, eta=0.3, gamma=0.55):
        self.optimizer = optimizer
        self.eta = eta
        self.gamma = gamma
        self.t = 0
        self.param_groups = optimizer.param_groups

    def step(self):
        normal = torch.empty(1).normal_(mean=0, std=np.sqrt(self.eta/((1+self.t)**self.gamma)))\
            .to(self.optimizer.param_groups[0]['params'][0].device)
        for group_idx, group in enumerate(self.param_groups):
            for idx, param in enumerate(group['params']):
                self.optimizer.param_groups[group_idx]['params'][idx].grad.data += normal
                self.optimizer.step()
                self.t += 1


class GradientDropout(Optimizer):
    """
    Gradient dropout as proposed by https://arxiv.org/abs/1912.00144.
    This is a wrapper class that can be applied to an instantiated optimizer.
    Note that this method does not improve optimization significantly and
    is only here for comparison to Learning Rate Dropout.
    """
    def __init__(self, optimizer, grad_retain=0.9):
        self.optimizer = optimizer
        self.grad_retain = grad_retain
        self.grad_bernoulli = Bernoulli(probs=grad_retain)
        self.param_groups = optimizer.param_groups

    def step(self):
        for group_idx, group in enumerate(self.param_groups):
            for idx, param in enumerate(group['params']):
                grad_mask = self.grad_bernoulli.sample(param.size()).to(param.device)
                self.optimizer.param_groups[group_idx]['params'][idx].grad.data *= grad_mask
                self.optimizer.step()
    