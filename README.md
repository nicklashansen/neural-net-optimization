# Optimization for Deep Learning

This repository contains PyTorch implementations of popular/recent optimization algorithms for deep learning, including SGD, SGD w/ momentum, SGD w/ Nesterov momentum, SGDW, RMSprop, Adam, Nadam, Adam w/ L2 regularization, AdamW, RAdam, RAdamW, Gradient Noise, Gradient Dropout, Learning Rate Dropout and Lookahead.

All extensions have been implemented such that it allows for mix-and-match optimization, e.g. you can train a neural net using RAdamW with both Nesterov momentum, Gradient Noise, Learning Rate Dropout and Lookahead.

_____


## Related papers

Material in this repository has been developed as part of a special course / study and reading group. This is the list of papers that we have discussed and/or implemented:

[An Overview of Gradient Descent Optimization Algorithms](https://arxiv.org/abs/1609.04747)

[Optimization Methods for Large-Scale Machine Learning](https://arxiv.org/abs/1606.04838)

[On the importance of initialization and momentum in deep learning](https://www.cs.toronto.edu/~fritz/absps/momentum.pdf)

[Aggregated Momentum: Stability Through Passive Damping](https://arxiv.org/abs/1804.00325)

[ADADELTA: An Adaptive Learning Rate Method](https://arxiv.org/abs/1212.5701)

[RMSprop](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

[On the Convergence of Adam and Beyond](https://arxiv.org/abs/1904.09237)

[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)

[On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265v1)

[Incorporating Nesterov Momentum Into Adam](https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ)

[Adaptive Gradient Methods with Dynamic Bound of Learning Rate](https://arxiv.org/abs/1902.09843)

[On the Convergence of AdaBound and its Connection to SGD](https://arxiv.org/abs/1908.04457v1)

[Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610)

[The Marginal Value of Adaptive Gradient Methods in Machine Learning](https://arxiv.org/abs/1705.08292)

[Why Learning of Large-Scale Neural Networks Behaves Like Convex Optimization](https://arxiv.org/abs/1903.02140v1)

[Towards Explaining the Regularization Effect of Initial Large Learning Rate in Training Neural Networks](https://arxiv.org/abs/1907.04595)

[Curriculum Learning in Deep Neural Networks](https://arxiv.org/abs/1904.12887)

[HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent](https://arxiv.org/abs/1106.5730)

[Adding Gradient Noise Improves Learning for Very Deep Networks](https://arxiv.org/abs/1511.06807)

[Learning Rate Dropout](https://arxiv.org/abs/1912.00144)

[Towards Explaining the Regularization Effect of Initial Large Learning Rate in Training Neural Networks](https://arxiv.org/abs/1907.04595)

_____


## How to run

You can run the experiments and algorithms by calling e.g.

```python main.py -num_epochs 30 -dataset cifar -num_train 50000 -num_val 2048 -lr_schedule True```

with arguments as specified in the ```main.py``` file. The algorithms can be run on two different datasets, MNIST and CIFAR-10. For MNIST a small MLP is used for proof of concept, whereas a 808,458 parameter CNN is used for CIFAR-10. You may optionally decrease the size of the dataset and/or number of epochs to decrease computational complexity, but the arguments given above were used to produce the results shown here.

_____


## Results

Below you will find our main results. As for all optimization problems, the performance of particular algorithms is highly dependent on the problem details as well as hyper-parameters. While we have made no attempt at fine-tuning the hyper-parameters of individual optimization methods, we have kept as many hyper-parameters as possible constant to better allow for comparison. Wherever possible, default hyper-parameters as proposed by original authors have been used.

When faced with a real application, one should always try out a number of different algorithms and hyper-parameters to figure out what works better for your particular problem.

![cifar_sgd](https://raw.githubusercontent.com/nicklashansen/neural-net-optimization/master/results/loss_cifar_sgd.png)

![cifar_rmsprop_adam](https://raw.githubusercontent.com/nicklashansen/neural-net-optimization/master/results/loss_cifar_rmsprop_adam.png)

![cifar_adam_weight_decay](https://raw.githubusercontent.com/nicklashansen/neural-net-optimization/master/results/loss_cifar_adam_weight_decay.png)

![cifar_adam](https://raw.githubusercontent.com/nicklashansen/neural-net-optimization/master/results/loss_cifar_adam.png)

![cifar_lrd](https://raw.githubusercontent.com/nicklashansen/neural-net-optimization/master/results/loss_cifar_lrd.png)

![cifar_gradnoise](https://raw.githubusercontent.com/nicklashansen/neural-net-optimization/master/results/loss_cifar_gradnoise.png)

![cifar_lookahead](https://raw.githubusercontent.com/nicklashansen/neural-net-optimization/master/results/loss_cifar_lookahead.png)
