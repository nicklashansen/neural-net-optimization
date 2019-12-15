import os
import pickle as pkl
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import seaborn as sns
import torch
import torchvision
from torch.utils import data
from torchvision import transforms

import optimizers

optim_dict = {
		'sgd': {
			'label': 'SGD',
			'lr': 1e-3
		},
		'sgd_momentum': {
			'label': 'SGD w/ momentum',
			'lr': 1e-3,
			'mu': 0.99
		},
		'sgd_nesterov': {
			'label': 'SGD w/ Nesterov momentum',
			'lr': 1e-3,
			'mu': 0.99,
			'nesterov': True
		},
		'sgd_weight_decay': {
			'label': 'SGDW',
			'lr': 1e-3,
			'mu': 0.99,
			'weight_decay': 1e-6
		},
		'sgd_lrd': {
			'label': 'SGD w/ momentum + LRD',
			'lr': 1e-3,
			'mu': 0.99,
			'lrd': 0.5
		},
		'adam': {
			'label': 'Adam',
			'lr': 1e-3
		},
		'adamW':{
			'label': 'AdamW',
			'lr': 1e-3,
			'weight_decay': 1e-4
		},
		'adam_l2':{
			'label': 'AdamL2',
			'lr': 1e-3,
			'l2_reg': 1e-4
		},
		'adam_lrd': {
			'label': 'Adam w/ LRD',
			'lr': 1e-3,
			'lrd': 0.5
		},
		'Radam': {
			'label': 'RAdam',
			'lr': 1e-3,
			'rectified': True
		},
		'RadamW': {
			'label': 'RAdamW',
			'lr': 1e-3,
			'rectified': True,
			'weight_decay': 1e-4
		},
		'Radam_lrd': {
			'label': 'RAdam w/ LRD',
			'lr': 1e-3,
			'rectified': True,
			'lrd': 0.5
		},
		'nadam': {
			'label': 'Nadam',
			'lr': 1e-3,
			'nesterov': True
		},
		'rmsprop': {
			'label': 'RMSprop',
			'lr': 1e-3,
			'beta2': 0.9,
		},
		'lookahead_sgd': {
			'label': 'Lookahead (SGD)',
			'lr': 1e-3,
			'mu': 0.99
		},
		'lookahead_adam': {
			'label': 'Lookahead (Adam)',
			'lr': 1e-3
		},
		'gradnoise_adam': {
			'label': 'Gradient Noise (Adam)',
			'lr': 1e-3
		},
		'graddropout_adam': {
			'label': 'Gradient Dropout (Adam)',
			'lr': 1e-3
		}
	}


def split_optim_dict(d:dict) -> tuple:
	"""
	Splits an optimization dict into label and dict.
	"""
	temp_d = deepcopy(d)
	label = temp_d['label']
	del temp_d['label']

	return label, temp_d


def load_cifar(num_train=50000, num_val=2048):
	"""
	Loads a subset of the CIFAR dataset and returns it as a tuple.
	"""
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

	train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
	val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

	train_dataset, _ = torch.utils.data.random_split(train_dataset, lengths=[num_train, len(train_dataset)-num_train])
	val_dataset, _ = torch.utils.data.random_split(val_dataset, lengths=[num_val, len(val_dataset)-num_val])

	return train_dataset, val_dataset


def load_mnist(filename='data/mnist.npz', num_train=4096, num_val=512):
	"""
	Loads a subset of the grayscale MNIST dataset and returns it as a tuple.
	"""
	data = np.load(filename)

	x_train = data['X_train'][:num_train].astype('float32')
	y_train = data['y_train'][:num_train].astype('int32')

	x_valid = data['X_valid'][:num_val].astype('float32')
	y_valid = data['y_valid'][:num_val].astype('int32')

	train_dataset = Dataset(x_train, y_train)
	val_dataset = Dataset(x_valid, y_valid)

	return train_dataset, val_dataset


def task_to_optimizer(task:str) -> torch.optim.Optimizer:
	"""
	Takes a task as string and returns its respective optimizer class.
	"""
	optimizer = None

	if 'sgd' in task.lower():
		optimizer = getattr(optimizers, 'SGD')
	if 'adam' in task.lower():
		optimizer = getattr(optimizers, 'Adam')
	if 'rmsprop' in task.lower():
		optimizer = getattr(optimizers, 'RMSProp')
	
	if optimizer is None:
		raise ValueError(f'Optimizer for task \'{task}\' was not recognized!')

	return optimizer


def wrap_optimizer(task:str, optimizer):
	"""
	Wraps an instantiated optimizer according to its task specified as a string.
	"""
	if 'gradnoise' in task.lower():
		optimizer = optimizers.GradientNoise(optimizer, eta=0.3, gamma=0.55)

	if 'graddropout' in task.lower():
		optimizer = optimizers.GradientDropout(optimizer, grad_retain=0.9)

	if 'lookahead' in task.lower():
		optimizer = optimizers.Lookahead(optimizer, k=5, alpha=0.5)

	return optimizer


class AvgLoss():
	"""
	Utility class that tracks the average loss.
	"""
	def __init__(self):
		self.sum, self.avg, self.n = 0, 0, 0
		self.losses = []

	def __iadd__(self, other):
		try:
			loss = other.data.numpy()
		except:
			loss = other
		
		if isinstance(other, list):
			self.losses.extend(other)
			self.sum += np.sum(other)
			self.n += len(other)
		else:
			self.losses.append(float(loss))
			self.sum += loss
			self.n += 1

		self.avg = self.sum / self.n

		return self

	def __str__(self):
		return '{0:.4f}'.format(round(self.avg, 4))

	def __len__(self):
		return len(self.losses)


class Dataset(data.Dataset):
	def __init__(self, X, y):
		self.X = X
		self.y = y

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]


def save_losses(losses, dataset:str, filename:str):
	if not os.path.exists(f'losses_{dataset}/'): os.makedirs(f'losses_{dataset}/')
	with open(f'losses_{dataset}/{filename}.pkl', 'wb') as f:
		pkl.dump(losses, f, protocol=pkl.HIGHEST_PROTOCOL)


def load_losses(dataset:str, filename:str):
	try:
		with open(f'losses_{dataset}/{filename}.pkl', 'rb') as f:
			return pkl.load(f)
	except:
		return None


def plot_mnist(X):
	idx, dim, classes = 0, 28, 10
	canvas = np.zeros((dim*classes, classes*dim))

	for i in range(classes):
		for j in range(classes):
			canvas[i*dim:(i+1)*dim, j*dim:(j+1)*dim] = X[idx].reshape((dim, dim))
			idx += 1

	sns.set(style='darkgrid')
	plt.figure(figsize=(9, 9))
	plt.axis('off')
	plt.tight_layout(pad=0)
	plt.imshow(canvas, cmap='gray')
	plt.savefig('mnist_examples.png')
	plt.clf()


def plot_loss(losses, val_losses, num_epochs):
	sns.set(style='darkgrid')
	plt.figure(figsize=(12, 6))
	plt.plot(np.linspace(0, num_epochs, num=len(losses)), losses.losses, label='Training loss')
	plt.plot(np.linspace(0, num_epochs, num=len(val_losses)), val_losses.losses, label='Validation loss')
	plt.tight_layout(pad=2)
	plt.xlabel('Epoch')
	plt.ylabel('Negative log likelihood')
	plt.savefig('loss.png')
	plt.clf()


def plot_losses(losses, val_losses, labels, num_epochs, title, plot_val=False, yscale_log=False, max_epochs=None):
	sns.set(style='darkgrid')
	plt.figure(figsize=(12, 6))
	colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:cyan', 'tab:olive']

	for i in range(len(losses)):
		plt.plot(np.linspace(0, num_epochs, num=len(losses[i])), smooth(losses[i].losses, 81), label=labels[i], alpha=1, c=colors[i])
		plt.plot(np.linspace(0, num_epochs, num=len(losses[i])), smooth(losses[i].losses, 21), alpha=0.25, c=colors[i])
		if plot_val:
			plt.plot(np.linspace(0, num_epochs, num=len(val_losses[i])), smooth(val_losses[i].losses, 81), alpha=1, linestyle='--', c=colors[i])

	plt.tight_layout(pad=2)
	plt.xlabel('Epoch')
	plt.ylabel('Cross-entropy')
	if yscale_log:
		plt.yscale('log')
	if max_epochs is not None:
		plt.xlim(-1, max_epochs)
	plt.ylim(0, 3)
	plt.title('CNN benchmark on CIFAR-10' if title == 'cifar' else 'MLP benchmark on MNIST')
	plt.legend(loc='upper right')
	plt.savefig(f'loss_{title}.png')
	plt.clf()


def smooth(signal, kernel_size, polyorder=3):
	return savgol_filter(signal, kernel_size, polyorder)
