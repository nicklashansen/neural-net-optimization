import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils import data

def load_mnist(filename='data/mnist.npz', num_train=2048, num_val=256, num_test=512):
	"""
	Loads a subset of the grayscale MNIST dataset and returns it as a tuple.
	"""
	data = np.load(filename)

	x_train = data['X_train'][:num_train].astype('float32')
	y_train = data['y_train'][:num_train].astype('int32')

	x_valid = data['X_valid'][:num_val].astype('float32')
	y_valid = data['y_valid'][:num_val].astype('int32')

	x_test = data['X_test'][:num_test].astype('float32')
	y_test = data['y_test'][:num_test].astype('int32')

	return x_train, y_train, x_valid, y_valid, x_test, y_test


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
