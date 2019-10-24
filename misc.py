import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data

def load_mnist(filename='data/mnist.npz'):
	data = np.load(filename)

	x_train = data['X_train'][:1024].astype('float32')
	y_train = data['y_train'][:1024].astype('int32')

	x_valid = data['X_valid'][:512].astype('float32')
	y_valid = data['y_valid'][:512].astype('int32')

	x_test = data['X_test'][:512].astype('float32')
	y_test = data['y_test'][:512].astype('int32')

	print(f'Loaded data: ({len(x_train)}, {len(x_valid)}, {len(x_test)})')

	return x_train, y_train, x_valid, y_valid, x_test, y_test


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

	plt.figure(figsize=(4, 4))
	plt.axis('off')
	plt.imshow(canvas, cmap='gray')
	plt.title('MNIST handwritten digits')
	plt.savefig('mnist_examples.png')
