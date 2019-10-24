import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import Parameter
import matplotlib.pyplot as plt
import argparse


class MLP(nn.Module):
	def __init__(self, num_features, num_hidden, num_outputs):
		self.W_1 = Parameter(init.xavier_normal_(torch.Tensor(num_hidden, num_features)))
		self.b_1 = Parameter(init.constant_(torch.Tensor(num_hidden), 0))

		self.W_2 = Parameter(init.xavier_normal_(torch.Tensor(num_outputs, num_hidden)))
		self.b_2 = Parameter(init.constant_(torch.Tensor(num_outputs), 0))

	def forward(self, x):
		x = F.relu(F.linear(x, self.W_1, self.b_1))
		x = F.linear(x, self.W_2, self.b_2)

		return x


def fit(net, data):

	x_train, y_train, x_val, y_val = data



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
	plt.show()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-algorithm', type=str, default='SGD')
	args = parser.parse_args()

	data = np.load('mnist.npz')
	num_classes = 10
	x_train = data['X_train'][:1000].astype('float32')
	targets_train = data['y_train'][:1000].astype('int32')

	x_valid = data['X_valid'][:500].astype('float32')
	targets_valid = data['y_valid'][:500].astype('int32')

	x_test = data['X_test'][:500].astype('float32')
	targets_test = data['y_test'][:500].astype('int32')

	print("Information on dataset")
	print("x_train", x_train.shape)
	print("targets_train", targets_train.shape)
	print("x_valid", x_valid.shape)
	print("targets_valid", targets_valid.shape)
	print("x_test", x_test.shape)
	print("targets_test", targets_test.shape)

	plot_mnist(x_train)



	net = MLP()



