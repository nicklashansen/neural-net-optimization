import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import Parameter
from torch import utils
import matplotlib.pyplot as plt
import argparse
import misc


class MLP(nn.Module):

	def __init__(self, num_features, num_hidden, num_outputs):
		super(MLP, self).__init__()

		self.W_1 = Parameter(init.xavier_normal_(torch.Tensor(num_hidden, num_features)))
		self.b_1 = Parameter(init.constant_(torch.Tensor(num_hidden), 0))

		self.W_2 = Parameter(init.xavier_normal_(torch.Tensor(num_outputs, num_hidden)))
		self.b_2 = Parameter(init.constant_(torch.Tensor(num_outputs), 0))

	def forward(self, x):
		x = F.relu(F.linear(x, self.W_1, self.b_1))
		x = F.linear(x, self.W_2, self.b_2)

		return x


def fit(net, data, optimizer):
	batch_size = 64
	num_epochs = 100

	x_train, y_train, x_val, y_val, _, _ = data

	train_generator = utils.data.DataLoader(misc.Dataset(x_train, y_train), batch_size=batch_size)
	val_generator = utils.data.DataLoader(misc.Dataset(x_val, y_val), batch_size=batch_size)

	for epoch in range(num_epochs+1):

		epoch_loss = 0

		for x, y in train_generator:

			yhat = net(x)

			loss = F.cross_entropy(yhat, y.type(torch.LongTensor))
			epoch_loss += loss.data.numpy()

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		if epoch % 10 == 0:
			print(f'Epoch {epoch}/{num_epochs}, loss: {epoch_loss}')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-algorithm', type=str, default='SGD')
	args = parser.parse_args()

	data = misc.load_mnist()
	misc.plot_mnist(data[0])

	net = MLP(num_features=784, num_hidden=64, num_outputs=10)
	opt = torch.optim.SGD(net.parameters(), lr=1e-3)

	fit(net, data, opt)

