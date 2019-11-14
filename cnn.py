import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
	"""
	A small convolutional neural network with parameters that we can optimize for the task.
	"""
	def __init__(self, num_layers=3, num_filters=32, num_classes=10, input_size=(3, 32, 32)):
		super(CNN, self).__init__()

		self.channels = input_size[0]
		self.height = input_size[1]
		self.width = input_size[2]
		self.num_filters = num_filters

		self.conv_in = nn.Conv2d(self.channels, self.num_filters, kernel_size=3, padding=1)
		cnn = []
		for _ in range(num_layers):
			cnn.append(nn.Conv2d(self.num_filters, kernel_size=3, padding=1))
			cnn.append(nn.ReLU())
		self.cnn = nn.Sequential(*cnn)

		self.out_lin = nn.Linear(self.channels*self.width*self.height, num_classes)

	def forward(self, x):
		x = F.relu(self.conv_in(x))
		x = self.cnn(x)

		return self.out_lin(x)
