import argparse
from copy import deepcopy

import misc
import optimizers
from mlp import MLP, fit


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-num_epochs', type=int, default=250)
	args = parser.parse_args()

	data = misc.load_mnist()
	misc.plot_mnist(data[0])
	print(f'Loaded data partitions: ({len(data[0])}, {len(data[2])}, {len(data[4])})')

	optim_dict = {
		'sgd': {
			'lr': 1e-3
		},
		'sgd_momentum': {
			'lr': 1e-3,
			'mu': 0.99
		},
		'sgd_nesterov': {
			'lr': 1e-3,
			'mu': 0.99,
			'nesterov': True
		},
		'sgd_weight_decay': {
			'lr': 1e-3,
			'mu': 0.99,
			'weight_decay': 1e-6
		},
		'adam': {
			'lr': 1e-3
		}
	}

	opt_labels = ['sgd', 'sgd_momentum', 'sgd_nesterov', 'sgd_weight_decay', 'adam']
	opt_losses = []

	def do_stuff(opt):
		net = MLP(num_features=784, num_hidden=64, num_outputs=10)

		if 'sgd' in opt:
			optimizer = optimizers.SGD(
				params=net.parameters(),
				**optim_dict[opt]
			)
		elif 'adam' in opt:
			optimizer = optimizers.Adam(
				params=net.parameters(),
				**optim_dict[opt]
			)

		return fit(net, data[:4], optimizer, num_epochs=args.num_epochs)

	for opt in opt_labels:
		opt_losses.append(do_stuff(opt))

	misc.plot_losses(opt_losses, labels=opt_labels, num_epochs=args.num_epochs, plot_epochs=True)
