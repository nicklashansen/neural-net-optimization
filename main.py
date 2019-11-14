import argparse
from copy import deepcopy

import torch

import misc
import optimizers
from networks import MLP, CNN, fit


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-num_epochs', type=int, default=100)
	parser.add_argument('-only_plot', type=bool, default=False)
	args = parser.parse_args()

	data = misc.load_cifar()
	print(f'Loaded data partitions: ({len(data[0])}), ({len(data[1])})')

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
		},
		'adamW':{
			'lr': 1e-3,
			'weight_decay': 1e-4
		},
		'adam_l2':{
			'lr': 1e-3,
			'l2_reg': 1e-4
		},
		'Radam': {
			'lr': 1e-2,
			'rectified': 1
		},
		'RadamW': {
			'lr': 1e-2,
			'rectified': 1,
			'weight_decay': 1e-4
		}
	}

	opt_labels = ['sgd', 'sgd_momentum', 'sgd_nesterov', 'sgd_weight_decay', 'adam', 'adamW', 'Radam', 'RadamW']
	opt_losses, opt_val_losses = [], []

	def do_stuff(opt):
		print(f'\nTraining {opt} for {args.num_epochs} epochs...')
		net = CNN()
		opt_class = getattr(optimizers, 'SGD' if 'sgd' in opt else 'Adam')
		optimizer = opt_class(
			params=net.parameters(),
			**optim_dict[opt]
		)

		return fit(net, data, optimizer, num_epochs=args.num_epochs)

	for opt in opt_labels:
		if args.only_plot:
			losses = misc.load_losses(filename=opt)
			val_losses = misc.load_losses(filename=opt+'_val')
		else:
			losses, val_losses = do_stuff(opt)
			misc.save_losses(losses, filename=opt)
			misc.save_losses(val_losses, filename=opt+'_val')

		opt_losses.append(losses)
		opt_val_losses.append(val_losses)

	if not torch.cuda.is_available():
		misc.plot_losses(opt_losses, opt_val_losses, labels=opt_labels, num_epochs=args.num_epochs, plot_epochs=False)
