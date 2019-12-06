import argparse
from copy import deepcopy

import torch

import misc
import optimizers
from networks import MLP, CNN, fit


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-num_epochs', type=int, default=50)
	parser.add_argument('-dataset', type=str, default='mnist')
	parser.add_argument('-num_train', type=int, default=4096)
	parser.add_argument('-num_val', type=int, default=2048)
	parser.add_argument('-lr_schedule', type=bool, default=True)
	parser.add_argument('-only_plot', type=bool, default=False)
	args = parser.parse_args()

	data = getattr(misc, 'load_'+args.dataset)(
		num_train=args.num_train,
		num_val=args.num_val
	)

	print(f'Loaded data partitions: ({len(data[0])}), ({len(data[1])})')

	opt_tasks = ['sgd', 'sgd_momentum', 'sgd_nesterov', 'sgd_weight_decay', 'adam', 'adamW', 'Radam', 'RadamW', 'nadam', 'lookahead_adam']
	opt_losses, opt_val_losses, opt_labels = [], [], []

	def do_stuff(opt):
		print(f'\nTraining {opt} for {args.num_epochs} epochs...')
		net = CNN() if args.dataset == 'cifar' else MLP()
		_, kwargs = misc.split_optim_dict(misc.optim_dict[opt])
		optimizer = misc.task_to_optimizer(opt)(
			params=net.parameters(),
			**kwargs
		)

		if 'lookahead' in opt.lower():
			optimizer = optimizers.Lookahead(optimizer, k=5, alpha=0.5)

		return fit(net, data, optimizer, num_epochs=args.num_epochs, lr_schedule=True)

	for opt in opt_tasks:
		if args.only_plot:
			losses = misc.load_losses(dataset=args.dataset, filename=opt)
			val_losses = misc.load_losses(dataset=args.dataset, filename=opt+'_val')
		else:
			losses, val_losses = do_stuff(opt)
			misc.save_losses(losses, dataset=args.dataset, filename=opt)
			misc.save_losses(val_losses, dataset=args.dataset, filename=opt+'_val')

		if losses is not None:
			opt_losses.append(losses)
			opt_val_losses.append(val_losses)
			opt_labels.append(misc.split_optim_dict(misc.optim_dict[opt])[0])

	if not torch.cuda.is_available():
		assert len(opt_losses) == len(opt_val_losses)
		misc.plot_losses(
			losses=opt_losses,
			val_losses=opt_val_losses,
			labels=opt_labels,
			num_epochs=args.num_epochs,
			title=args.dataset,
			plot_val=False,
			yscale_log=False,
			max_epochs=50
		)
