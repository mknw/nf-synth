import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.nn.functional import normalize

import numpy as np
import pandas as pd


def load_cifar_test(args):
	# Note: No normalization applied, since RealNVP expects inputs in (0, 1).
	transform_train = transforms.Compose([
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor()
	])
	#torchvision.transforms.Normalize((0.1307,), (0.3081,)) # mean, std, inplace=False.
	transform_test = transforms.Compose([
		transforms.ToTensor()
	])
	trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
	trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
	testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
	testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
	return testloader

def load_mnist_test(args):
	transform_train = transforms.Compose([
		transforms.ToTensor()
		# transforms.ColorJitter(brightness=0.3)
	])
	#torchvision.transforms.Normalize((0.1307,), (0.3081,)) # mean, std, inplace=False.
	transform_test = transforms.Compose([
		transforms.ToTensor()
	])
	# trainset = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform_train)
	# trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
	testset = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform_test)
	testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
	return testloader

def load_network(model_dir, device, conf, checkpoint=True):
	if conf.arch == 'glow':
		from model import Glow
		net = Glow(3, conf.n_flows, conf.n_blocks, affine=conf.affine, conv_lu=not conf.no_lu)
		from train import calc_loss
		loss_fn = calc_loss
	elif conf.arch in ['densenet', 'resnet']:
		raise NotImplementedError

	net = net.to(device)
	if str(device).startswith('cuda'):
		net = torch.nn.DataParallel(net, conf.gpus)
		cudnn.benchmark = conf.benchmark

	# load checkpoint
	if checkpoint:
		checkpoint = torch.load(model_dir)
		try:
			net.load_state_dict(checkpoint['net'])
		except RuntimeError as re:
			print(re)
			raise ArchError('There is a problem importing the model, check parameters.')

	return net, loss_fn



class ArchError(Exception):
	def __init__(self, message=None):
		if not message:
			self.message = "State dictionary not matching your architecture. Check your params."
		else:
			self.message = message


class ModelNotFoundError(Exception):
	def __init__(self, path, message=None):
		self.path = path

		if message is not None:
			self.message = "Model not found. Change `resume` parameter to False?"
		self.path = path

		super().__init__(self.message)
	
	def __str__(self):
		return f'{self.path} -> {self.message}'



if __name__ == '__main__':
	raise NotImplementedError
