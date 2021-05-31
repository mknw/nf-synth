#!/var/scratch/mao540/miniconda3/envs/maip-venv/bin/python

import argparse
import os
import torch 
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
from tqdm import tqdm

from train import calc_z_shapes, find_last_model_relpath
from config.config import ConfWrap

def main(C):

	device = torch.device("cuda:0" if torch.cuda.is_available() and len(C.net.gpus) > 0 else "cpu")
	print("training on: %s" % device)
	start_epoch = 0
	# C.model_dir = C.root_dir + '/epoch_150000'
	C.model_dir = find_last_model_relpath(C.training.root_dir) # /model_{str(i + 1).zfill(6)}.pt'
	if os.path.isfile(f'{C.training.root_dir}/best_model'): 
		with open(f'{C.training.root_dir}/best_model', 'r') as bm:
			C.model_dir = bm.read()
	C.sampling.out_dir = f'{C.training.root_dir}/{C.sampling.out_dir}'

	os.makedirs(C.sampling.out_dir, exist_ok=True)

	if C.net.arch == 'glow':
		from model import Glow
		model = Glow(3, C.net.n_flows, C.net.n_blocks, affine=C.net.affine, conv_lu=C.net.lu_conv)
		net = model.to(device)

	if str(device).startswith('cuda'):
		net = torch.nn.DataParallel(net, C.net.gpus)
		cudnn.benchmark = C.training.benchmark

	# Load checkpoint.
	print('Resuming from checkpoint at ' + C.model_dir + '/model.pth.tar...')
	assert os.path.isdir(C.model_dir), 'Error: no checkpoint directory found!'
	checkpoint = torch.load(C.model_dir + '/model.pth.tar')
	net.load_state_dict(checkpoint['net'])
	# loss = checkpoint['test_loss']

	with torch.no_grad():
		for t in range(20):
			C.sampling.temp = t * 0.1
			sample_wrapper(C, net, device)


def sample_wrapper(C, net, device, noise=0, n_steps=0):
	net.eval()
	save_path = f'{C.sampling.out_dir}/n-{C.sampling.num_samples}_sample_t-{C.sampling.temp:.2f}'
	if C.sampling.img_size != C.training.img_size:
		save_path += f'_hw-{C.sampling.img_size}'
	if C.sampling.noise:
		net = net_noise(net, device, noise)
		save_path += f'_eps-{noise}'

	x = sample(net, device, C, norm_img='std')
	print(f'Stats for 1st image batch: std: {x.std()}, mean: {x.mean()}')
	
	if C.sampling.n_steps:
		save_path += f'_s-{C.sampling.n_steps}'
		for s_i in range(C.sampling.n_steps):
			# x = (x - x.mean()) / x.std()
			x = resample(x, net, device, C, norm_img ='std')
			print(f'step{s_i}: std: {x.std()}, mean: {x.mean()}')

	# x = (x - x.min()) / (x.max() - x.min())
	save_path += '.png'
	torchvision.utils.save_image(x, save_path, normalize=True,
			nrow=int(C.sampling.num_samples ** 0.5), scale_each=True,
			pad_value=255) # , range=(-0.5, 0.5))
	print(f'saved to {save_path}')

def sample(net, device, C, norm_img=True, exp=False):
	"""Sample from RealNVP model.

	Args:
		net (torch.nn.DataParallel): The RealNVP model wrapped in DataParallel.
		batch_size (int): Number of samples to generate.
		device (torch.device): Device to use.
	"""
	if not exp:
		z_sample = []
		z_shapes = calc_z_shapes(3, C.sampling.img_size, C.net.n_flows, C.net.n_blocks)
		for z in z_shapes:
			z_new = torch.randn(C.sampling.num_samples, *z) * C.sampling.temp
			z_sample.append(z_new.to(device))
	else:
		z_sample = torch.randn((C.sampling.num_samples, 3, C.sampling.img_size, C.sampling.img_size),
		                        dtype=torch.float32, device=device)
		z_sample = net(z_sample, partition=True)
	
	print(f"sampling Z with size: {C.sampling.img_size}x{C.sampling.img_size}.")
	x = net(z_sample, reverse=True)
	# import ipdb; ipdb.set_trace()
	if norm_img == 'img':
		x = (x.sub(x.view(x.shape[0], -1).mean(dim=1))) #  / x.std() # (x.max() - x.min())
	elif norm_img == 'std':
		std_x = x.view(x.shape[0], -1).std(dim=1)
		x = x.div(std_x[:, None, None, None])
	elif norm_img == 'center std':
		std_x = x.view(x.shape[0], -1).std(dim=1)
		x = x.sub(x.view(x.shape[0], -1).mean(dim=1)[:, None, None, None]).div(std_x[:, None, None, None])
	elif norm_img == 'batch':
		x = (x - x.min()) / (x.max() - x.min())
	return x

def resample(z, net, device=None, C=None, norm_img = True):

	z_sample = net(z, partition=True)

	x = net(z_sample, reverse=True)
	if norm_img == 'img':
		x = (x.sub(x.view(x.shape[0], -1).mean(dim=1))) #  / x.std() # (x.max() - x.min())
	elif norm_img == 'std':
		std_x = x.view(x.shape[0], -1).std(dim=1)
		x = x.div(std_x[:, None, None, None])
	elif norm_img == 'center std':
		std_x = x.view(x.shape[0], -1).std(dim=1)
		x = x.sub(x.view(x.shape[0], -1).mean(dim=1)[:, None, None, None]).div(std_x[:, None, None, None])
	elif norm_img == 'batch':
		x = (x - x.min()) / (x.max() - x.min())
	return x


def net_noise(net, device, loc=0, scale=0.005):
	with torch.no_grad():
		for param in net.parameters():
			try:
				param.add_(torch.randn(param.size()).to(device) * scale)
			except:
				continue
	return net

class GaussianNoise(object):

	def __init__(self, mean=0., std=.1, restrict_range=True):
		self.std = std
		self.mean = mean
		self.restrict_range = restrict_range

	def __call__(self, tensor):
		tensor += torch.randn(tensor.size()) * self.std + self.mean
		if self.restrict_range:
			return tensor.clamp(1e-8, 1)
		else:
			return tensor

	def __repr__(self):
		return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)



class Normie(object):
	'''class for normies'''
	def __init__(self, min, max):
		self.min = min
		self.max = max

	def __call__(self, tensor):
		tensor -= tensor.min()
		tensor /= tensor.max()
		return tensor

def find_last_epoch_model(fp):
	dirs_l = os.listdir(fp)
	dirs_e = [d for d in dirs_l if d.startswith('epoch_') 
		                             and d[-3:].isdigit()]
	dirs_e.sort()
	last_epoch = dirs_e[-1]
	print('Last model it.: ' + last_epoch)
	return fp + '/' + last_epoch

if __name__ == '__main__':
	conf_name = 'config/ffhq128_c.yml'

	parser = ArgumentParser(description='Utility for sampling from Gaussian latent space Z.')
	parser.add_argument('--config', '-c', default=conf_name, dest='config')
	ap = parser.parse_args()

	C = ConfWrap(fn=ap.config)
	C.resume = True
	main(C)
