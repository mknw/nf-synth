#!/var/scratch/mao540/miniconda3/envs/maip-venv/bin/python

import argparse
import os
import torch 
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from numpy import log
# from models import RealNVP, RealNVPLoss
from model import Glow
from tqdm import tqdm
from shell_util import AverageMeter, bits_per_dim
from load_data import find_last_model_relpath

from config.config import ConfWrap
# from glob import glob
from shutil import rmtree
from torch.distributions.gamma import Gamma
from glob import glob
from argparse import ArgumentParser

def main(C):

	device = torch.device("cuda:0" if torch.cuda.is_available() and len(C.net.gpus) > 0 else "cpu")
	print("training on: %s" % device)
	start_epoch = 0


	# net, = load_network(model_fp, device, C.net)
	model = Glow(3, C.net.n_flows, C.net.n_blocks, affine=C.net.affine, conv_lu=C.net.lu_conv)
	net = model.to(device)
	if str(device).startswith('cuda'):
		net = torch.nn.DataParallel(net, C.net.gpus)
		cudnn.benchmark = C.training.benchmark

	if C.training.resume: # or not C.resume:
		# XXX changed
		# C.model_dir = find_last_model_relpath(C.training.root_dir) # /model_{str(i + 1).zfill(6)}.pt'
		if os.path.isfile(f'{C.training.root_dir}/best_model'): 
			with open(f'{C.training.root_dir}/best_model', 'r') as bm:
				C.model_dir = bm.readline().strip()
		print(f'Resuming from checkpoint at {C.model_dir}')
		checkpoint = torch.load(C.model_dir+'/model.pth.tar')
		net.load_state_dict(checkpoint['net'])
		try:
			C.training.best_loss = checkpoint['loss'] # XXX changed
		except:
			C.training.best_loss = checkpoint['test_loss'] # XXX remove
		start_epoch = checkpoint['epoch'] + 1 # avoid overwriting
		print(f"Resuming from epoch: {checkpoint['epoch']}"
		      f"Model loss: {C.training.best_loss}") # {checkpoint['loss']}")
	else:
		os.makedirs(C.training.root_dir, exist_ok=True)
		os.makedirs(C.training.sample_dir, exist_ok=True)

	optimizer = optim.Adam(net.parameters(), lr=float(C.training.learning_rate))
	if C.training.resume:
		optim_state_dict = torch.load(f'{C.model_dir}/optim.pt')
		optimizer.load_state_dict(optim_state_dict)
	
	if C.training.z_dist == 'normal':
		z_path = find_or_make_z(C.training.root_dir + '/z_samples.pkl',
	                          3, C.training.img_size, C.net.n_flows, C.net.n_blocks,
	                          C.training.n_samples, C.training.temp, device)
		C.training.z_path = z_path
	else:
		raise ValueError
	train(C.training, net, device, optimizer, start_epoch)


def calc_loss(log_p, logdet, image_size, n_bins):
	# log_p = calc_log_p([z_list])
	n_pixel = image_size * image_size * 3

	c = -log(n_bins) * n_pixel
	loss = c + logdet + log_p

	return (
		(-loss / (log(2) * n_pixel)).mean(),
		(log_p / (log(2) * n_pixel)).mean(),
		(logdet / (log(2) * n_pixel)).mean(),
	)

def train(config, net, device, optimizer, start_epoch, z_sample=None):
	''' training loop. `config` is a subfield of C '''
	if config.dataset == 'celeba':
		from load_data import sample_celeba
		dataset = iter(sample_celeba(config.batch_size, config.img_size))
	elif config.dataset == 'ffhq':
		from load_data import sample_from_directory
		if config.img_size > 128:
			dataset = iter(sample_from_directory('data/FFHQ/images1024x1024', config.batch_size,
				config.img_size, shuffle=True))
		else:
			dataset = iter(sample_from_directory('data/FFHQ/thumbnails128x128', config.batch_size,
				config.img_size, shuffle=True)) # , norm=norm))
	elif config.dataset == 'meyes':
		from load_data import sample_FFHQ_eyes
		from load_data import RandomRotatedResizedCrop as RRRC
		dataset = iter(sample_FFHQ_eyes(config.batch_size, config.img_size, shuffle=True,
		                                transform=RRRC(output_size=config.img_size)))

	best_loss = C.training.best_loss
	# best_loss = 2 # first iters
	figs = len(str(config.iter))
	n_bins = 2. ** config.n_bits
	quant_bpd = log(2) * (3 * config.img_size ** 2)

	loss_meter = AverageMeter()
	bpd_meter = AverageMeter()
	p_imgs = 0
	net.train()
	pbar = tqdm(range(start_epoch, config.iter))
	pbar.update(start_epoch); pbar.refresh()
	for i in pbar:
		x, _ = next(dataset)
		x = x.to(device)
		# if False:
			# # quantise
			# # x = x * 255
			# # if config.n_bits < 8:
			# # 	x = torch.floor(x / 2 ** (8 - config.n_bits))
			# # x * n_bins
			# x = torch.floor(x *  n_bins - 0.5)

		if i == 0:
			with torch.no_grad():
				log_p, logdet, _ = net(x + torch.rand_like(x) / n_bins)
				continue
		else:
			log_p, logdet, z_out = net(x + torch.rand_like(x) / n_bins)

		logdet = logdet.mean()

		loss, log_p, log_det = calc_loss(log_p, logdet, config.img_size, n_bins)
		net.zero_grad()
		loss.backward()
		# warmup_lr = C.lr * min(1, i * batch_size / (50000 * 10))
		warmup_lr = config.learning_rate
		optimizer.param_groups[0]['lr'] = warmup_lr
		optimizer.step()
		# TODO: invert meters
		loss_meter.update(loss.item(), x.size(0))
		bpd_meter.update(loss*quant_bpd)
		# logP: {log_p.item():.3f}; logdet: {log_det.item():.3f}; lr: {warmup_lr:.4f}; 
		pbar.set_description(
				f'BPD: {loss.item():.3f}; NLL: {loss.item()*quant_bpd:.3f}, imgs: {p_imgs}'
		)
		p_imgs += x.size(0)

		if i % 1000 == 0:
			del x
			# torch.cuda.empty_cache()
			# save model (note: can be factorised)
			if i % 10000 == 0:
				# TEST
				model_dir = f'{config.root_dir}/epoch_{str(i).zfill(figs)}'
				os.makedirs(model_dir, exist_ok=True)
				tenK_ep_ = glob(f"{config.root_dir}/epoch_*")
				tenK_ep_.sort()
			else:
				tenK_ep_ = []
				model_dir = config.root_dir
				
			if loss_meter.avg < best_loss or tenK_ep_:
				torch.save({'net': net.state_dict(), 'loss': loss_meter.avg,
				            'epoch': i},  f'{model_dir}/model.pth.tar')
				torch.save(optimizer.state_dict(), f'{model_dir}/optim.pt')
				print(f"Ep.{i}: Model with loss: {loss_meter.avg} saved to {model_dir}/model.pth.tar")
				if loss_meter.avg < best_loss:
					best_loss = loss_meter.avg
					with open(f'{config.root_dir}/best_model', 'w') as bm:
						bm.writelines([f'{model_dir}\n',
						               f'{i}\n',
						               f'{best_loss}\n'])
					# only remove previous checkpoints if loss improves.
					if len(tenK_ep_) > 5:
						for prev_epoch in tenK_ep_[:-4]:
							rmtree(prev_epoch) #, but keep N backup model(s)
			# Sampling.
			if config.z_dist == 'normal':
				z_sample = find_or_make_z(config.z_path, reuse=True)
			else: raise NotImplementedError
			png_filename = f'{config.sample_dir}/{str(i).zfill(figs)}.png'
			with torch.no_grad():
				torchvision.utils.save_image(net(z_sample, reverse=True).cpu().data,
				                             png_filename, normalize=True,
				                             nrow = int(config.n_samples ** 0.5))
			del z_sample
			# torch.cuda.empty_cache()
			with open(f'{config.root_dir}/log', 'a') as l:
				report = f'{loss_meter.avg:.5f},{bpd_meter.avg},{log_p.item():.5f},{p_imgs}\n'
				l.write(report)
			print(f'Sampling done. Log saved.')
			# XXX change (every n?)
			loss_meter = AverageMeter()
			bpd_meter = AverageMeter()



def make_z_gamma(mean, std, channels, img_size, n_flows, n_block, num_sample, t, device):
	shape = (mean / std) ** 2
	rate = mean / std ** 2
	print(f'Method of moments shape parameter: {shape}.', end='')
	# if shape < 1:
	shape = 2
	rate = 0.7
	z_sample = []
	z_shapes = calc_z_shapes(channels, img_size, n_flows, n_block)
	for z in z_shapes:
		z_new = Gamma(shape, rate).sample(torch.Size([num_sample, *z])) * t
		z_sample.append(z_new.to(device))
	return z_sample, shape, rate

def find_or_make_z(path, channels=3, img_size=128, n_flows=4, n_block=16,
	                num_sample=16, t=0.7, device=None, reuse=False):

	if os.path.isfile(path) and reuse:
		z_sample = torch.load(path)
		return z_sample
	else:
		z_sample = []
		z_shapes = calc_z_shapes(channels, img_size, n_flows, n_block)
		for z in z_shapes:
			z_new = torch.randn(num_sample, *z) * t
			z_sample.append(z_new.to(device))
		torch.save(z_sample, path)
		return path

def calc_z_shapes(n_channel, input_size, n_flows, n_block):
	z_shapes = []

	for i in range(n_block - 1):
		input_size //= 2
		n_channel *= 2
		z_shapes.append((n_channel, input_size, input_size))

	input_size //= 2
	z_shapes.append((n_channel * 4, input_size, input_size))
	return z_shapes


def save_imgrid(tensor, name):
	grid = torchvision.utils.make_grid(tensor, nrow=int(tensor.shape[0] ** 0.5), padding=1, pad_value=255)
	torchvision.utils.save_image(grid, name)
	return

def sample(net, num_samples, in_channels, device, resize_hw=None):
	"""Sample from RealNVP model.

	Args:
		net (torch.nn.DataParallel): The RealNVP model wrapped in DataParallel.
		batch_size (int): Number of samples to generate.
		device (torch.device): Device to use.
	"""
	
	if not resize_hw:
		side_size = 28
	else:
		side_size, side_size = resize_hw
	print(f"sampling with z space sized: {side_size}x{side_size}.")
	z = torch.randn((num_samples, in_channels, side_size, side_size), dtype=torch.float32, device=device) #changed 3 -> 1
	x, _ = net(z, reverse=True)
	return x, z


def test(epoch, net, testloader, device, loss_fn, **C):
	global best_loss
	net.eval()
	loss_meter = util.AverageMeter()
	bpd_meter = util.AverageMeter()
	with torch.no_grad():
		with tqdm(total=len(testloader.dataset)+1) as progress_bar:
			for x, _ in testloader:
				x = x.to(device)
				z, sldj = net(x, reverse=False)
				loss = loss_fn(z, sldj)
				loss_meter.update(loss.item(), x.size(0))
				# bits per dimensions
				bpd_meter.update(util.bits_per_dim(x, loss_meter.avg), x.size(0))

				progress_bar.set_postfix(loss=loss_meter.avg,
																 bpd=bpd_meter.avg)
				progress_bar.update(x.size(0))
				
	# Save checkpoint
	save_dir = C['dir_samples'] + '/epoch_{:03d}'.format(epoch) #  + str(epoch)
	os.makedirs(save_dir, exist_ok=True)

	# if loss_meter.avg < best_loss or epoch % 10 == 0 or
	# 		epoch > 100 or epoch < 20:
	if True:
		print('\nSaving...')
		state = {
			'net': net.state_dict(),
			'test_loss': loss_meter.avg,
			'epoch': epoch,
		}
		torch.save(state, save_dir + '/model.pth.tar')
	C.training.resume = True
	best_loss = loss_meter.avg

	sample_fields = ['num_samples', 'in_channels', 'resize_hw']
	images, latent_z = sample(net, device=device, **filter_args( C, fields=sample_fields ) )

	# plot x and z
	num_samples = C['num_samples']
	images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
	z_concat = torchvision.utils.make_grid(latent_z, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
	torchvision.utils.save_image(images_concat, save_dir+'/x.png')
	torchvision.utils.save_image(z_concat, save_dir+'/z.png')

	# with open(, 'wb') as z_serialize:
	# 	pickle.dump(latent_z, z_serialize)
	torch.save(latent_z, f = save_dir+'/z.pkl')

	# dict keys as returned by "train"
	train_loss = C['train_loss']
	train_bpd = C['train_bpd']
	report = [epoch, loss_meter.avg, bpd_meter.avg] + [train_loss, train_bpd]

	dir_samples = C['dir_samples']
	with open('{}/log'.format(dir_samples), 'a') as l:
		report = ", ".join([str(m) for m in report])
		report += "\n"
		print("\nWriting to disk:\n" + report + "At {}".format(dir_samples))
		l.write(report)


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




if __name__ == '__main__':

	# C = ConfWrap(fn='config/ffhq256lu_c.yml')
	# C = ConfWrap(fn='config/glow_celeba_aff.yml')
	# C = ConfWrap(fn='config/ffhq64_gamma_c.yml')
	# import ipdb; ipdb.set_trace()
	conf_name = 'config/celeba128_c.yml'
	parser = ArgumentParser(description='Glow training on different datasets.')
	parser.add_argument('--config', '-c', default=conf_name, dest='config')
	ap = parser.parse_args()

	C = ConfWrap(fn=ap.config)

	C.training.sample_dir = C.training.root_dir + '/samples'
	main(C)
