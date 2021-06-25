#!/var/scratch/mao540/miniconda3/envs/maip-venv/bin/python

import torch 

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from umap import UMAP
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import BoundaryNorm as BoundaryNorm

# import util
import argparse
import os
import shutil
from random import randrange
from numpy.linalg import svd

class  Synthesizer(object):
	def __init__(self, C, steps = ['net', 'pca', 'umap'],
							net=None, device=None,
							store_config=True, quantise_fun=None, svd_solver='auto'):

		np.random.seed(24)
		if isinstance(steps, str):
			steps = [steps]
		elif not isinstance(steps, list):
			print('steps is not type list nor string:'
					f'{type(steps)} instead.')
			raise ValueError

		self.steps = steps
		self.n_steps = len(steps)
		self.models = dict()

		if 'net' in steps:
			assert net is not None, 'net object not provided.'
			assert steps.index('net') == 0, 'Net must be first, if present.'
			self.models['net'] = net
			# self.net_state = 'trained'
			steps = steps[1:]
			if device is not None:
				self.device = device
			else:
				raise ValueError(f'Neural net was given, but device not provided')

		for step in steps:
			if step == 'pca':
				self.models[step] = PCA(n_components=C.pca_n_pcs, whiten = C.pca_whiten,
						                  random_state=42, svd_solver=svd_solver)
			if step == 'svd':
				self.models[step] = batch_SVD(n_components=C.pca_n_pcs, whiten=C.pca_whiten,
						img_size = C.img_size, img_channels = C.in_channels, tile_RGB=C.tile_RGB, 
						quantise_fun = quantise_fun)
			if step == 'umap':
				self.models[step] = UMAP(n_neighbors = C.umap_nn, min_dist = C.umap_min_dist,
			                    n_components = C.umap_n_comps, random_state=42)
		if store_config:
			self.C = C
			self.channels = C.in_channels
			self.img_size = C.img_size
			self.batch = C.batch_size


	def fit(self, data, skip_l=['net']):
		''' Main fit method .
		Args: 
			- data: numpy array of size (n x p) '''
		
		for step in self.steps:
			if step in skip_l:
				print(f'{step.capitalize()} already trained. Skipping..')
				continue
			else:
				print(f'fitting {step}')
				if step is not self.steps[-1]:
					data = self.models[step].fit_transform(data)
				else:
					self.models[step].fit(data)
		del data
		return self

	def transform(self, data, show_steps='all', skip_l=['net']):
		''' Main transform method.
		Args:
			- data: numpy array of size (n x p), 
			           can be different than the one given to .fit()'''
		# TODO: manage 'levels' for returning reductions at different steps.
		selected_steps = self.select_steps(show_steps)
		out = list()
		for step in self.steps:
			if step in skip_l:
				print(f'Skipping {step}...')
				continue
			else:
				print(f'{step.capitalize()} transform...', end='')
				if step == 'net':
					data = self.net_infer(data)
				else:
					data = self.models[step].transform(data)
				if step in selected_steps:
					if len(selected_steps) == 1:
						return data
					else:
						out.append(data)
		return out

	def inverse_transform(self, data, show_steps='all', resample=True, skip_l=[None]):
		selected_steps = self.select_steps(show_steps)
		out = list()
		for step in reversed(self.steps):
			if step in skip_l:
				print(f'Skipping {step}...')
				continue
			else:
				print(f'{step.capitalize()} inverse transform...')
				if step == 'net':
					data = self.net_generate(data, resample=resample)
				else:
					data = self.models[step].inverse_transform(data)
				if step in selected_steps:
					out.append(data)
					if len(selected_steps) == 1:
						return out[0]
		return out

	def fit_transform(self, data, show_steps='all', skip_l=['net']):
		''' Main fit method .
		Args: 
			- data: numpy array of size (n x p) '''
		selected_steps = self.select_steps(show_steps)
		out = list()
		for step in self.steps:
			if step in skip_l:
				print(f'{step.capitalize()} already trained. Skipping..')
				continue
			else:
				print(f'{step.capitalize()} fit-transform...', end='')
				if step == 'net': # often skipped
					data = self.net_infer(data)
				else:
					data = self.models[step].fit_transform(data)
				if step in selected_steps:
					out.append(data)
		# ADD QUANTISATION
		return out
	
	def select_steps(self, show_steps):
		if isinstance(show_steps, int):
			show_steps = [show_steps]
		elif show_steps=='all':
			show_steps = [i for i in range(len(self.steps))]
		elif not isinstance(show_steps, list):
			raise ValueError(f'Cannot show steps of type: {type(show_steps)}')
		return [self.steps[step_idx] for step_idx in show_steps]
	
	def net_infer(self, input_data):
		# actual transformation
		# TODO: implement loader  (load_data.load_celeba)
		if input_data.shape[0] <= self.batch:
			batch_size = 1
		else:
			batch_size = self.batch
		sample_size = input_data.shape[0]

		if sample_size % batch_size:
			samples_in_excess = sample_size % batch_size
			print(f'synthesizer.net_ infer() : trimming last {samples_in_excess} datapoints.')
			input_data = input_data[:-samples_in_excess]
		
		
		T = input_data.reshape(
		        -1,
		        batch_size,
		        self.channels,
		        self.img_size,
		        self.img_size
		   )
		
		data_out = np.array([]).reshape(0, input_data.shape[1])

		for batch in T:
			b = torch.from_numpy(batch).to(self.device)
			try:
				log_p, log_det, imgs = self.models['net'](b)
			except RuntimeError as re:
				print(str(re), b.shape)
				raise RuntimeError

			# reconstruct output data
			imgs = self.models['net'](
		                         imgs, reverse=True, reconstruct=True
		                         ) .cpu().detach().numpy(
		                         ).reshape(-1, input_data.shape[1])

			data_out = np.concatenate([data_out, imgs])
		return data_out
	
	def net_generate(self, input_data, resample=True):
		''' f(z) -> x
		args: input_data (n x p),
		resample: (false if input_data is novel (e.g. randomly created), else true)'''
		# partition input data
		# if data.shape[0] > self.batch:
		# 	data.reshape(-1, self.channels, self.img_size, self.img_size)

		with torch.no_grad():
			data = self.models['net'](
			                          torch.from_numpy(
			                            input_data.reshape(
			                              input_data.shape[0],
			                              self.channels,
			                              self.img_size,
			                              self.img_size
			                              ) # .astype(np.float32)
			                            ).to(self.device),
			                          partition=True
			                         )
			# actual "inverse" transformation
			data = self.models['net'](
			                          data, 
			                          reverse=True,
			                          resample=resample
			                          ).reshape(input_data.shape[0], -1)
		return data.cpu().detach().numpy()


class batch_SVD:
	'''
	Perform dim. red. for individual images. 
	'''
	
	def __init__(self, n_components, whiten, img_size, img_channels,
		          tile_RGB=False, quantise_fun=None, random_state=123129387):
		self.n_components = n_components
		self.whiten = whiten
		if self.whiten:
			warn("batch_SVD with whiten=True is unstable. Please use plain version.")
		self.center = True
		# TODO: change param names to match.
		self.im_s = img_size
		self.im_ch = img_channels
		self.tile_RGB = tile_RGB
		self.quantise_fun = quantise_fun
	
	def fit(self, X, y=None):

		X = self._preproc_img(X)
		U, S, Vt = self._fit_full(X, self.n_components)
		# U = U[..., :self.n_components]
		return self

	def fit_transform(self, X, y=None):
		
		# X = self._preproc_img(X)
		U, S, Vt = self._fit_full(X)

		if self.whiten:
			# changed this for stacked pca
			U *= np.sqrt(self.im_s - 1)
		else:
			U *= S[:, np.newaxis, :self.n_components]
		# if self.quantise_fun:
		# 	U = self.quantise_fun(U)
		return U

	def _preproc_img(self, X):
		
		if X.shape[0] == 1:
			raise ValueError("Only batch of images accepted")
		
		n_imgs = X.shape[0]
		if not self.tile_RGB:
			if len(X.shape) == 2 or len(X.shape) == 4:
				X = X.reshape(n_imgs*self.im_ch, self.im_s, self.im_s)
		elif self.tile_RGB == 'v':
			X = X.reshape(n_imgs, self.im_s * self.im_ch, self.im_s)
		elif self.tile_RGB == 'h':
			X = X.reshape(n_imgs, self.im_s, self.im_s * self.im_ch)
		else:
			raise ValueError

		if not 0 < self.n_components <= min(X.shape[-2:]):
			raise ValueError
		
		return X

	def _fit_full(self, X):
		X = self._preproc_img(X)

		if self.center:
			mean_ = X.mean(-2, keepdims=True)
			X -= mean_

		n_components = self.n_components

		# Using numpy.linalg.svd which computes SVD for each matrix in the stack.
		U, S, Vt = svd(X, full_matrices=False)
		
		# TODO: flip eigenvectors' sign to enforce deterministic output.
		# U, Vt = batch_svd_flip_(U, Vt)

		if self.quantise_fun:
			print("performing quantisation at SVD level.")
			self.components_, bits_Vt = self.quantise_fun(Vt[:, :n_components, :])
			U, bits_U = self.quantise_fun(U[..., :n_components])
			self.bit_count = bits_Vt + bits_U
		else:
			self.components_ = Vt[:, :n_components, :]
			U = U[..., :n_components]

		sample_size = U.shape[1]
		explained_variance_ = S**2 / sample_size
		total_var = explained_variance_.sum(axis=1)
		explained_variance_ratio_ = explained_variance_ / total_var[:, np.newaxis]
		# singular_values_ = S.copy()

		# noise variance for each image.
		self.noise_variance_ = (np.square(S[:, n_components:]) / (sample_size - 1)).mean(axis=-1)
		self.explained_variance_ = explained_variance_
		self.mean_ = mean_
		return U, S, Vt


	def _postproc(self, X):
		if not self.tile_RGB:
			return X.reshape(int(X.shape[0]/self.im_ch), self.im_ch, self.im_s, self.im_s)
		else:
			return X.reshape(X.shape[0], self.im_ch, self.im_s, self.im_s)


	def transform(self, X):
		raise NotImplementedError
		X = self._preproc_imgs(X)
		X -= self.mean_
		
		X_t = X @ self.components_.transpose(axes=(0, 2, 1))
		if self.whiten:
			X_t /= np.sqrt(self.explained_variance_)
		return X_t

	def inverse_transform(self, X):
		if self.whiten:
			scaled_components = np.sqrt(self.explained_variance_[:, :, np.newaxis]) * \
		            self.components_
			X_t = X @ scaled_components
			X_t += self.mean_
			return self._postproc(X_t)
		else:
			X_t = X @ self.components_ + self.mean_
		return self._postproc(X_t)


def batch_svd_flip_(u, v, u_based_decision=True):
	""" From: `scikit.utils.extmath.svd_flip`; but for `len(u,v.shape)` > 2.
	Sign correction to ensure deterministic output from SVD.
	Adjusts the columns of u and the rows of v such that the loadings in the
	columns in u that are largest in absolute value are always positive.

	Parameters
	----------
	u : ndarray
	    u and v are the output of `linalg.svd` or
	    :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
	    dimensions so one can compute `np.dot(u * s, v)`.
		 
	v : ndarray
	    u and v are the output of `linalg.svd` or
	    :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
	    dimensions so one can compute `np.dot(u * s, v)`.
	    The input v should really be called vt to be consistent with scipy's
	    ouput.

	u_based_decision : bool, default=True
	    If True, use the columns of u as the basis for sign flipping.
	    Otherwise, use the rows of v. The choice of which variable to base the
	    decision on is generally algorithm dependent.
	

	Returns
	-------
	u_adjusted, v_adjusted : arrays with the same dimensions as the input.
	
	"""
	import ipdb; ipdb.set_trace()
	# u = u.reshape(u.shape[0] * u.shape[1], u.shape[2])

	if u_based_decision:
		# columns of u, rows of v
		max_abs_cols = np.argmax(np.abs(u), axis=1)
		signs = np.sign(u[max_abs_cols[:,:,np.newaxis], range(u.shape[1])])
		u *= signs
		v *= signs[:, np.newaxis]
	else:
		raise NotImplementedError
		# rows of v, columns of u
		max_abs_rows = np.argmax(np.abs(v), axis=1)
		signs = np.sign(v[:, range(v.shape[0]), max_abs_rows])
		u *= signs
		v *= signs[:, np.newaxis]
	return u, v
