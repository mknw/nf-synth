#!/bin/env /var/scratch/mao540/miniconda3/envs/revive/bin/python

#/var/scratch/mao540/miniconda3/envs/maip-venv/bin/python
import torch 
import os
# from mnist_simil import cleanup_version_f
from config.config import ConfWrap
from synthesyzer  import Synthesizer

from utils import ArchError
from load_data import load_network, Attributes, select_model
import pickle
import numpy as np
# Plot analysis:
from render import save_dataset_reduction, plot_compression_flow
from similarity import track_z
from load_data import load_celeba, load_ffhq
from glob import glob
from argparse import ArgumentParser

import datetime as dt

def main(C, epoch=4340000, save=True):

	p_value_counts = dict()
	model_meta_stuff = select_model(C.training.root_dir, select_epoch='auto', figures=7)

	config = C
	for tile_rgb in ['h']: # False, 'h', 'v']: # , 'h', 'v']:
		for q in [8, 7, 6, 5]:
			config.compress.quantise = q
			config.compress.tile_RGB = tile_rgb
			analyse_synthesizer(config, model_meta_stuff)


def analyse_synthesizer(C, model_meta_stuff = None):

	device = torch.device("cuda:0" if torch.cuda.is_available() and len(C.net.gpus) > 0 else "cpu")
	
	model_root_fp, model_fp, vmarker_fp = model_meta_stuff
	# if only plotting is needed and all vecs are stored, do not load_net 
	if not C.compress.no_model:
		net, _ = load_network( model_fp, device, C.net)
	epoch = model_root_fp.split('_')[-1]
	# Load statistics:

	s_str = '-'.join(map(lambda x: x.lower(), C.compress.data)) # stats-string
	data_fn = f"{s_str}_{C.compress.dataset}.npz"
	archive_filename = f'{model_root_fp}/{data_fn}'
	# C.data_fn = data_fn
	if C.compress.use_data_archive:
		if os.path.isfile(archive_filename):
			dataset = np.load(archive_filename)
			print(f'Loading saved arrays: `{s_str}`', end='')
			print(f' from {data_fn}`')
		else:
			potential_archive_fn = glob(f'{model_root_fp}/*{data_fn}')
			if len(potential_archive_fn)==1:
				warn(f"Did not find exact match for {archive_filename}."
			        f"Loading {potential_archive_fn[0]} instead.")
				dataset = np.load(potential_archive_fn[0])
	else:
		# Minimize the need to compute X and Z for the full test set;
		# Create archives instead.
		if C.compress.dataset.lower() == 'celeba':
			loader = load_celeba(128, C.training.img_size, test=True, shuffle=False)
		elif C.compress.dataset.lower() == 'ffhq':
			loader = load_ffhq(C.training.batch_size,
			                  C.training.img_size, test=True, shuffle=False)
		dataset = track_z(net, device, C.training.img_size, loader, stats=C.compress.data)
		if C.compress.archive_data:
			np.savez(archive_filename, **dataset)

	''' filepaths '''
	# compr_fp = model_root_fp + '/compr'

	data_z = dataset['Z'].reshape(dataset['Z'].shape[0], -1)
	data_x = dataset['X'].reshape(dataset['X'].shape[0], -1)
	if C.compress.dataset == 'celeba':
		attributes = Attributes(dataset='celeba')
	elif C.compress.dataset == 'ffhq':
		# mockup 1 column of ones.
		attributes = Attributes(dataset='ffhq')

	## make cache directory for all analyses (PCA red. + reupsampling).
	if C.compress.rans:
		root_d = f'{C.training.root_dir}/rANS_compress_{C.compress.dataset}'
	else:
		root_d = f'{C.training.root_dir}/compress_{C.compress.dataset}'
	if 'svd' in C.compress.steps[0]:
		root_d += '_svd'
	if C.compress.tile_RGB:
		root_d += f'/rgb_{C.compress.tile_RGB}'
	os.makedirs(root_d, exist_ok=True)
	# C.cache_bn = make_basename(root_d, cache=True)

	# if not C.compress.no_model:
	# 	C.compress.pca.n_pcs = C.compress.pca.n_pcs_sc
	# 	reducer = Synthesizer(C, steps = C.compress.steps[0],
	#                       net=net, device=device)
	# C.basename = make_basename(root_d, subdir='pca', pca_pcs=C.pca.n_pcs_sc)
	# pca_reduction_on_attributes(C, reducer, data_z, attributes)
	kept_out_l = [True, False]
	steps_l = C.compress.steps
	dataset_ratio_l = C.compress.dataset_ratios
	quantise_fun = quantisation_simulator(quantise_bits=C.compress.quantise,
	                                      rans=C.compress.rans)
	# re-upsampling/reconstruction visualization.
	for stps in steps_l: 
		# select dim. red. parameters
		if 'umap' in stps:
			reup_umap_npc = [i for i in range(*eval(C.compress.umap_n_comps))]
		else:
			reup_umap_l = ['empty']
		if 'pca' in stps:
			pca_npc_l = [int(i**3.5) for i in range(2, 17)]
			pca_npc_l += [data_z.shape[0]-16]
		elif 'svd' in stps:
			# pca_npc_l = [int(C.compress.img_size / r) for r in range(1, 11)]
			pca_npc_l = [int(i**1.6) for i in range(5, 19)]
			kept_out_l = [False]
			dataset_ratio_l = [1.] # HaCk
		for umap_pcs in reup_umap_l:
			# if not C.compress.no_model:
			for pca_pcs in pca_npc_l:
				for ds in dataset_ratio_l:
					for kept_out in kept_out_l:
						if (kept_out and pca_pcs > int(ds * data_z.shape[0])-16) \
						   or (not kept_out and pca_pcs > int(ds * data_z.shape[0])):
							continue
						# in function use
						C.steps = stps # !
						C.ds_ratio = ds
						C.kept_out = kept_out
						# for Synth use:
						C.compress.umap_n_comps = umap_pcs
						C.compress.pca_n_pcs = pca_pcs
						# make folders
						C.basename = make_basename(root_d, config=C)
						# C.cache_bn = make_basename(root_d, config=C, cache=True)
						print('Analysing synth with\n'
						      f'PCA nps:{pca_pcs}, UMAP dims: {umap_pcs}, '
						      f'ko:{C.kept_out}, ratio:{ds}')
						if not C.compress.no_model:
							reducer = Synthesizer(C.compress, steps=stps, net=net,
						                           device=device, quantise_fun=quantise_fun)
							compute_reduction_reupsampling(C, attributes, data_z.copy(),
							                       data_x.copy(), reducer,
							                       quantise_bits=C.compress.quantise, seed=69)
						else:
							compute_reduction_reupsampling(C, attributes, data_z.copy(),
							                       data_x.copy(), quantise_bits=C.compress.quantise)


def make_basename(root, mark_time=False, subdir=None, pca_pcs=None, umap_pcs=None, cache=False,
		quantise_bits=None, add_log=True, config=None):
	# extract arguments from `config` dictionary.
	if quantise_bits is None:
		quantise_bits = config.compress.quantise
	if pca_pcs is None:
		pca_pcs = config.compress.pca_n_pcs
	if umap_pcs is None:
		umap_pcs = config.compress.umap_n_comps
	if subdir is None:
		subdir = C.ds_ratio

	# chain string
	time = ''
	basename = root 
	# two subdirectories
	if quantise_bits:
		basename = f'{root}/q{quantise_bits}'
	else:
		basename = f'{root}/no-q'
	if subdir:
		# prefix (used for ratio in reupsam, otherwise analysis type ('pca'))
		if isinstance(subdir, float):
			# basename += f'/rat{subdir}' # XXX !!! XXX
			if quantise_bits:
				basename += f'-rat{subdir}' # XXX !!! XXX
			else:
				basename += f'/rat{subdir}' # XXX !!! XXX
		else:
			basename += f'/{subdir}'
	if cache:
		basename += '/cache'
	os.makedirs(basename, exist_ok=True)
	# filename prefix
	if mark_time or add_log: 
		t = str(dt.datetime.now()).split(sep='.')[0].replace('2021-', '').replace(' ', '_')
		if mark_time:
			basename += t
		if add_log:
			C.dump(f'{basename}/log-{t}.yaml')
	basename += f'/syn'
	if umap_pcs and (umap_pcs != 'empty'):
			basename += f'_uc{umap_pcs}'
	if pca_pcs:
		basename += f'_pc{pca_pcs}'
	print(f"basename: {basename}")
	return basename


def pca_reduction_on_attributes(C, reducer, data, attributes):
	# TODO 1 move reducer.fit() method outside of function
	# TODO 2 test for cases where len(att_ind) > 1.
	reducer.fit(data)
	''' x -> low_dim '''
	# TODO: IN- and OUT- of set reduction / transformation.
	var_exp_ratio = reducer.models['pca'].explained_variance_ratio_
	data_red = reducer.transform(data)[0]
	# C.basename
	for i in range(0, 39, 5):
		fn = f'{C.basename}_att{i}.png'
		save_dataset_reduction(data_red, var_exp_ratio, attributes, k=10, att_ind=i, filename=fn)

def compute_reduction_reupsampling(C, attributes, data=None, data_x=None, reducer=None,
		quantise_bits=8, seed=1823904943):
	''' Save x's and z's, visualized pre (original) 
	and post (reconstructed) dimensionality reduction.'''
	rng = np.random.default_rng(seed=seed)
	if attributes.dataset == 'celeba': # stitch filename together (till 194)
		# Categ's: blond, brown air, smiling, wearing hat.
		att_ind = [5, 11, 31, 35] # XXX !!! XXX
		att_ind = [i + 1 for i in att_ind]
		# att_ind = [1, 3, 25, 29]
		att_ind = [4, 12, 19, 33]
		att_str = '-'.join(map(str, att_ind)) # [str(a) for a in att_ind])
	elif attributes.dataset== 'ffhq':
		att_ind = 0
		att_str = 'test-split'
	ko = '_ko' if C.kept_out else ''
	ds_trunk_idx = int(data.shape[0] * C.ds_ratio)
	ds_rat = f'_{ds_trunk_idx}' if not isinstance(C.ds_ratio, float) else ''
	filename = f'{C.basename}_{att_str}{ko}{ds_rat}.png'
	# cache_fn = f'{C.cache_bn}_{att_str}{ko}{ds_rat}.npz'

	# middle two keys change depending on whether UMAP is used.
	step_vec_keys = ['X', 'Z', 'PC/eigenZ', 'PC/UMAP', 'rec_Z', 'rec_X']

	# def resample_archive(C, attributes, data, data_x, reducer, quantise_bits):
	if C.compress.dataset == 'celeba':
		kept_out_df, kept_out_idcs = attributes.pick_last_n_per_attribute(att_ind, n=4)
		att_names = list(kept_out_df.columns)
	elif C.compress.dataset == 'ffhq':
		kept_out_idcs = rng.choice(5000, 16, False)
		att_names = [f'row {n}' for n in range(1, 5)]

	if C.compress.use_step_archive and False: #  and os.path.isfile(cache_fn):
		step_vector = list()
		with np.load(cache_fn) as data:
			for k in step_vec_keys:
				step_vector.append(data[k])
	elif not C.compress.no_model:
		# att_ind = list(range(0, 40, 10))
		# split dataset
		z_s = data[kept_out_idcs].copy()
		x_s = data_x[kept_out_idcs].copy()

		if 'svd' in C.steps:
			red_data = reducer.fit_transform(z_s)
		else:
			if C.kept_out:
				data = np.delete(data, kept_out_idcs, axis=0)
			# rng.shuffle(data, axis=0)
			if ds_trunk_idx < data.shape[0]:
				data = data[:ds_trunk_idx]
		
			reducer.fit(data)
			# TODO: replace show_steps arguments with argument selection
			red_data = reducer.transform(z_s, show_steps='all')

		# the last element of red(uced)_data is the lower level representation.
		if quantise_bits:
			if 'svd' not in C.steps:
				red_data[-1] = sim_quantisation(red_data[-1])
			rec_data = reducer.inverse_transform(red_data[-1], show_steps='all',
				                               resample=C.compress.resample_z)
		else:
			rec_data = reducer.inverse_transform(red_data[-1], show_steps='all',
		                                  resample=C.compress.resample_z)

		del data_x; del data
		if 'umap' in C.steps:
			step_vector = [x_s, z_s] + red_data + rec_data[1:]
			outer_grid = (3, 2)
		elif 'pca' in C.steps:
			step_vector = [x_s, z_s] + [reducer.models['pca'].components_] \
			               + red_data + rec_data
			outer_grid = (3, 2)
		elif 'svd' in C.steps:
			step_vector = [x_s, z_s] + rec_data
			outer_grid = (2, 2)
		if C.compress.archive_step:
			np.savez(cache_fn, **dict(zip(step_vec_keys, step_vector)))
	else:
		raise RuntimeError

	plot_compression_flow(step_vector, filename, att_names, C.steps, outer_grid=outer_grid)
	print('done.')
	
def quantisation_simulator(quantise_bits=8, axis=(1, 2), rans=False):
	def inner_quantisation(data):
		return sim_quantisation(data, quantise_bits, axis, rans=rans)
	return inner_quantisation

def sim_quantisation(data, quantise_bits, axis=0, rans=False):
	filename = f'data/glow_ffhq128/rANS_compress_ffhq/{dt.datetime.now().isoformat()}_rANS_log.txt'
	assert isinstance(quantise_bits, int)
	quantise_lvls = 2 ** quantise_bits - 1
	# q_data = red_data[-1].copy()
	data_std = data.std(axis=axis, keepdims=True)
	data /= data_std
	data_min = data.min(axis=axis, keepdims=True)
	data -= data_min
	data[data == 0.0] = 0.0000000000001
	bits_ratio = quantise_lvls/data.max(axis=axis, keepdims=True)
	data *= bits_ratio
	q_data = np.rint(data)
	if rans: 
		from rANSCoder import Encoder, Decoder
		from time import time
		q_data = q_data.astype(np.int16)
		# express probability of integer quantised values as frequencies
		uni_vals, freq_vals = np.unique(q_data, return_counts=True)
		missing_vals = [i for i in range(quantise_lvls) if i not in list(uni_vals)]
		
		probs = freq_vals / q_data.size
		for m in missing_vals:
			# give lowest prob. to integers in range 0-2^(N bits), but not appearing in Z.
			# XXX Is this what give the wrong number of bits in decoding phase?
			probs = np.insert(probs, m, 0.000000000001)
		''' But maybe: 
			Encode:
			1. create a vector = np.range(len(img))
			2. new_freq_vals = [v for v in freq_vals if v != 0]
			3. probs = new_freq_vals / np.sum(new_freq_vals)
			Decode:
			img = list()
			for i in range(len(decoded_img)):
				if freq_vals[i] != 0:
					img.append(decoded_img[i])
				else:
					img.append(0)
		'''
		# normalise probs
		probs /= np.sum(probs)

		rans_encoder = Encoder()
		lengths, enc_times, dec_times = list(), list(), list()
		for i, image in enumerate(q_data):
			img_shape = image.shape
			T1 = time()
			flt_img = image.flatten()
			for px_val in flt_img:
				rans_encoder.encode_symbol(probs, px_val)

			encoded_img = rans_encoder.get_encoded()
			lengths.append(len(encoded_img))
			encoding_T = time() - T1

			rans_decoder = Decoder(encoded_img)
			decoded_img = []
			T2 = time()
			for _ in range(len(encoded_img)):
				decoded_img.append(rans_decoder.decode_symbol(probs))
			decoded_img.reverse()
			decoding_T = time() - T2
			enc_times.append(encoding_T)
			dec_times.append(decoding_T)
			q_data[i] = np.array(decoded_img).reshape(img_shape)

		commas = lambda x : ','.join(map(str, x))
		text = ['Bit lengths,{commas(lengths)}\n']
		text += 'Encoding times,{commas(enc_times)}\n'
		text += 'Decoding times,{commas(dec_times)}\n'
		with open(filename, 'w') as log_f:
			log_f.writelines(text)
		print(f"log written to: {filename}")
		data = (q_data / bits_ratio + data_min) * data_std
		return data, bit_lengths
	# inverse
	data = (q_data / bits_ratio + data_min) * data_std
	print(f"simulated quantisation with bits: {quantise_lvls+1} on axes: {axis}")
	return data


if __name__ == '__main__':

	conf_name = 'config/ffhq128_c.yml'
	parser = ArgumentParser(description='Compression visualisations of NF-synth.')
	parser.add_argument('--config', '-c', default=conf_name, dest='config')
	ap = parser.parse_args()

	C = ConfWrap(fn=ap.config)
	# C.archive_data = True
	# C.archive_step = True
	# C.use_data_archive = False # True
	# C.use_step_archive = False
	main(C)

