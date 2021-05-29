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


def main(C, epoch=4340000, save=True):

	p_value_counts = dict()
	model_meta_stuff = select_model(C.training.root_dir, select_epoch='auto', figures=7)

	analyse_synthesizer(C, model_meta_stuff)


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
		potential_archive_fn = glob(f'{model_root_fp}/*{data_fn}')
		if len(potential_archive_fn)==1:
			archive_filename = potential_archive_fn[0]
			print(f"matched archive: {archive_filename}")

		if os.path.isfile(archive_filename):
			print(f'Loading saved arrays: `{s_str}`', end='')
			print(f' and {data_fn}`')
			dataset = np.load(archive_filename)
	else:
		# Z's computation is expensive and should only
		# be computed once. Therefore, we do it now
		# and save it to disk.
		if C.compress.dataset == 'celeba':
			loader = load_celeba(128, C.training.img_size, test=True)
		elif C.compress.dataset == 'ffhq':
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
		attributes = Attributes().fetch()
	elif C.compress.dataset == 'ffhq':
		attributes = Attributes(dataset='ffhq')

	## make cache directory for all analyses (PCA red. + reupsampling).
	root_d = f'{C.training.root_dir}/compress_{C.compress.dataset}'
	os.makedirs(root_d, exist_ok=True)
	# C.cache_bn = make_basename(root_d, cache=True)

	# if not C.compress.no_model:
	# 	C.compress.pca.n_pcs = C.compress.pca.n_pcs_sc
	# 	reducer = Synthesizer(C, steps = C.compress.steps[0],
	#                       net=net, device=device)
	# C.basename = make_basename(root_d, subdir='pca', pca_pcs=C.pca.n_pcs_sc)
	# pca_reduction_on_attributes(C, reducer, data_z, attributes)

	# re-upsampling/reconstruction visualization.
	reup_umap_npc = [i for i in range(*eval(C.compress.umap_n_comps))]
	reup_pca_npc = [int((i**3.5)) for i in range(*eval(C.compress.pca_n_pcs))]
	reup_pca_npc += [data_z.shape[0]-10]
	steps_l = C.compress.steps
	for stps in steps_l: 
		reup_umap_l = (['empty'] if 'umap' not in stps else reup_umap_npc)
		for umap_pcs in reup_umap_l:
			# if not C.compress.no_model:
			for pca_pcs in reup_pca_npc:
				for ds in C.compress.dataset_ratios:
					for kept_out in [True, False]:
						if (kept_out and pca_pcs >= int(ds * data_z.shape[0])-16) \
						   or (not kept_out and pca_pcs >= int(ds * data_z.shape[0])):
							continue
						# in function use
						C.steps = stps # !
						C.ds_ratio = ds
						C.kept_out = kept_out
						# for Synth use:
						C.compress.umap_n_comps = umap_pcs
						C.compress.pca_n_pcs = pca_pcs
						# make folders
						C.basename = make_basename(root_d, subdir=ds, pca_pcs=pca_pcs,
						                           umap_pcs=umap_pcs)
						C.cache_bn = make_basename(root_d, subdir=ds, pca_pcs=pca_pcs,
						                           umap_pcs=umap_pcs, quantise_bits=C.compress.quantise, cache=True)
						print(f'Analysing synth with \
						        PCA nps:{pca_pcs}, UMAP dims: {umap_pcs}, \
						        ko:{C.kept_out}, ratio:{ds}')
						if not C.compress.no_model:
							reducer = Synthesizer(C.compress, steps=stps, net=net,
						                           device=device)
							compute_reduction_reupsampling(C, attributes, data_z,
							                        data_x, reducer, quantise_bits=C.compress.quantise)
						else:
							compute_reduction_reupsampling(C, attributes, data_z,
							                               data_x, quantise_bits=C.compress.quantise)


def make_basename(root, mark_time=False, subdir=None, pca_pcs=None, umap_pcs=None, cache=False,
		quantise_bits=False):
	time = ''
	basename = root 
	# two subdirectories
	if subdir:
		# prefix (used for ratio in reupsam, otherwise analysis type ('pca'))
		if isinstance(subdir, (int, float)):
			# basename += f'/rat{subdir}' # XXX !!! XXX
			if quantise_bits:
				basename += f'/q{quantise_bits}-rat{subdir}' # XXX !!! XXX
			else:
				basename += f'/noq-rat{subdir}' # XXX !!! XXX
		else:
			basename += f'/{subdir}'
	if cache:
		basename += '/cache'
	os.makedirs(basename, exist_ok=True)
	# filename prefix
	if mark_time: 
		import datetime as dt
		t = str(dt.datetime.now()).split(sep='.')[0].replace('2021-', '').replace(' ', '_')
		basename += t
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
	if attributes.ds == 'celeba':
		# Cat's: blond, brown air, smiling, wearing hat.
		att_ind = [5, 11, 31, 35] + 1 # XXX !!! XXX
		att_str = '-'.join(map(str, att_ind)) # [str(a) for a in att_ind])
	elif attributes.ds == 'ffhq':
		att_ind = 0
		att_str = 'test-split'
	ko = '_ko' if C.kept_out else ''
	ds_trunk_idx = int(data.shape[0] * C.ds_ratio)
	ds_rat = f'_{ds_trunk_idx}' if C.ds_ratio != 1 else ''
	filename = f'{C.basename}_{att_str}{ko}{ds_rat}.png'
	cache_fn = f'{C.cache_bn}_{att_str}{ko}{ds_rat}.npz'

	# middle two keys change depending on whether UMAP is used.
	step_vec_keys = ['X', 'Z', 'PC/eigenZ', 'PC/UMAP', 'rec_Z', 'rec_X']

	if C.compress.dataset == 'celeba':
		kept_out_df, kept_out_idcs = attributes.pick_last_n_per_attribute(att_ind, n=4)
		att_names = list(kept_out_df.columns)
	elif C.compress.dataset == 'ffhq':
		# kept_out_idcs = range(16)
		kept_out_idcs = rng.choice(5000, 16, False)
		att_names = [f'row {n}' for n in range(1, 5)]

	if C.compress.use_step_archive and os.path.isfile(cache_fn):
		step_vector = list()
		with np.load(cache_fn) as data:
			for k in step_vec_keys:
				step_vector.append(data[k])
	elif not C.compress.no_model:
		# att_ind = list(range(0, 40, 10))
		# split dataset
		z_s = data[kept_out_idcs].copy()
		x_s = data_x[kept_out_idcs].copy()
		# TODO: add individual image transform for
		# stand-along compression
		# if mono: 
		# 	for d in data:
		# 		red_d = reducer.fit_transform(d)
		# else:
		if C.kept_out:
			data = np.delete(data, kept_out_idcs, axis=0)
		rng.shuffle(data, axis=0)
		if ds_trunk_idx < data.shape[0]:
			data = data[:ds_trunk_idx]
		
		reducer.fit(data)
		# TODO: replace show_steps arguments with argument selection
		red_data = reducer.transform(z_s, show_steps='all')

		# the last element of red(uced)_data is the lower level representation.
		if quantise_bits:
			assert isinstance(quantise_bits, int)
			quantise_lvls = 2**quantise_bits - 1
			q_data = red_data[-1].copy()
			q_data_min = q_data.min(axis=0, keepdims=True)
			q_data -= q_data_min
			q_data[q_data == 0.0] = 0.0000000000001
			bits_ratio = quantise_lvls/q_data.max(axis=0, keepdims=True)
			q_data *= bits_ratio
			q_data = np.rint(q_data)
			# inverse
			q_data = q_data / bits_ratio + q_data_min
			print(f"simulated quantisation with bits: {quantise_lvls+1}")
			red_data[-1] = q_data

		rec_data = reducer.inverse_transform(red_data[-1], show_steps='all',
		                                  resample=C.compress.resample_z)
		del data_x; del data
		if 'umap' in C.steps:
			step_vector = [x_s, z_s] + red_data + rec_data[1:]
		else:
			step_vector = [x_s, z_s] + [reducer.models['pca'].components_] \
			               + red_data + rec_data
		if C.compress.archive_step:
			np.savez(cache_fn, **dict(zip(step_vec_keys, step_vector)))
	else:
		raise RuntimeError

	plot_compression_flow(step_vector, filename, att_names, C.steps)
	print('done.')
	

def plot_reduced_dataset(pca, z_s, att, k, att_ind, filename):
	import warnings
	raise warnings.DeprecationWarning
	# sort from highest variance
	from sklearn.decomposition import PCA
	if isinstance(pca, PCA):
		components = pca.components_[:k][::-1]
		var_exp = pca.explained_variance_[:k][::-1]
		ratio_var_exp = pca.explained_variance_ratio_[:k][::-1]
		'''z_s, y = label_zs(z_s)'''
		from celeb_simil import subset_attributes, category_from_onehot
		
		sel_att_df = subset_attributes(att, att_ind, overall_indexer=True, complementary=True)
		red_z = pca.transform(z_s[ sel_att_df.iloc[:, -1]].reshape(sel_att_df.shape[0], -1))
		reduced_z = red_z[:, :k][:,::-1]   # PCs['X'].T becomes (306,10000)
	else: # should be type: sklearn.decomposition.PCA
		raise NotImplementedError
	
	from matplotlib import pyplot as plt

	symbols = "." # can be used for orthogonal attributes.
	n_pcs = k # can use this to index components and create grid.
	fs = int(n_pcs * 2)
	fig, axs = plt.subplots(n_pcs, n_pcs, figsize=(fs, fs), sharex='col', sharey='row')
	cmap = plt.cm.winter # get_cmap('Set1')

	# Use subset dataframe turn 1 hot vectors into indices,
	# then add column for "both" categories if overlapping.
	color_series, overlapping_attributes = category_from_onehot(sel_att_df)
	# color_series += 2 # make it red
	
	for row in range(n_pcs):
		# random permutation of reduced datapoints for 
		# visualization that is evened among categories. 
		# indices = np.random.permutation(reduced_z.shape[0])
		# reduced_z = np.take(reduced_z, indices, axis=0)
		# y = np.take(y, indices)
		for col in range(n_pcs):
			if row > col:
				path_c = axs[row, col].scatter(reduced_z[:,col], reduced_z[:,row], c=np.array(color_series), cmap=cmap, s=.50, alpha=0.6)
				axs[row, col].annotate('% VE:\nC{}={:.2f}\nC{}={:.2f}'.format(n_pcs - row, ratio_var_exp[row]*100,
										 n_pcs-col, ratio_var_exp[col]*100), xy=(0.7, 0.7), xycoords='axes fraction', fontsize='xx-small')
				if row == n_pcs-1:
					axs[row, col].set_xlabel(f'component {n_pcs-col}') 
					axs[row, col].tick_params(axis='x', reset=True, labelsize='x-small')
				if col == 0:
					axs[row, col].set_ylabel(f'component {n_pcs-row}')
					axs[row, col].tick_params(axis='y', reset=True, labelsize='x-small')
			else:
				axs[row, col].remove()
				axs[row, col] = None

	handles, labels = path_c.legend_elements(prop='colors')
	if overlapping_attributes:
		assert isinstance(att_ind, (list, tuple))
		labels = att.columns[np.array(att_ind)] + ['both']
	else:
		assert isinstance(att_ind, int)
		labels= [att.columns[att_ind]] + ['Complement cat.']
	plt.legend(handles, labels, bbox_to_anchor=(.75, .75), loc="upper right", 
	           bbox_transform=fig.transFigure)
	
	fig.tight_layout()
	fig.subplots_adjust(top=.88)
	plt.savefig(filename, bbox_inches='tight')
	plt.close()
	print(f'Saved to {filename}.')


if __name__ == '__main__':

	conf_name = 'config/ffhq128_c.yml'
	parser = ArgumentParser(description='RealNVP training on various datasets.')
	parser.add_argument('--config', '-c', default=conf_name)
	parser.parse_args()

	C = ConfWrap(fn=parser.config)
	# C.archive_data = True
	# C.archive_step = True
	# C.use_data_archive = False # True
	# C.use_step_archive = False
	main(C)

