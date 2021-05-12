#!/bin/env /var/scratch/mao540/miniconda3/envs/revive/bin/python

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
from reduce import make_basename
from hdbscan import HDBSCAN
from matplotlib import pyplot as plt

def main(C, epoch=160000, save=True):

	p_value_counts = dict()
	model_meta_stuff = select_model(C.training.root_dir, select_epoch='auto')

	analyse_synthesizer(C, model_meta_stuff)


def analyse_synthesizer(C, model_meta_stuff = None):

	device = torch.device("cuda:0" if torch.cuda.is_available() and len(C.net.gpus) > 0 else "cpu")
	
	model_root_fp, model_fp, vmarker_fp = model_meta_stuff
	if not C.no_model:
		net, _ = load_network( model_fp, device, C.net)
	epoch = model_root_fp.split('_')[-1]
	archive_filename= f"{model_root_fp}/data_{C.dataset}.npz"
	import ipdb; ipdb.set_trace()
	if os.path.isfile(archive_filename) and C.use_data_archive:
		# stats = torch.load(stats_filename)
		dataset = np.load(archive_filename)
		print(f'`{C.data}` stats loaded.')
	else:
		dataset = track_z(net, device, C.training.img_size, batch_size=256, stats=C.data)
		if C.archive_data:
			np.savez(archive_filename, **dataset)

	if 'z' in C.data:
		data_z = dataset['Z'].reshape(dataset['Z'].shape[0], -1)
	if 'x' in C.data:
		data_x = dataset['X'].reshape(dataset['X'].shape[0], -1)
	attributes = Attributes().fetch()

	## Extract parameters in need of eval.
	root_d = C.training.root_dir
	# C.cache_bn= make_basename(root_d, subdir='cluster', pca_pcs=C.pca.n_pcs_sc,
	#                            cache=True)

	# re-upsampling/reconstruction visualization.
	C.plot_dir = model_root_fp
	clus_umap_npc = [3, 3]
	clus_pca_npc = [50, 500]
	for stps in [C.steps]:
		for umap_pcs in clus_umap_npc:
			C.umap.n_comps= umap_pcs
			for pca_pcs in clus_pca_npc:
				C.pca.n_pcs = pca_pcs
				for kept_out in [True, False]:
					C.kept_out = kept_out
					C.basename = make_basename(root_d, pca_pcs=pca_pcs,
					                           umap_pcs=umap_pcs)
					C.cache_bn = make_basename(root_d, pca_pcs=pca_pcs,
					                           umap_pcs=umap_pcs, cache=True)

					print(f'Analysing synth with\
					        PCA nps:{C.pca.n_pcs}, UMAP dims: {C.umap.n_comps}, ko:{C.kept_out}') #, ratio:{C.dataset_ratio}')
					if not C.no_model:
						reducer = Synthesizer(C, steps=stps, net=net,
						                        device=device)
						cluster_lowdim(C, attributes, data_z, reducer)
					else:
						cluster_lowdim(C, attributes, data_z)



def cluster_lowdim(C, attributes, data=None, reducer=None):
	''' Save x's and z's, visualized pre (original) 
	and post (reconstructed) dimensionality reduction.'''
	# blond vs brown air, smiling vs. wearing hat.
	att_ind = [5, 11, 31, 35]
	ko = '_ko' if C.kept_out else ''
	att_str = '-'.join([str(a) for a in att_ind])
	# ds_trunk_idx = int(data.shape[0] * C.dataset_ratio)
	# ds_rat = f'_{ds_trunk_idx}' if C.dataset_ratio != 1 else ''
	# filename = f'{C.basename}_{att_str}{ko}{}.png'
	cache_fn = f'{C.cache_bn}_{att_str}.npz'

	# select attributes
	if not C.no_model:
		# att_ind = list(range(0, 40, 10))
		kept_out_df, kept_out_idcs = attributes.pick_last_n_per_attribute(att_ind, n=4)
		# split dataset
		z_s = data[kept_out_idcs].copy()
		# x_s = data_x[kept_out_idcs].copy()
		att_names = list(kept_out_df.columns)
		
		if C.kept_out:
			data = np.delete(data, kept_out_idcs, axis=0)
		rng = np.random.default_rng()
		rng.shuffle(data, axis=0)
		# if ds_trunk_idx < data.shape[0]: # useful for dataset_ratois
		# 	data = data[:ds_trunk_idx]

		lowd_embedding = reducer.fit_transform(data)
		# PCA
		print("fitting pca hdbscan")
		pca_hdb = HDBSCAN(min_samples=20, min_cluster_size=200)
		pca_preds = pca_hdb.fit_predict(lowd_embedding[-2])
		# To viz preds on pca, use umap vector for now
		# UMAP
		print("fitting umap hdbscan")
		umap_hdb = HDBSCAN(min_samples=20, min_cluster_size=200)
		umap_preds = umap_hdb.fit_predict(lowd_embedding[-1])

		if C.save_vector_cache:
			step_vector = {'pca_pred': pca_preds,
			               'umap_pred': umap_preds,
			               'pca': lowd_embedding[0],
			               'umap': lowd_embedding[-1]}
			np.savez(cache_fn, **step_vector)
	else:
		with np.load(cache_fn) as data:
			pca_preds = data['pca_pred']
			umap_preds = data['umap_pred']
			lowd_embedding = data['pca']
			lowd_embedding.append(data['umap'])

	filename = f'{C.basename}_{att_str}_pcaPred.png'
	plot_3d_pred(lowd_embedding[-1], pca_preds, attributes,
			fn=filename)

	filename = f'{C.basename}_{att_str}_umapPred.png'
	plot_3d_pred(lowd_embedding[-1], umap_preds, attributes,
	             fn=filename)
	print('done.')

def plot_3d_pred(data, predictions, attrs, fn, pca=False):
	x = data[:, 0]
	y = data[:, 1]
	z = data[:, 2]
	# from matplotlib.colors import ListedColormap
	cmap = plt.cm.get_cmap('jet')
	# attrs_sel = range(40)
	for a_i in range(40):
		name = attrs.df.iloc[:, a_i].name
		# plot
		fig = plt.figure(figsize=(10, 10))
		ax = fig.add_subplot(111, projection='3d')
		# legend
		ax_pth_coll = ax.scatter(x, y, z, c=predictions, cmap=cmap, s=1, alpha=0.9)
		# handles, labels = ax_pth_coll.legend_elements(prop='colors')
		# color_attribute = f'no-{col_arr.name}', col_arr.name
		# ax.legend(handles, color_attribute)
		# save
		fig.tight_layout()
		fig.subplots_adjust(top=.88)
		plt.savefig(fn)
		plt.close()


if __name__ == '__main__':
	C = ConfWrap(fn='config/cluster_c.yml')
	# C.use_data_archive = True
	# C.archive_data = False
	C.dataset = 'celeba'
	main(C)
