#!/var/scratch/mao540/miniconda3/envs/revive/bin/python

import torch 
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.nn.functional import normalize

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
# from mnist_simil import cleanup_version_f
# from utils import load_network
from warnings import warn
from load_data import Attributes, select_model, load_network, \
                      load_ffhq, load_celeba
from shell_util import maketree
from  config.config import ConfWrap



def main(C, epochs=[160000], save=True):

	p_value_counts = dict()
	# for e in epochs:
	model_meta_stuff = select_model(C.training.root_dir, select_epoch='auto', figures=7)
	# fp_model_root, fp_model, fp_vmarker = model_meta_stuff
	# mark_version(C.version, fp_vmarker) # echo '\nV-' >> fp_vmarker
	analyse_epoch(C, model_meta_stuff)
	# mark_version(C.version, fp_vmarker, finish=True) # echo '0.2' >> fp_vmarker


def analyse_epoch(C, model_meta_stuff = None):
	device = torch.device("cuda:0" if torch.cuda.is_available() and len(C.net.gpus) > 0 else "cpu")
	print("evaluating on: %s" % device)
	# select model.
	root_dir = C.training.root_dir
	if not model_meta_stuff:
		fp_model_root, fp_model, fp_vmarker = select_model(root_dir, select_epoch=i)
	else:
		fp_model_root, fp_model, fp_vmarker = model_meta_stuff
	# fp_model_root = '/home/mao540/glow-pytorch/data/glow_ffhq128/backup_epoch'
	# fp_model = fp_model_root + '/model.pth.tar'
	# fp_vmarker = None
	# root_dir = fp_model_root

	epoch = fp_model_root.split('_')[-1]
	# stats
	_stats = C.similarity.data

	s_str = '-'.join(_stats).lower()
	archive_filename = f'{root_dir}/{s_str}_{C.training.dataset}.npz'

	if os.path.isfile(archive_filename) and C.similarity.use_data_archive:
		print('Found cached file, skipping computations of mean and std.')
		stats = dict()
		with np.load(archive_filename) as data:
			for s in _stats:
				# .lower() because previous code and missing time.
				stats[s.lower()] = data[s]
	else:
		if C.similarity.dataset == 'mnist':
			loader = load_mnist_test(C)
		if C.similarity.dataset == 'celeba':
			loader = load_celeba(C.training.batch_size, C.training.img_size, test=True)
		elif C.similarity.dataset == 'ffhq':
			loader = load_ffhq(C.training.batch_size, C.training.img_size, test=True)

		net, _ = load_network( fp_model, device, C.net)
		net.eval()
		stats = track_z(net, device, C.training.img_size, loader, stats=_stats, batch_size=128)
		np.savez(archive_filename, **stats)
		stats = {k.lower(): v for (k, v) in stats.items()} # back.compatib.

	''' Attributes annotation '''
	if C.similarity.dataset == 'celeba':
		if C.track_y:
			warn("`track_y` Should not be used unless you know why.")
			# no gt for FFHQ
			if 'net' not in dir():
				net, _ = load_network( fp_model, device, C.net)
				net.eval()
			y = tellme_ys(net, testloader, device)
			attr_y = Attributes().create_df(y).serialize_df()
		else:
			attr_y = Attributes().fetch()
	else:
		attr_y = Attributes(dataset='ffhq')
	''' test without these 3 lines '''
	# if isinstance(attr_y, Attributes):
	# 	# compatibility previous `Attributes` class version
	# 	attr_y = attr_y.df

	''' filepaths '''
	anls_root_dir = root_dir + f'/similarity-r-{C.similarity.dataset}' # Analysis root dir_
	fp_distr = anls_root_dir + '/distr' # distributions
	fp_simil = anls_root_dir + '/similarity' # similarity
	fp_replace = anls_root_dir + '/replace_from_grandz'
	fp_pca = anls_root_dir + '/pca'
	fp_umap = anls_root_dir + '/umap'
	paths = [fp_distr, fp_simil, fp_replace, fp_pca]
	# lstify = lambda s: [s] if isinstance(s, str)
	maketree(paths)
	# [os.makedirs(ppp, exist_ok=True) for ppp in [fp_distr, fp_simil, fp_replace, fp_pca]]
	
	
	''' distributions analyses '''
	print("analysing z distribution... ")
	scatter_all(stats, fp_distr + '/meanstds.png')
	if attr_y.ds == 'celeba':
		# this is a bit of a hack to diff. between CelebA and FFHQ
		scatter_attr(stats, attr_y, fp_distr + '/faces_subplots.png', C.similarity.dataset)
	# # # violin_eachdigit(stats, fp_distr + '/dig_violins.png', epoch)
	# ''' distance analysis '''
	
	# Add repetition for noint=False variant
	if attr_y.ds == 'celeba':
		distances, inst_count_mean = calculate_distance(stats, attr_y, measure='mean')
		heatmap(distances, attr_y, fp_simil + '/distances_mean_100_int_count.png',
					plot_title='Average distance between attributes in Z space (means)')
		# noheatmap(inst_count_mean, filename=fp_simil+'/att_count_mean.png')
		distances, inst_count_std = calculate_distance(stats, attr_y, measure='std')
		heatmap(distances, attr_y, fp_simil + '/distances_std_100_int_count.png',
					plot_title='Average distance between digits in Z space (std)')
		# noheatmap(inst_count_std, filename=fp_simil+'/att_count_std.png')
		# distances = calculate_distance(stats, attr_y, joint=True)
		heatmap(distances, attr_y, fp_simil + '/distances.png')

	# distances = y_distance_z(stats)
	# measure = 'mean'
	# for m in range(distances.shape[0]):
	# 	heatmap(distances[m], fp_simil + f'/pixelwise_dist_{measure}.png',
	# 							 plot_title=f'pixelwise {measure} similarity')
	# 	measure = 'std'
	if attr_y.ds == 'celeba':
		all_zs = grand_z(stats, attr_y)
	elif attr_y.ds == 'ffhq':
		all_zs = grand_z(stats, attr_y, reps=16)

	# overall Z for each digit (averaged across batches).
	plot_grand_z(all_zs, attr_y.headers, anls_root_dir + '/grand_zs_normimg.png')
	plot_grand_z(all_zs, attr_y.headers, anls_root_dir + '/grand_zs_normarray.png', norm='array')
	plot_grand_z(all_zs, attr_y.headers, anls_root_dir + '/grand_zs_normarrayb.png', norm='arrayb')
	# plot_grand_z_rgb(all_zs, attr_y.columns, anls_root_dir + '/grand_zs_rgb.png')

	if 'net' not in dir():
		net, _ = load_network( fp_model, device, C.net)
		net.eval()

	
	maketree(fp_replace)

	with torch.no_grad():
		for t in [.75, .9, 1.]:
			for i in range(1, 11):
				dims = np.prod(all_zs.shape[1:])
				k = int(dims / 2 ** i)
				sample_from_crafted_z(net, all_zs, attr_y, absolute=True, kept=k, device=device, reps=1,
									 save_dir=fp_replace, temp=t) #monster_mode=True)

			sample_from_crafted_z(net, all_zs, attr_y, absolute=True, kept=dims, device=device, reps=1,
									 save_dir=fp_replace, temp=t) #monster_mode=True)

	# 		# sample_from_crafted_z(net, all_zs, absolute=True, kept=12288, device=device, reps=1,
	# 		# 						 save_dir=fp_replace, temp=t, monster_mode=True)


	''' dimensionality reduction '''
	if 'net' not in dir():
		net, _ = load_network( fp_model, device, C.net)
		net.eval()

	dataset = stats['z'].reshape(stats['z'].shape[0], -1)

	# # # # pick_components = 350
	# fp_pca += '/std'; 
	fp_pca_red = fp_pca + '/red'
	maketree([fp_pca, fp_pca_red])
	for pick_components in [50, 100, 1000, 2000, 4000, 10000, stats['z'].shape[0]-1]: #'mle']:
		print('computing PCA...{}...'.format(pick_components), end='')
		pca = PCA(n_components=pick_components, whiten=True).fit(dataset) # dataset.data
		if pick_components == 'mle':
			print(f' N PCs found: {pca.n_components_}')
		analyse_principal_components(pca,stats,attr_y,fp_pca,16, net, device)
		del pca

	print('umap analysis')
	if 'net' not in dir():
		net, _ = load_network( fp_model, device, C.net)
		net.eval()
	umap_inverse_wrapper(stats, attr_y, fp_umap, net, device)
	fp_umap += '_3d'; maketree(fp_umap)
	# plot_sphere_umap(fp_umap, stats, attr_y, n_neighbors=10)
	plot_sphere_umap(fp_umap, stats, attr_y, n_neighbors=10, metric='euclidean', n_components=3)
	print('done')

	# # UMAP -- `test_umap` will use directories `<anls_root_dir>/umap/{,3d}`
	# fn_prefix = anls_root_dir + '/umap'
	# os.makedirs(fn_prefix+'/3d', exist_ok=True)
	# for nn in [7, 20, 400]:
	# 	for md in [0, 0.1, 0.5]:
	# 		test_umap(stats, fn_prefix, n_neighbors=nn, min_dist=md, n_components=3)


def issue_z_from_pc(PC, stats, filename):
	raise NotImplementedError
	z_s = grand_z(stats)
	z_s = z_s.reshape(z_s.shape[0], -1)
	components = PC['components']
	for i in range(10):
		
		print('h'+i)
		# every N
		for c in components:
			# every PC

			pass
	pass


def PCA_eig_np(Z, k, center=True, scale=False):
	'''
	https://medium.com/@ravikalia/pca-done-from-scratch-with-python-2b5eb2790bfc
	'''
	n, p = Z.shape
	ones = np.ones([n, 1])
	# subtract from each column its mean, to ensure mean = 0.
	h = ((1/n) * np.matmul(ones, ones.T)) if center else np.zeros([n, n])
	H = np.eye(n) - h
	Z_center = np.matmul(H, Z)
	covariance = 1/(n-1) * np.matmul(Z_center.T, Z_center)
	# divide each column by its std. Only if outcome is independent of variance.
	scaling = np.sqrt(1/np.diag(covariance)) if scale else np.ones(p)
	scaled_covariance = np.matmul(np.diag(scaling), covariance)
	w, v = np.linalg.eig(scaled_covariance)
	components = v[:, :k]
	explained_variance = w[:k]
	return {'z': Z, 'k': k, 'components': components.T,
				'exp_var': explained_variance}


def analyse_principal_components(pca, stats, att, fp_pca,pk, net=None,
		device=None):
	'''
	Arguments: 
		- pca: sklearn.decomposition. PCA object. after calling .fit()
		 stats: Z stats
		- fp_pca: root filepath for plot saving.
		- pk: componets to show in plots PCgrid.
	'''

	print("plotting reconstructed Z... ", end='')
	if att.ds == 'celeba':
		n_cols = 2
		n_images = 20
		sel_attr = [[i + n_cols*j for i in range(n_cols)] for j in range(n_images)] # == 40
	elif att.ds == 'ffhq':
		sel_attr = [0]

	for att_selection in sel_attr:
		if isinstance(att_selection, list):
			fn = fp_pca + '/rZ_att-{}_ncomps-{}.png'.format(
			                                ''.join([str(i) for i in att_selection]),
			                                pca.n_components_)
		else:
			fn = f'{fp_pca}/rZ_att-{att_selection}_ncomps-{pca.n_components_}.png'
		plot_rZ(pca, stats, att, filename=fn, n_examples=6,
		       net=net, device=device, selected_attributes=att_selection)
	print("done.")

	fn = fp_pca + '/PCgrid_first-{}.png'.format(pk)
	if not os.path.isfile(fn):
		print("plotting components in grid format... ", end='')
		plot_PCgrid(pca, fn, pk=16) # pk for FFHQ
		print("done.")

	print("plotting variance explained... ", end='')
	fn = fp_pca + '/VE_ncomps-{}.png'
	plot_expvar(pca.n_components_, pca.explained_variance_ratio_, fn)
	print("done.")

	os.makedirs(fp_pca, exist_ok=True)
	for i in range(len(att.headers)):
		print(f"plotting {att.headers[i]} reduced z's... ", end='')
		fn = fp_pca + '/red/rZatt_{}-{}.png'.format(i, att.headers[i])
		plot_reduced_dataset(pca, stats['z'], att, k=15, att_ind=i, filename=fn)
	print("done.")


def plot_rZ(pca, stats, att, filename, n_examples, net, device, selected_attributes=5,
		norm='std'):

	from matplotlib.offsetbox import OffsetImage, AnnotationBbox
	'''z_s, y'''
	if not isinstance(selected_attributes, int):
		nrows, ncols = n_examples, len(selected_attributes) # 3 instances, 10 n of sel. categories.
	elif isinstance(selected_attributes, int):
		nrows, ncols = n_examples, 2
		selected_attributes = [selected_attributes] * 2
	elif selected_attributes == 'random':
		nrows, ncols = n_examples, selected_attributes
		selected_attributes = np.random.choice([i for i in range(40)], size=ncols, replace=False)
	z_s = stats['z']
	x_s = stats['x']

	h_size = n_examples
	C_H_W = z_s.shape[1:]
	if C_H_W[-1]<=64:
		figsize = (ncols*6,nrows*2)
	else:
		figsize = (ncols*7,nrows*2)
	fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

	pc_i = 1
	net.eval()
	for col in range(ncols):
		with torch.no_grad():
			# ori_Z = z_s[dig_idx].astype(np.float32)
			# Select nrows of celebrities
			n_att = selected_attributes[col]
			ori_Z = z_s[att.df.iloc[:, n_att].astype(bool)].astype(np.float32)
			ori_X = x_s[att.df.iloc[:, n_att].astype(bool)]
			faces_idx = np.random.randint(ori_Z.shape[0], size=nrows)
			ori_Z = ori_Z[faces_idx].copy()
			ori_X = ori_X[faces_idx].copy()
			
			### original Zs
			# keep original Zs for plotting; oriZ for generation
			# variable without underscores `_` are used for the GPU (pytorch).
			ori_Z = ori_Z.reshape(nrows,*C_H_W)
			# standardise ori_Z if celeba  (old load_celeba)
			# if att.ds == 'celeba':
			# 	# oZ_min = ori_Z.min(axis=(1, 2, 3),keepdims=True)
			# 	# oZ_max = ori_Z.max(axis=(1,2,3),keepdims=True)
			# 	oZ_std = ori_Z.std(axis=(1,2,3),keepdims=True)
			# 	ori_Z = ori_Z / oZ_std

			# oriZ = torch.from_numpy(ori_Z).to(device)
			# oriZ = net(oriZ, partition=True)
			## oriX = net(oriZ, reverse=True, resample=True) # True)
			## oriX = oriX.cpu().detach().numpy()
			# Transform with var. explained by PCs
			red_Z = pca.transform(ori_Z.reshape(nrows, -1))
			rec_Z = pca.inverse_transform(red_Z)
			### reconstruced Zs
			rec_Z = rec_Z.reshape(nrows, *C_H_W)
			# keep rec_Z for plotting; recZ for generation.
			recZ = torch.from_numpy(rec_Z.astype(np.float32)).to(device)
			# # Standardise after reconstruction (Deprec.)
			# if norm=='std':
			# 	recZ = (recZ - recZ.mean(dim=(1, 2, 3), keepdim=True)) \
			# 	                 / recZ.std(dim=(1, 2, 3), keepdim=True)
			recZ_p = net(recZ, partition=True)
			recX = net(recZ_p, reverse=True, resample=True) #True)
			rec_X = recX.cpu().detach().numpy()
			rec_Z = recZ.cpu().detach().numpy()
			### normalize over array cel_rZ
			# rZ_min = np.min(rec_Z, dim=(1, 2, 3), keepdim=True)
			# rZ_min = rec_Z.min(axis=(1, 2, 3), keepdims=True)
			# rZ_max = rec_Z.max(axis=(1, 2, 3), keepdims=True)
			# if norm=='img':
			# 	cel_rZ = (rec_Z - rZ_min) / (rZ_max - rZ_min)
			# elif norm == 'std':
			# 	rZ_std = np.std(rec_Z, axis=(1, 2, 3), keepdims=True)
			# 	cel_rZ = (rec_Z - rZ_min) / rZ_std
			# 	cel_rZ = cel_rZ / cel_rZ.max(axis=(1,2,3), keepdims=True)
			axs[0, col].set_title(att.headers[n_att], fontsize='small')

		rX_min = rec_X.min(axis=(1, 2, 3), keepdims=True)
		rX_max = rec_X.max(axis=(1,2,3),keepdims=True)
		rec_X = (rec_X - rX_min) / (rX_max - rX_min)
		rec_X = (rec_X.copy() * 255).astype(np.uint8)

		rZ_min = rec_Z.min(axis=(1,2,3),keepdims=True)
		rZ_max = rec_Z.max(axis=(1,2,3),keepdims=True)
		rec_Z = (rec_Z - rZ_min) / (rZ_max - rZ_min)
		# cel_rZ = (rec_Z.copy() * 255).astype(np.uint8)
		fac_rZ = (rec_Z.copy() * 255).astype(np.uint8)

		# rec_Z = (cel_rZ.copy() * 255).astype(np.uint8)
		for row in range(nrows):
			
			axs[row, col].imshow(np.moveaxis(fac_rZ[row], 0, -1))
			axs[row, col].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False,
					right=False, left=False, labelleft=False)

			# oriX_imbox.image.axes = axs[row, col]
			annotext = "$O\;\mu: {:.2f}, \sigma: {:.2f} || R\;\mu: {:.2f}, \sigma:{:.2f}$".format(
					ori_Z[row].mean(), ori_Z[row].std(), fac_rZ[row].mean(), fac_rZ[row].std())
			axs[row, col].set_xlabel(annotext, fontsize='xx-small')

			# Show original and reconstructed X
			# before = (data[col + 4*row].copy() * 255).astype(np.uint8)
			before = np.moveaxis(ori_X[row].reshape(*C_H_W), 0, -1)
			before = (before - before.min()) / (before.max() - before.min())
			if C_H_W[-1]<=64:
				zoom = 1.2
			else:
				zoom = .8
			oriX_imbox = OffsetImage(before, zoom=zoom)
			oX_ab = AnnotationBbox(oriX_imbox, xy=(-0.6, 0.5), 
					          xycoords='data', boxcoords="axes fraction")
			axs[row, col].add_artist(oX_ab)

			after = np.moveaxis(rec_X[row].reshape(*C_H_W), 0, -1)
			recX_imbox = OffsetImage(after, zoom=zoom)
			# x_imagebox.image.axes = axs[row, col]
			rX_ab = AnnotationBbox(recX_imbox, xy=(1.6, 0.5), 
					          xycoords='data', boxcoords="axes fraction")
			axs[row, col].add_artist(rX_ab)


	plt.tight_layout()
	plt.savefig(filename)
	plt.close()


def plot_PCgrid(PCA, filename, pk=None, reconstruct=False):
	if not pk: # Plot Komponents
		pk = min(PCA.n_components_, 25)
	PCs = PCA.components_
	n_pcs = PCA.n_components_
	var_exp = PCA.explained_variance_
	# y = PCs['y']
	nrows = ncols = int(pk**.5)
	# nrows +=1
	fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*2, nrows*2))
	H = W = int((PCs.size / (3 * n_pcs)) ** .5)
	PCs = PCs.reshape(n_pcs, 3, H, W)
	# PCs = (comp - comp.min()) / (comp.max() - comp.min())
	# cmap = plt.cm.RdBu
	mean_red_z = PCA.mean_.reshape(3 , H, W)
	mean_red_z = (mean_red_z - mean_red_z.min()) / (mean_red_z.max() - mean_red_z.min())
	axs[0, 0].imshow(np.moveaxis(mean_red_z, 0, -1))
	axs[0, 0].set_title('$\mu$')
	pc_i = 0
	for row in range(nrows):
		for col in range(ncols):
			if col == 0 and row == 0: continue # mean is displayed here.

			if pc_i < pk: # for the case where pca.n_PCs < 25
				comp_img = (np.moveaxis(PCs[pc_i], 0, -1) - PCs[pc_i].min()) / (PCs[pc_i].max() - PCs[pc_i].min())
				comp_img = (comp_img * 255).astype(np.uint8)
				axs[row, col].imshow(comp_img)
				axs[row, col].set_title('PC{}'.format(pc_i+1))
			# else:  # if pc_i >= pk:
			# 	# axs[row, col].remove()
			# 	# axs[row, col] = None
			pc_i += 1


	plt.tight_layout()
	fig.subplots_adjust(top=.88)
	plt.savefig(filename) # TOFIX
	plt.close()


def PCA_test(z_s, k, center=False, mode='self'):
	
	dataset, y = label_zs(z_s)
	n_datapoints = dataset.shape[0]
	p = dataset.shape[1] # feature number

	if mode == 'sklearn':
		PCs = PCA(n_components=k).fit(dataset) # dataset.data
		return {'z': dataset, 'k': k, 'components': PCs.components_,
					'exp_var': PCs.explained_variance_, 'y': y}
	elif mode == 'self':
		PCs = PCA_eig_np(dataset, k, center)
		PCs['y'] = y
		return PCs

def subset_attributes(attr_df, att_ind, overall_indexer=False, complementary=True, max_attributes=10):
	''' Arguments:
		attr_df: pandas dataframe with labeled attributes for celebA
		att_ind: int or [list of ints], for n attributes subset to be returned
		Returns:
			dataframe object with all attributes + complementary
		Example headers of a possible output:
			subset:              (optional columns:)
			| att_1 | att_5 |    ( comp_att_1 | comp_att_5 | anyistrue )   |

		Note: all attributes are 1-hot encoded (1 = True, 0 = False)
			'''

	if not isinstance(att_ind, (int, list, tuple)):
		raise TypeError

	# if isinstance(att_ind, int):
	# 	subsel_df = pd.concat([attr_df.iloc[:, att_ind], (attr_df.iloc[:, att_ind] == 0)], axis=1)
	# 	overall_indexer = (subsel_df == 1).any(axis=1)
	# 	return pd.concat([subsel_df, overall_indexer], axis=1)
	# if length > 1: # assumed sequence

	subsel_df = pd.DataFrame(attr_df.iloc[:, att_ind]) # was numpy.array(5)
	attribute_names = list(subsel_df.columns)
	if complementary:
		complement_categs_names = ['comp_' + c for c in attribute_names]
		# concatenate selected attributes df with its complementary.
		subsel_df = pd.concat([subsel_df, (attr_df.iloc[:, att_ind] == 0)], axis=1)
		subsel_df.columns = attribute_names + complement_categs_names
	if overall_indexer:
		index_all = (subsel_df == 1).any(axis=1)
		subsel_df = pd.concat([subsel_df, index_all], axis=1)
	return subsel_df



def plot_reduced_dataset(pca, z_s, att, k, att_ind, filename):
	# sort from highest variance
	if isinstance(pca, PCA):
		components = pca.components_[:k][::-1]
		var_exp = pca.explained_variance_[:k][::-1]
		ratio_var_exp = pca.explained_variance_ratio_[:k][::-1]
		# sel_att_df = subset_attributes(att, att_ind, overall_indexer=True, complementary=True)
		# comp = (len(att.headers) == 40)
		print(type(att_ind))
		sel_att_df = att.subset(att_ind) # , complementary=False)

		# import ipdb; ipdb.set_trace()
		red_z = pca.transform(z_s.reshape(sel_att_df.shape[0], -1))
		reduced_z = red_z[:, :k][:,::-1]   # PCs['X'].T becomes (306,10000)
	else: # should be type: sklearn.decomposition.PCA
		raise NotImplementedError

	symbols = "." # can be used for orthogonal attributes.
	n_pcs = k # can use this to index components and create grid.
	fs = int(n_pcs * 2)
	fig, axs = plt.subplots(n_pcs, n_pcs, figsize=(fs, fs), sharex='col', sharey='row')
	cmap = plt.cm.winter # get_cmap('Set1')

	# Use subset dataframe turn 1 hot vectors into indices,
	# then add column for "both" categories if overlapping.
	# import ipdb; ipdb.set_trace()
	if att.ds == 'celeba':
		# color_series, overlapping_attributes = category_from_onehot(sel_att_df)
		color_series = sel_att_df
	else:
		color_series = np.ones((5000,))
	overlapping_attributes = 0


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
		labels= [att.headers[att_ind]] + ['Complement cat.']
	plt.legend(handles, labels, bbox_to_anchor=(.75, .75), loc="upper right", 
	           bbox_transform=fig.transFigure)
	
	fig.tight_layout()
	fig.subplots_adjust(top=.88)
	plt.savefig(filename, bbox_inches='tight')
	plt.close()
	print(f'Saved to {filename}.')



def plot_PCA(PCs, filename):
	# sort from highest variance
	if isinstance(PCs, dict):
		components = PCs['components'][::-1]
		var_exp = PCs['exp_var'][::-1]
		z_s = PCs['z']
		y = PCs['y']
		reduced_z = components.dot(z_s.T) # might want to check this.
	else: # should be type: sklearn.decomposition.PCA
		components = PCs.components_[::-1]
		var_exp = PCs.explained_variance_[::-1]
		reduced_Z = PCs.transform()

	n_pcs = len(var_exp) # can use this to index components and create grid.
	fig, axs = plt.subplots(n_pcs, n_pcs, figsize=(12, 12), sharex='col', sharey='row')
	
	for row in range(n_pcs):
		for col in range(n_pcs):
			if row > col:
				axs[row, col].scatter(components[row], components[col], s=.50)# label=f'{col}x{row}')
				axs[row, col].annotate('var.exp.:\nC{}={:.3f}\nC{}={:.3f}'.format(n_pcs - row, var_exp[row],
										 n_pcs-col, var_exp[col]), xy=(.30, .30), fontsize='xx-small')
				if row == n_pcs-1:
					axs[row, col].set_xlabel(f'component {n_pcs-col}') 
					axs[row, col].tick_params(axis='x', reset=True, labelsize='x-small', which='both')
				if col == 0:
					axs[row, col].set_ylabel(f'component {n_pcs-row}')
					axs[row, col].tick_params(axis='y', reset=True, labelsize='x-small', which='both')
			else:
				axs[row, col].remove()
				axs[row, col] = None
	
	fig.tight_layout()
	fig.subplots_adjust(top=.88)
	plt.savefig(filename, bbox_inches='tight')
	plt.close()

#	horrible i know
#	nrows = ncols = len(var_exp) # can use this to index components and create grid.
#	fig, axs = plt.subplots(nrows, ncols, figsize=(12, 12), sharex='col', sharey='row')
#	
#	for row in range(nrows):
#		for col in range(ncols):
#			if row > col:
#				axs[row, col].scatter(components[row], components[col], s=.50)# label=f'{col}x{row}')
#				axs[row, col].annotate('var.exp.:\nC{}={:.3f}\nC{}={:.3f}'.format(5 - row, var_exp[row],
#					                     5-col, var_exp[col]), xy=(.30, .30), fontsize='xx-small')
#				if row == nrows-1:
#					axs[row, col].set_xlabel(f'component {5-col}') 
#				if col == 0:
#					axs[row, col].set_ylabel(f'component {5-row}')
#				if row == nrows-1 or col == 0:
#					axs[row, col].tick_params(reset=True, labelsize='x-small')
#			else:
#				axs[row, col].remove()
#				axs[row, col] = None
	

def plot_expvar(n_pcs, r_var_exp, filename):
	
	fig, ax = plt.subplots(figsize=(8, 6))

	# n_pcs = len(var_exp)
	cs = np.cumsum(r_var_exp)
	ax.plot(cs)

	ax.set_ylabel("cumulative ratio of explained variance")

	ax.axhline(cs[-1], color="k", alpha=0.5)

	trans = mpl.transforms.blended_transform_factory(
		      ax.get_yticklabels()[0].get_transform(), ax.transData)
	ax.text(0, cs[-1], "{:.4f}".format(cs[-1]), color="blue", transform=trans,
		      ha="right", va="center")

	ax.set_xlim(left=0, right=n_pcs)

	xtick_vals, xtick_labels = list(plt.xticks())
	if float(n_pcs) not in xtick_vals:
		try:
			ax.set_xticks(xtick_vals + [n_pcs])
			ax.set_xticklabels( xtick_labels + [n_pcs])
		except ValueError as err:
			print(err)
			import ipdb; ipdb.set_trace()
	# plt.title("variance explained first PC")
	plt.savefig(filename.format(n_pcs))
	plt.close()


def make_interpolated_grid(angles=None):
	'''
	make grid.
	angles should be: 
		- (xmin, ymin, xmax, ymax) of type int.
	'''
	
	# xmin, ymin, xmax, ymax = angles
	# angles = [[xmin, ymax], [xmax, ymax], [xmin, ymin], [xmax, ymin]]

	corners = np.array(angles)
	test_pts = np.array([
		(corners[0]*(1-x) + corners[1]*x)*(1-y) +
		(corners[2]*(1-x) + corners[3]*x)*y
		for y in np.linspace(0, 1, 10)
		for x in np.linspace(0,1,10)
	])
	return test_pts


def rotate_around_point(point, radians, origin=(0, 0)):
	"""Rotate a point around a origin.
	[taken from https://gist.github.com/LyleScott/e36e08bfb23b1f87af68c9051f985302]
	I call this the "low performance" version since it's recalculating
	the same values more than once [cos(radians), sin(radians), x-ox, y-oy).
	It's more readable than the next function, though.
	"""
	x, y = point
	ox, oy = origin

	qx = ox + np.cos(radians) * (x - ox) + np.sin(radians) * (y - oy)
	qy = oy + -np.sin(radians) * (x - ox) + np.cos(radians) * (y - oy)
	return qx, qy


def translation_matrix(xy, to_origin=True):
	transl_mtx= np.eye(3)
	if to_origin:
		transl_mtx[:-1, -1] = -1 * xy # grid back inplace.
	else:
		transl_mtx[:-1, -1] = xy # grid back inplace.
	return transl_mtx

def scale_matrix(xy_scale_factors):
	x_sf, y_sf = xy_scale_factors
	return np.array([[x_sf,0,0],[0,y_sf,0], [0,0,1]]) # only scale x, rest is rotation.

def rotation_matrix(angle):
	theta = angle
	return np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0],[0,0,1]])

def make_grid(boundaries, degrees=0, scale_factor = 0.5, edge_scale_factors=(1, 1)):
	xmin, ymin, xmax, ymax = boundaries
	# get center of grid.
	radians = degrees * np.pi / 180 # TODO define degrees
	grid_center = np.array((xmin + (xmax-xmin) / 2, ymin + (ymax-ymin) / 2))
	# make array for affine transf. out of upper/lower/left/right boundaries.
	vertices = np.array([[xmin, ymax], [xmax, ymax], [xmin, ymin], [xmax, ymin]])
	vertices = np.concatenate([vertices, np.ones(vertices.shape[0]).reshape(vertices.shape[0],1)], axis=1)
	# translation matrix: to origin and back to grid centre (or point cloud center).
	transl_mtx_to_origin = translation_matrix(grid_center)
	transl_mtx_to_grid_center = translation_matrix(grid_center, to_origin=False)
	# define scaling matrix (for upper edge first, whole matrix then)
	edge_scale_mtx = scale_matrix(edge_scale_factors)
	scale_mtx = scale_matrix((scale_factor, scale_factor))
	# define rotation matrix
	rot_mtx = rotation_matrix(radians)

	# Apply transformations.
	alt_mtx = transl_mtx_to_origin.dot(vertices.T)
	# only change upper edge, the rest is integral to the whole frame
	alt_mtx[:,:2] = edge_scale_mtx.dot(alt_mtx[:, :2])
	alt_mtx = scale_mtx.dot(alt_mtx)
	alt_mtx = rot_mtx.dot(alt_mtx)
	alt_mtx = transl_mtx_to_grid_center.dot(alt_mtx).T
	# produce smooth linear grid among vertices.
	test_pts = make_interpolated_grid(alt_mtx)
	return  test_pts


def category_from_onehot(onehot_df, idcs=(0, 1), vals=(1, 2), over_val=None):
	if idcs == (0, 1) and onehot_df.shape[1] > 3:
		print("Warning: dataframe is misshaped for idcs (0, 1). Continue at own risk.")
	i, j = idcs
	val_i, val_j = vals
	color_series = onehot_df.iloc[:, i].copy()
	color_series[color_series == 1] = val_i
	color_series[onehot_df.iloc[:, j]] = val_j

	if (onehot_df.iloc[:, np.array(idcs)].sum(axis=1) > 1).any():
		overlapping_onehot_ = True
		warn(' Overlapping categories found: filling with value: {over_val}')
		color_series[onehot_df.iloc[:,np.array(idcs)].sum(axis=1)] = over_val
	else: overlapping_onehot_ = False
	return color_series, overlapping_onehot_



def umap_inverse_wrapper(stats, att, fp_umap, net=None, device=None, n_neighbors_l=None):
	metric='euclidean'
	dataset = stats['z'].reshape(stats['z'].shape[0], -1)

	''' attributes '''
	# att_ind = (20, 31)
	'''here''' # and call it in plot_inverse_umap
	# import ipdb; ipdb.set_trace()
	celeba = (len(att.headers) == 40) 
	
	if not n_neighbors_l:
		n_neighbors_l = [5, 20, 100] # , 400]
		min_dist_l = [0.2] * len(n_neighbors_l)
	else:
		n_neighbors_l = [n_neighbors_l]
	for nn, md in zip(n_neighbors_l, min_dist_l):
		for temp in np.linspace(0.6, 0.8, 5):
			fp_umap_ = f'{fp_umap}/resamp_{temp:.1f}'
			maketree(fp_umap_)
			# was for nn, md in zip(...):
			print(f'UMAP: nearest_n = {nn}, min_dist = {md:.2f}')
			deg = 45
			knn = 100
			knn_w = 'distance'
			# grid_s = 0.35
			umap = UMAP(n_neighbors=nn, min_dist= md,
									n_components=2, random_state=42)
			umap = umap.fit(dataset)
			inverse_sampling = True
			for a_i in range(len(att.headers)):
				# attributes = subset_attributes(att, att_ind=a_i)
				attributes = att.subset(a_i, complementary=celeba)
				if att.ds == 'celeba' and \
					len(attributes.shape) > 1:
					col_arr = attributes.iloc[:, 0]
					sym_arr = attributes.iloc[:, 1]
					names = attributes.iloc[:,0].name + '-' + attributes.iloc[:,1].name
				else:
					# col_arr = np.ones((5000,))
					# sym_arr = np.ones((5000,))
					col_arr = attributes.iloc[:]
					sym_arr = attributes.iloc[:]
					names = attributes.name  # 'test split'

				print(f'computing inverse umap sampling for: {names}', end='')
				basename = '/{}_{}_nn{:d}_md{:.2f}_knn{:d}'.format(a_i, names, nn,md,knn)
				fn = fp_umap_ + basename + '.jpg'
				plot_inverse_umap(fn, att=att, n_neighbors=nn, min_dist=md, knn=knn, deg_rot=deg,
				                  col_arr=col_arr, knn_weights=knn_w,
				                  inverse_sampling=inverse_sampling,mapper=umap,net=net,
				                  device=device, temp=temp, metric=metric)
			del umap

					# inverse_sampling = False


def plot_knn_boundaries(x1, x2, y, nn=5, weights='distance', h=.02, ax=None,
	                      n_colors=2, cmap=None):

	from sklearn import neighbors
	from matplotlib.colors import ListedColormap
	# Lighter tab10 version:
	if not cmap:
		tab20 = mpl.cm.get_cmap('tab20')
		newcolors = tab20(np.linspace(0, 1, 20))[1 : 1+n_colors*2 : 2]
		cmap = ListedColormap(newcolors)

	clf = neighbors.KNeighborsClassifier(n_neighbors=nn, weights=weights)
	dataset = np.c_[x1, x2]
	clf.fit(dataset, y.astype(int))

	x_min, x_max = x1.min() - 1, x1.max() + 1
	y_min, y_max = x2.min() - 1, x2.max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
	                     np.arange(y_min, y_max, h))
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

	Z = Z.reshape(xx.shape)
	if not ax:
		plt.pcolormesh(xx, yy, Z, cmap='tab10', alpha=.2)
	else:
		ax.pcolormesh(xx, yy, Z, cmap=cmap)
		return ax, clf


def plot_sphere_umap(path, stats=None, att=None,
		                  n_neighbors=None, min_dist=0.02, n_components=2, inverse_sampling=False,
		                  deg_rot=45, mapper=None, col_arr=None, sym_arr=None, knn=100,
		                  knn_weights=None, net=None, device=None, grid_scale=0.7, temp=1,
								metric=None, **kwarg):
	if isinstance(att, Attributes):
		att = att.df
	if metric == None:
		metric = 'haversine'
		n_components=2
	dataset = stats['z'].reshape(stats['z'].shape[0], -1)
	print('computing UMAP projection: ', end='')
	print(f'n_neighbors = {n_neighbors}; min_dist = {min_dist}...', end='')
	umap = UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist,
	            metric=metric, output_metric=metric, random_state=42)
	sphere_mapper = umap.fit(dataset)
	if metric == 'haversine':
		x = np.sin(sphere_mapper.embedding_[:, 0]) * np.cos(sphere_mapper.embedding_[:, 1])
		y = np.sin(sphere_mapper.embedding_[:, 0]) * np.sin(sphere_mapper.embedding_[:, 1])
		z = np.cos(sphere_mapper.embedding_[:, 0])
	elif metric == 'euclidean' and n_components == 3:
		x = sphere_mapper.embedding_[:, 0]
		y = sphere_mapper.embedding_[:, 1]
		z = sphere_mapper.embedding_[:, 2]

	from matplotlib.colors import ListedColormap
	cmap = plt.cm.get_cmap('cool')
	# set1_cmap = mpl.cm.get_cmap('Set1')
	# newcolors = set1_cmap(np.linspace(0, 1, 8))[0 : 2]
	# subset1_cmap = ListedColormap(newcolors)
	for a_i in range(0, 39):
		col_arr = att.iloc[:, a_i].copy()
		# att2_arr = att.iloc[:, a_j]
		name = col_arr.name # + '-' + att2_arr.name
		# col_arr[ (att1_arr == 0) & (att2_arr == 0)] = 0 # F / no S
		# col_arr[ (att1_arr == 1) & (att2_arr == 0)] = 1 # M / no S
		# col_arr[ (att1_arr == 0) & (att2_arr == 1)] = 2 # F / S
		# col_arr[ (att1_arr == 1) & (att2_arr == 1)] = 3 # M / S 
		fig = plt.figure(figsize=(10, 10))
		ax = fig.add_subplot(111, projection='3d')
		# legend
		ax_pth_coll = ax.scatter(x, y, z, c=col_arr, cmap=cmap, s=1, alpha=0.9)
		handles, labels = ax_pth_coll.legend_elements(prop='colors')
		color_attribute = f'no-{col_arr.name}', col_arr.name
		ax.legend(handles, color_attribute)
		# save
		fig.tight_layout()
		fig.subplots_adjust(top=.88)
		plt.savefig(f'{path}/{name}_sphere_umap.png')
		plt.close()



def plot_inverse_umap(filename, stats=None, att=None,
		                  n_neighbors=None, min_dist=0.02, n_components=2, inverse_sampling=False,
		                  deg_rot=45, mapper=None, col_arr=None, sym_arr=None, knn=100,
		                  knn_weights=None, net=None, device=None, grid_scale=0.7, temp=1,
								metric=None, **kwC):
	if metric == None:
		metric = 'euclidian'
	if not mapper:
		raise NotImplementedError
		if not stats:
			raise ValueError
		dataset = stats['z'].reshape(stats['z'].shape[0], -1)
		print('computing UMAP projection: ', end='')
		print(f'n_neighbors = {n_neighbors}; min_dist = {min_dist}...', end='')
		umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
		            n_components=n_components,
		            metric=metric, random_state=42)
		mapper = umap.fit(dataset)

	grid_boundaries = [mapper.embedding_[:,0].min(), mapper.embedding_[:,1].min(),
	          mapper.embedding_[:,0].max(), mapper.embedding_[:,1].max()]
	test_pts = make_grid(grid_boundaries, degrees=0, scale_factor=grid_scale)

	# setup plotting grid
	from matplotlib.gridspec import GridSpec
	if inverse_sampling:
		fig = plt.figure(figsize=(22, 11))
		gs = GridSpec(10, 20, fig)
		scatter_ax = fig.add_subplot(gs[:, :10])
		faces_axes = np.zeros((10, 10), dtype=object)
		for i in range(10):
			for j in range(10):
				faces_axes[i, j] = fig.add_subplot(gs[i, 10+j])
	else:
		fig = plt.figure(figsize=(10, 10))
		scatter_ax = fig.add_subplot()

	# colormaps for knn boundaries and UMAP scatter.
	from matplotlib.colors import ListedColormap

	if sym_arr:
		n_colors = col_arr.nunique() + sym_arr.nunique()
	else:
		n_colors = col_arr.nunique()


	pastel_cmap = mpl.cm.get_cmap('Pastel1')
	newcolors_knn = pastel_cmap(np.linspace(0, 1, 8))[0 : n_colors]
	pastel_subcmap_knn = ListedColormap(newcolors_knn)

	set1_cmap = mpl.cm.get_cmap('Set1')
	newcolors_a = set1_cmap(np.linspace(0, 1, 8))[0 : n_colors//2]
	set1_subcmap_a = ListedColormap(newcolors_a)
	newcolors_b = set1_cmap(np.linspace(0, 1, 8))[n_colors//2: n_colors]
	set1_subcmap_b = ListedColormap(newcolors_b)

	col_arr_knn = col_arr.copy() # 0:  F / no S  || 1: M / no S

	if sym_arr is not None:
		col_arr_knn[ (col_arr == 0) & (sym_arr == 0)] = 0 # F / no S
		col_arr_knn[ (col_arr == 1) & (sym_arr == 0)] = 1 # M / no S
		col_arr_knn[ (col_arr == 0) & (sym_arr == 1)] = 2 # F / S
		col_arr_knn[ (col_arr == 1) & (sym_arr == 1)] = 3 # M / S
	# KNNeighbors
	if metric == 'euclidian':
		scatter_ax, knn_clf = plot_knn_boundaries(mapper.embedding_[:,0], mapper.embedding_[:,1],
	                                 weights=knn_weights, y=col_arr_knn, nn=knn,
	                                 ax=scatter_ax, cmap=pastel_subcmap_knn)
	# scatter embeddings. 
	# We plot 2 categories and their respective complementaries.
	# Category_1 is colorcoded (in col_arr)
	# Category_2 is represented by markers (in sym_arr)
	# here we separate the dataset by markers (symbols).
	if sym_arr is not None:
		embedding_cat_2b = mapper.embedding_[sym_arr.astype(bool), :] # Smiling
		embedding_cat_2a = mapper.embedding_[(sym_arr== 0), :]       #  Not smiling
		# col_arr = np.array(col_arr)
		col_arr_cat_2a = col_arr[(sym_arr==0)]          # no Smiling
		col_arr_cat_2b = col_arr[sym_arr.astype(bool)] # Smiling
	
		# Smiling (F/M)# green / violet
		sctt_ax_cat_2b = scatter_ax.scatter(embedding_cat_2b[:,0], embedding_cat_2b[:,1],
				         c=col_arr_cat_2b.astype(int), marker = '+', cmap=set1_subcmap_b, s=7)
	else:
		embedding_cat_2a = mapper.embedding_
		col_arr_cat_2a = col_arr
	# No Smiling (F/M)# red / blue
	sctt_ax_cat_2a = scatter_ax.scatter(embedding_cat_2a[:,0], embedding_cat_2a[:,1],
				         c=col_arr_cat_2a.astype(int), marker = 'x', cmap=set1_subcmap_a, s=7)


	# scatter_ax.set(xticks=[], yticks=[])
	if inverse_sampling:
		scatter_ax.scatter(test_pts[:,0], test_pts[:,1], marker='x', c='k', s=15, alpha=1)
		print()
		print(f'starting inverse transform for {test_pts.shape[0]} datapoints...', end='')

		inv_transformed_points = mapper.inverse_transform(test_pts[:,:-1])
		print('done!')
		# ### original Zs
		# # keep original Zs for plotting; oriZ for generation
		grid_test_pts = mapper.transform(inv_transformed_points * temp)
		scatter_ax.scatter(grid_test_pts[:,1], grid_test_pts[:,0], marker='+', c='w', s=15, alpha=1)
		torch.cuda.empty_cache()
		# ESTIMATE image vector DIMENSIONS
		img_s = int((inv_transformed_points.shape[1] // 3) ** .5)
		tZ = torch.from_numpy(inv_transformed_points.reshape(100, 3,img_s, img_s).astype(np.float32)).to(device)
		# tZ = torch.randn((100, 3, 64, 64), dtype=torch.float32, device=device) #changed 3 -> 1

		tX = np.array([], dtype=np.float32).reshape(0, 3, img_s, img_s)
		for t_Z_ in tZ.view(4, 25, 3, img_s, img_s):
			with torch.no_grad():
				t_x_ = net(net(t_Z_ * temp, partition=True), reverse=True, resample=True)
				tX = np.concatenate([tX, t_x_.to('cpu').detach().numpy()])

		# plot generated digits:
		for i in range(10):
			for j in range(10):
				x_face = tX[i*10+j].reshape(3, img_s, img_s)
				x_face = (x_face - x_face.min()) / (x_face.max() - x_face.min())
				faces_axes[i, j].imshow(np.moveaxis(x_face, 0, -1))
				faces_axes[i, j].set(xticks=[], yticks=[])
	
	handles_2a, labels_2a = sctt_ax_cat_2a.legend_elements(prop='colors')
	color_attributes = 'no-' + col_arr.name, col_arr.name
	if sym_arr is not None:
		handles_2b, labels_2b = sctt_ax_cat_2b.legend_elements(prop='colors')
		handles = handles_2a + handles_2b
		marker_attributes =  'no-' + sym_arr.name, sym_arr.name
		labels = ['/'.join([c, m]) for c in marker_attributes for m in color_attributes]
	else:
		handles = handles_2a
		labels = [c for c in color_attributes]


	scatter_ax.legend(handles, labels, loc='best')

	scatter_ax.legend(handles, labels, loc='best')
	# plt.title('n_neighbors = {:d}; min_dist = {:.2f}'.format(n_neighbors, min_dist))

	plt.tight_layout()
	plt.savefig(filename)
	plt.close()
	print(' Saved {}'.format(filename.split('/')[-1]))


def test_umap(stats, fn_prefix, n_neighbors=15, min_dist=0.1, n_components=2,
				  metric='euclidean', **kwC):

	# dataset, col_arr = label_zs(stats['z'])
	dataset = stats['z'].reshape(stats['z'].shape[0], -1)

	print('computing UMAP projection: ', end='')
	print(f'n_neighbors = {n_neighbors}; min_dist = {min_dist}...', end='')

	reductor = UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
									n_components=n_components, 
									metric=metric, random_state=42)
	embeddings = reductor.fit_transform(dataset)

	if n_components == 2:
		fig, ax = plt.subplots(figsize=(10, 10))
		scatter = ax.scatter(embeddings[:,0], embeddings[:,1], c=col_arr, cmap='Spectral', s=4)
	elif n_components == 3:
		fn_prefix += '/3d'
		from mpl_toolkits.mplot3d import Axes3D
		fig = plt.figure(figsize=(11, 11))
		ax = fig.add_subplot(111, projection='3d')
		scatter = ax.scatter(embeddings[:,0], embeddings[:,1], embeddings[:,2], alpha=0.3,
												c=col_arr, cmap='Spectral', s=4)
	
	handles, labels = scatter.legend_elements(prop='colors')
	ax.legend(handles, labels, loc='best', title='off-line digits')
	plt.title(f'n_neighbors = {n_neighbors}; min_dist = {min_dist}')

	filename = fn_prefix + '/nn{:d}.jpg'.format(n_neighbors, min_dist, n_components)
	plt.savefig(filename)
	plt.close()
	print(' Saved {}'.format(filename.split('/')[-1]))



def test_arrays():
	'''deprecated utility function'''
	arr_1 = np.arange(24).reshape(2,3,4)
	arr1 = arr_1[0,::2].copy()
	arr2 = arr_1[1,1:].copy()
	arr_1[0,::2] = arr2[:,::-1]
	arr_1[1,1:] = arr1[:,::-1]
	arr_3= np.zeros(shape=(arr_1.shape[0], 1, 3, 4))
	arr_2 = np.zeros(shape=(arr_1.shape[0], 3, 4))
	arr_2[:] = 99
	return (arr_1, arr_2, arr_3)

def replace_highest_along_axis(arr_ref, arr_2, arr_3, k):
	'''
	Replaces values from arr_2 to arr_3, according to highest k values
	in arr_ref.
	Input:
		- arr_ref: reference array to look up highest k values.
		- arr_2: source array, to extract values from.
		- arr_3: destination array, to inject values to at given indices.
	Output:
		- arr_3 with vals from arr_2 at indices presenting arr_1 k-highest values. 
	'''
	assert (arr_ref.shape[0] % arr_2.shape[0]) == 0, "Arrays mismatch"
	# arr_ref, arr_2, arr_3 = test_arrays()
	shape_arr_3 = arr_3.shape
	shape_arr_2 = arr_2.shape
	arr_mask = np.zeros_like(arr_2)
	reps = arr_3.shape[0] // arr_2.shape[0] # repetitions.
	# Reference array, with flattened batches.
	arr_ref_fb = arr_ref.reshape(arr_ref.shape[0], -1)
	# select k highest values. 
	maxk_b_ind = np.argpartition(arr_ref_fb, -k)[:, -k:] # without first :?
	maxk = np.take_along_axis(arr_2.reshape(arr_2.shape[0], -1), maxk_b_ind, -1)
	# put maxk in arr_mask for plotting
	np.put_along_axis(arr_mask.reshape(arr_2.shape[0], -1), maxk_b_ind, maxk, -1)
	if reps > 1:
		maxk = maxk.repeat(reps, axis=0)
		maxk_b_ind = maxk_b_ind.repeat(reps, axis=0)

	# arr_3 = arr_3.reshape(arr_ref_fb.shape, -1)
	np.put_along_axis(arr_3.reshape(arr_3.shape[0], -1), maxk_b_ind, maxk, -1) # channel dimension
	return arr_mask.reshape(shape_arr_2), maxk, arr_3.reshape(shape_arr_3)


def replace_highest(arr_1, arr_2, arr_3, k=1):
	'''
	Deprecated.
	Replaces values from arr_2 to arr_3, according to highest k values
	in arr_1.
	Input:
		- arr_1: reference array to look up highest k values.
		- arr_2: source array, to extract values from.
		- arr_3: destination array, to inject values at given indices.
	Output:
		- arr_3 with vals from arr_2 at indices presenting arr_1 k-highest values. 
	'''
	# arr_1, arr_2, arr_3 = test_arrays()
	assert arr_1.size == arr_3.size, "Arrays mismatch"
	# 1. reshape with shape=(batch_size, H*W)
	arr_b = arr_1.reshape(arr_1.shape[0], -1)
	# 2. find indices for k highest values for each item along 1st dimension. 
	maxk_b_ind= np.argpartition(arr_b, -k)[:, -k:]

	# 3. flatten and unravel indices
	maxk_ind_flat = maxk_b_ind.flatten() #<- LET OP: unravelling flattened inds.
	maxk_ind_shape = np.unravel_index(maxk_ind_flat, arr_1.shape)
	# unravel: form indices coordinates system references to: (arr_1.shape)
	batch_indices = np.repeat(np.arange(arr_1.shape[0]), k) # (batch_size, k).
	maxk_indices = tuple([batch_indices] + [ind for ind in maxk_ind_shape])

	maxk = arr_2.reshape(arr_3.shape)[maxk_indices] # 3. resume this. 
	arr_3[maxk_indices] = maxk
	return maxk_indices, maxk, arr_3

def compute_delta(grand_zs, absolute=True):
	diff = np.zeros_like(grand_zs)

	for i in range(grand_zs.shape[0]):
		# i
		if absolute:
			diff[i] = np.abs(grand_zs[i] - np.mean(np.delete(grand_zs, i, axis=0), axis=0))
		else:
			diff[i] = grand_zs[i] - np.mean(np.delete(grand_zs, i, axis=0), axis=0)
	return diff

def craft_z(grand_zs, absolute, kept=None, reps=10, fold=False, device="cuda:0"):
	''' Create z's from average, but with gaussian noise.
	Inputs:
		- nd_array: number-digits array with grand average of all z's spaces.
		- kept: integer or float. If integer, equals number of pixels to be kept
								  if float, and fold=True, equals proportion of 
								  pixels to be kept.
		- fold: dictates whether kept is read as absolute pixels count, or proportion.
	Outputs:
		- Artificial Z's for each digit. 
	'''
	# mean = np.mean(grand_zs, axis=0) # should be ~= 0.
	# abs_diff = np.abs(grand_zs - mean)
	diff = compute_delta(grand_zs, absolute=absolute)

	batch_size = grand_zs.shape[0] * reps
	batch = torch.randn((batch_size, *grand_zs.shape[1:]), dtype=torch.float32, device='cpu').numpy() # TODO: CHANGE 'CPU'
	arr_mask, _, batch = replace_highest_along_axis(diff, grand_zs, batch.copy(), kept)
	del diff
	return arr_mask, batch


def sample_from_crafted_z(net, all_zs,att, absolute, kept, reps, device, save_dir, temp=.75, monster_mode=False):
	''' 
	Input:
		all_zs: n-dimensional but also faces grand Z array.
	Output: plot.
	'''
	mask_zs, z = craft_z(all_zs, absolute=absolute, kept=kept, reps=reps)
	if monster_mode:
		(B, C, H, W) = z.shape
		hw = int(H * (B ** 0.5))
		z = z.reshape(1, C, hw, hw)
		# tile mask_zs horizontally (obtain 28 x 280 batches)
		mask_zs = np.tile(mask_zs, reps=(1, 1, 10))
		mask_zs.reshape(3, H*10, W*10)
		kept = f'{kept}x{reps}'

	z = torch.from_numpy(z*temp).to(device)
	z_p = net(z, partition=True)
	x = net(z_p, reverse=True)
	
	x = x.to('cpu').detach().numpy()

	plot_grand_z(x, att.headers, save_dir + f'/k{kept}_t{temp}_sample.png')
	plot_grand_z(mask_zs, att.headers, save_dir + f'/k{kept}_t{temp}_mask.png')
	del x, z, mask_zs

def plot_grand_z(grand_zs, names, filename, n_rows_cols=(6, 7), norm='img'):
	# mpl.rc('text', usetex=True)
	# mpl.rcParams['text.latex.preamble']=[r"\boldmath"]
	if len(names) >= 40: # CelebA
		n_rows, n_cols = n_rows_cols
		figsize = (16, 10)
	elif len(names) == 1 and grand_zs.shape[0]>1:
		n_cols = n_rows = int(grand_zs.shape[0] ** .5)
		figsize = tuple([int(i * 3) for i in [n_rows, n_cols]])
		names = names * grand_zs.shape[0]
	elif len(names) == 1 and grand_zs.shape[0] == 1:
		n_cols = n_rows = 1
		grand_zs.shape[0]
		figsize = (8, 8)

	fig, axs = plt.subplots(n_rows, n_cols, sharex='all', sharey='all', figsize=figsize)

	n = 0
	for col in range(n_cols):
		for row in range(n_rows):
			if n == grand_zs.shape[0]:
				axs[row, col].remove()
				axs[row, col] = None
				continue

			img_z = np.moveaxis(grand_zs[n].copy(), 0, -1)
			if norm == 'img':
				img_z = (img_z - img_z.min()) / (img_z.max() - img_z.min())
			elif norm == 'array':
				img_z = (img_z - img_z.min()) / (grand_zs.max() - grand_zs.min())
			elif norm == 'arrayb':
				img_z = (img_z - grand_zs.min()) / (grand_zs.max() - grand_zs.min())
			img_z = (img_z.copy() * 255).astype(np.uint8)
			axs[row, col].imshow(img_z)
			ttl = r"Grand-${{z}}$ for {}".format(names[n])
			# axs[row, col].title.set_text(ttl, fontsize='xx-small')
			axs[row, col].set_title(ttl, fontsize='xx-small')
			n += 1
	
	fig.suptitle(r"Grand ${{z}}$ for each celebA attribute.")
	fig.tight_layout()
	fig.subplots_adjust(top=.88)
	plt.savefig(filename, bbox_inches='tight')
	print('\nPlot saved to ' + filename)
	plt.close()
	del grand_zs


def plot_grand_z_rgb(grand_zs, names, filename, n_rows_cols=(10, 12)):
	# mpl.rc('text', usetex=True)
	# mpl.rcParams['text.latex.preamble']=[r"\boldmath"]

	n_rows, n_cols = n_rows_cols
	fig, axs = plt.subplots(n_rows, n_cols, sharex='all', sharey='all', figsize=(20, 10))
	# n_cols /= 3 # keep track of channels in for loop.

	# grand_zs = (grand_zs - grand_zs.min()) / (grand_zs.max() - grand_zs.min())

	n = 0
	for row in range(n_rows):
		for col in range(0, n_cols, 3):
			# import ipdb; ipdb.set_trace()
			z_img = grand_zs[n].copy() 
			for n_ch in range(3): 
				z_base_img = np.zeros(shape=(64, 64, 3), dtype=np.float64)
				ch_img = z_img[n_ch].copy() 
				ch_img = (ch_img - ch_img.min()) / (ch_img.max() - ch_img.min())
				z_base_img[:,:,n_ch] = ch_img
				axs[row, col + n_ch].imshow(z_base_img) # np.moveaxis(z_img, 0, -1) )
				if n_ch == 1:
					ttl = r"Grand-${{z}}$ for {}".format(names[n])
					# axs[row, col].title.set_text(ttl, fontsize='xx-small')
					axs[row, col + n_ch].set_title(ttl, fontsize='xx-small')
			n += 1
	
	fig.suptitle(r"Grand ${{z}}$ for each celebA attribute.")
	fig.tight_layout()
	fig.subplots_adjust(top=.88)
	plt.savefig(filename, bbox_inches='tight')
	plt.close()
	print('\nPlot saved to ' + filename)
	del grand_zs

def y_distance_z(stats, n_categories=10):
	''' take stats file, returns a similarity matrix
		for distances computed item-wise (pixel-wise).'''
	n_att = len(att.columns)
	distances = np.zeros(shape=(n_att, n_att))
	# if joint and measure:
	# 	raise ValueError("set either joint or measure argument, not both.")
	
	# z, y = label_zs(stats['z'])
	n_measures = 2
	out_size = [n_measures] + [stats['z'][0].shape[1], n_categories, n_categories]
	print(f'computing pixel-wise distances. Output matrix sized: {out_size}')
	pixelwise_mean = [np.mean(z, axis=0) for z in stats['z']]
	pixelwise_std = [np.std(z, axis=0) for z in stats['z']]

	y_distance_mean_std = np.zeros(out_size)
	
	for k, measure in enumerate([pixelwise_mean, pixelwise_std]):
		for d_i, m_i in enumerate(measure):
			for d_j, m_j in enumerate(measure):
				distance = np.linalg.norm(m_i - m_j)
				y_distance_mean_std[k, :, d_i, d_j] = distance
	
	return y_distance_mean_std


def calculate_distance(stats, att, joint=False, measure=None, n_reps=100,
		exclude_shared=True):
	if isinstance(att, Attributes):
		att=att.df # adapt to new Attributes object
	n_att = len(att.columns)
	distances = np.zeros(shape=(n_att, n_att))
	if joint and measure:
		raise ValueError("set either joint or measure argument, not both.")

	if joint:
		joint_stats = []
		# stds, means = stats['std'], stats['mean'] # 2 lists
		std_mean = np.concatenate([stats['std'][:,np.newaxis],
			                         stats['mean'][:,np.newaxis]], axis=1)
		# check if size == 19962 * 2
		for i in range(n_att):
			mask_i = att.iloc[:, i].astype(bool)
			s_m_i = std_mean[mask_i, :] # mask_i indexing dim=0???

			for j in range(n_att):
				mask_j = att.iloc[:, j].astype(bool)
				s_m_j = std_mean[mask_j, :]

				min_rows = np.min((s_m_i.shape[0], s_m_j.shape[0]))
				distance = np.linalg.norm(s_m_i[:min_rows] - s_m_j[:min_rows])
				distances[i, j] = distance
	else:
		count_atts = pd.DataFrame(data=None, index=att.columns, columns=att.columns, dtype=object)
		for i in range(n_att):
			mask_i = np.array(att.iloc[:, i].astype(bool))
			for j in range(n_att):
				mask_j = np.array(att.iloc[:, j].astype(bool))
				# remove overlaps: 
				if exclude_shared:
					mask_for_i = np.logical_xor(mask_i, mask_j) & mask_i
					mask_for_j = np.logical_xor(mask_i, mask_j) & mask_j
				else:
					mask_for_i = mask_i.copy()
					mask_for_j = mask_j.copy()
				meas_i = stats[measure][mask_for_i]
				meas_j = stats[measure][mask_for_j]
				count_atts.iloc[i, j] = (np.sum(mask_for_i), np.sum(mask_for_j))
				min_rows = np.min((meas_i.shape[0], meas_j.shape[0]))
				dist_avg = 0
				for r in range(n_reps):
					np.random.shuffle(meas_i)
					np.random.shuffle(meas_j)
					# distance = np.linalg.norm(meas_i[:min_rows] - meas_j[:min_rows])
					d_rep = np.sum(np.abs(meas_i[:min_rows] - meas_j[:min_rows]))/min_rows
					dist_avg = dist_avg + (d_rep-dist_avg)/ (r+1)
				distances[i, j] = dist_avg
	return distances, count_atts


def noheatmap(att_count_mtx, filename, plot_title=None):
	n_att = len(att_count_mtx.columns)
	'''we are running out of names'''
	fig, ax = plt.subplots(figsize=(12, 12))
	ax.set_xticks(range(n_att))
	ax.set_xticklabels(att_count_mtx.columns, rotation='vertical', fontsize='xx-small')
	# ax.set_xlim([-1, 40])
	ax.set_yticks(range(n_att))
	ax.set_yticklabels(att_count_mtx.columns, fontsize='xx-small')
	# ax.set_ylim([-2, 45])
	# ax.xticks(range(n_att), 
	a = np.array([0.9, 1]*20)[:,np.newaxis]
	grid_image = a @ a.T
	im = ax.imshow(grid_image, cmap="gray", vmin=0, vmax=1)
	for i in range(n_att):
		for j in range(n_att):
			if i != j:
				val = att_count_mtx.iloc[i, j]
				text = ax.text(j, i, f'({val[0]},\n{val[1]})', ha='center', va='center', size='xx-small')
	if plot_title is None:
		plot_title = 'Attribute count (non overlapping)'
	ax.set_title(plot_title)
	fig.tight_layout()
	plt.savefig(filename, bbox_inches='tight')
	plt.close()
	print(f'Plot saved to {filename}')


	

def heatmap(square_mtx, att, filename, plot_title="Magnitude of distance between digits",
		norm=False):

	if isinstance(att, Attributes):
		att=att.df # adapt to new Attributes object
	n_att = len(att.columns)

	fig, ax = plt.subplots(figsize=(10, 10))
	# configure main plot
	# show img, removing 0's to center color palette distribution. 
	norm_sq_mtx = square_mtx.copy()

	if square_mtx.shape[0] == 1: 
		# for new y_distance_z function
		square_mtx = square_mtx.reshape(square_mtx.shape[1:])
		norm_sq_mtx = norm_sq_mtx.reshape(norm_sq_mtx.shape[1:])

	# remove diagonal
	diagonal_sel = np.diag_indices(n_att)
	norm_sq_mtx[diagonal_sel] = None
	# subtract lowest value and divide
	norm_sq_mtx -= np.nanmin(norm_sq_mtx)
	norm_sq_mtx /= np.nanmax(norm_sq_mtx)
	im = ax.imshow(norm_sq_mtx, cmap="plasma")

	ax.set_xticks(range(n_att))
	ax.set_xticklabels(att.columns, rotation='vertical', fontsize='xx-small')
	ax.set_yticks(range(n_att))
	ax.set_yticklabels(att.columns, fontsize='xx-small')
	# ax.xticks(range(n_att), 
	im = ax.imshow(norm_sq_mtx, cmap="plasma")

	# annotate values within squares
	for i in range(n_att):
		for j in range(n_att):
			if i != j:
				val = norm_sq_mtx[i, j]
				col = 'w' if val < 0.6 else 'b'
				text = ax.text(j, i, "{:.2f}".format(square_mtx[i, j]),
				       ha='center', va='center', color=col, size='xx-small')
			else:
				break
	
	ax.set_title(plot_title)
	fig.tight_layout()
	plt.savefig(filename, bbox_inches='tight')
	plt.close()
	print(f'Plot saved to {filename}')




def violin_eachdigit(stats, att, filename, n_epoch):
	positions = [i for i in range(len(att.columns))]
	stds, means = [stats['std'], stats['mean']]

	fig = plt.figure()
	ax = fig.add_subplot()
	plt.title('z space stats for model at epoch {}'.format(n_epoch))
	plt.ylabel('Average $z$ values')
	plt.xlabel('Digits')
	
	# axs.set_title('Stats for epoch {} overview'.format(n_epoch))
	# 1. average std per digit
	std_each_digit = [np.mean(std) for std in stds]
	# 2. vector means for each digit
	mean_each_digit = [np.mean(m) for m in means]
	strings = ax.violinplot(dataset=means, positions=positions, showmedians=True, showextrema=False)

	for i, b in enumerate(strings['bodies']):
		b.set_facecolor(colors[i])
		b.set_edgecolor(color_variant(colors[i]))
		b.set_alpha(.6)

	'''
	axs[row,col].annotate('$\mu$: {:.2f}\n$\sigma^2$: {:.2f}'.format(ctrd_x, ctrd_y), (ctrd_x, ctrd_y))
	'''
	# fig.suptitle('Stats for epoch {} overview'.format(n_epoch))
	fig.tight_layout()
	# fig.subplots_adjust(top=.70) # .88
	#for dig in range(10):
	#	plt.scatter(means[dig], stds[dig], c=colors[dig], label=str(dig), alpha=.8, s=.8)
	plt.savefig(filename, bbox_inches='tight')
	plt.close()
	print('\nPlot saved to ' + filename)


# def tellme_ys(net, loader, device):
# 
# 	with torch.no_grad():
# 		with tqdm(total=len(loader.dataset)) as progress:
# 			attr_y = np.array([]).reshape(0, 40) # for i in range(10)
# 
# 			for _, y in loader:
# 				attr_y = np.concatenate(( attr_y, y.to('cpu').detach().numpy() ))
# 				progress.update(y.size(0))
# 
# 	return attr_y

def scatter_attr(stats, att, filename, dataset):
	stds, means = [stats['std'], stats['mean']]
	if isinstance(att, Attributes):
		att = att.df
	# Steps to replicate for each subplot:
	# plt.title('z space stats for model at epoch {}'.format(n_epoch))
	# plt.xlabel('mean')
	# plt.ylabel('std')
	nrows, ncols = (5, 8) # = 40
	fig, axs = plt.subplots(nrows, ncols, sharex='all', sharey='all', figsize=(10, 7))
	
	cmap = plt.get_cmap('gist_rainbow')
	# cmaplist = [cmap(i) for i in range(cmap.N)]
	# cmap = LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
	bounds = np.linspace(0, 39, 40)
	normidx = BoundaryNorm(bounds, cmap.N)
	
	# for n_att in range(len(att.columns)):
	n_att = 0.
	for col in range(ncols):
		for row in range(nrows):
			att_name = att.columns[row+col*nrows]
			mus = means[att[att_name].astype(bool)]
			att_ds_n = mus.shape[0]
			sigmas = stds[att[att_name].astype(bool)]
			# print('{}: {}, {}'.format(att_name, mus.shape, sigmas.shape))
			axs[row,col].set_title(att_name+ f' $n={att_ds_n}$' , fontsize='xx-small')
			axs[row,col].scatter(mus, sigmas, s=1,
			                     cmap=cmap, c=np.repeat(n_att, mus.shape[0]),
			                     norm=normidx, label=att_name)
			# ctrd_x = np.mean(means[dig])
			# ctrd_y = np.mean(stds[dig])
			# axs[row,col].scatter(ctrd_x, ctrd_y, c=color_variant(colors[dig]))
			if row == nrows-1:
				axs[row,col].set_xlabel('means', fontsize='small')
			if col == 0:
				axs[row,col].set_ylabel('std', fontsize='small')
			# axs[row,col].annotate('$\mu$: {:.2f}\n$\sigma$: {:.2f}'.format(
			# 	                           ctrd_x, ctrd_y), (ctrd_x, ctrd_y))
			n_att += 1 


	fig.suptitle(f'Z {dataset} overview')
	fig.tight_layout()
	fig.subplots_adjust(top=.88)
	plt.savefig(filename)
	plt.close()
	print('\nPlot saved to ' + filename)


def color_variant(hex_color, brightness_offset=-50):
	""" takes a color like #87c95f and produces a lighter or darker variant """
	if len(hex_color) != 7:
		raise Exception("Passed %s into color_variant(), needs to be in #87c95f format." % hex_color)
	rgb_hex = [hex_color[x:x+2] for x in [1, 3, 5]]
	new_rgb_int = [int(hex_value, 16) + brightness_offset for hex_value in rgb_hex]
	new_rgb_int = [min([255, max([0, i])]) for i in new_rgb_int] # make sure new values are between 0 and 255
	# hex() produces "0x88", we want just "88"
	hex_string = [hex(i)[2:] for i in new_rgb_int]
	hex_string = [val if len(val) == 2 else "0" + val for val in hex_string]
	return "#" + "".join(hex_string)


def scatter_all(stats, filename, n_epoch=False):
	colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
			  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
			  '#bcbd22', '#17becf']
	fig, ax = plt.subplots()
	stds, means = [stats['std'], stats['mean']]
	if n_epoch:
		plt.title('z space stats for model at epoch {}'.format(n_epoch))
	plt.xlabel('mean')
	plt.ylabel('std')

	ax.scatter(means, stds, c='#fab111', alpha=.95, s=.8)
	# ax.legend()
	plt.savefig(filename, bbox_inches='tight')
	plt.close()
	print('\nPlot saved to: ' + filename)


def grand_z(stats, att, filename=None, reps=1):

	Z = stats['z']

	n_att = len(att.df.columns)
	H_W = Z.shape[2:]
	all_zs = np.zeros(shape=(n_att*reps, 3, *H_W))

	for a in range(n_att):
		for r in range(reps):
			a_z = Z[att.df.iloc[:, a].astype(bool)]
			grand_nz = np.mean(a_z, axis=0)
			all_zs[ a * reps + r] = grand_nz
	
	return all_zs

# def loop_stats(track_dataset_stats):
# 	''' decorator for tracking function providing different loaders'''
# 	if dataset=='celeba':
# 		from load_data import sample_celeba
# 		def track_f(*args, **kwargs):
# 	
# 		return track_z(*args, **kwargs)
# 	elif dataset == 'ffhq':
# 		def track_f(*args, **kwargs):
# 			return 
# 
# 
# 	return track_f

def track_z(net, device, img_size, loader=None, n_bits=5,
	                 y_all_p = 'data/1_den_celeba/attr_y.pkl', stats=['Z'],
	                 batch_size=32, **kwargs):

	if any([s.islower() for s in stats]):
		raise ValueError
	if not loader:
		loader = load_celeba(batch_size, img_size, test=True)

	net.eval()
	if type(net.module).__name__ == 'Glow':
		n_bins = 2. ** n_bits
	out_dict = dict()
	# arrays for concatenation
	STD = np.array([])
	MEAN = np.array([])
	Z = np.array([], dtype=np.float32).reshape(0, 3, img_size, img_size)
	X = np.array([], dtype=np.float32).reshape(0, 3, img_size, img_size)
	axs = [1, 2, 3]
	with tqdm(total=len(loader.dataset)) as progress:
		with torch.no_grad():
			for x, y in loader:
				x = x.to(device)
				if type(net.module).__name__ == 'Glow':
					log_p, logdet, z = net(x + torch.rand_like(x) / n_bins)
					z = net(z, reverse=True, reconstruct=True)
				if 'MEAN' in stats:
					MEAN = np.concatenate(( MEAN, z.mean(axis=axs)
					                     .to('cpu').detach().numpy()))
				if 'STD' in stats:
					STD = np.concatenate(( STD, z.std(axis=axs)
					                     .to('cpu').detach().numpy()))
				if 'Z' in stats:
					Z = np.concatenate(( Z, z.to('cpu').detach().numpy() ))
				if 'X' in stats:
					X = np.concatenate(( X, x.to('cpu').detach().numpy() ))
				progress.update(x.size(0))
	for s in stats:
		exec(f'out_dict[s] = {s}')
	return out_dict


def save_stats(stats, stats_filename):
	import sys
	total_size = 0
	for v in stats.values():
		total_size +=sys.getsizeof(v)
	import ipdb; ipdb.set_trace()
	if total_size >= 4294960000: # ~ 4 Gibibytes (- 7k bytes)
		dir_name = os.path.dirname(stats_filename)
		warn.warnings("Stats named changed because of size constraints")
		for k in stats.keys():
			torch.save(stats[k], stats_filename.replace("z_mean_std", k))
	else:
		torch.save(stats, stats_filename)
	return stats

def load_stats(fp_model_root, keys=('z', 'mean', 'std')):
	dirs = os.listdir(fp_model_root)
	pkl_fns = [fn for fn in dirs if fn.endswith('.pkl')]
	single_stat_names = {f'{s}.pkl' in s in keys}
	single_pickled_filenames = single_stat_names.intersection(set(pkl_fns))
	if "z_mean_std.pkl" in pkl_fns:
		stats = torch.load(f'{fp_model_root}/z_mean_std.pkl')
	elif len(single_pickled_filenames) > 0:
		stats = dict()
		for k in keys:
			stats[k] = torch.load(f'{fp_model_root}/{k}.pkl')
	return stats


def verify_version(fp, version_string):
	''' helper function to define what version of analysis was performed. 
	Used to select a model without given analysis version #.'''
	if os.path.isfile(fp):
		with open(fp, 'r+') as f:
			l = f.readline()

			while l:
				print("debugging value for l: " + l)

				if l.strip('\n') == version_string:
					# if analysis # `version_string` was completed before:
					print("Matched version: " + version_string)
					return True
					break
				else:
					# TODO: fix eternal loop because f doesn't validate the while loop.
					instead = l.strip()
					print(f'Unmatched: {instead} != {version_string} ({fp}).', end='')
					print(' Continuing...')
				l = f.readline()
		return False
	else: 
		print(f'File {fp} not found.')
		return False # if file doesn't exist.


def mark_version(version_str, fp_vmarker, finish=False, sep='-'):
	''' write first and last parts of `version_str` to `fp_vmarker`
	Args:
		version_str
		fp_vmarker: str -- version marker file path
		finish: final call, else False
		sep: define first and last parts of `version_str`.
	'''
	vmarker = version_str.split(sep)
	m = open(fp_vmarker, 'a')
	if not finish:
		vmarker[0] += sep
		m.write('\n'+vmarker[0])
	else:
		m.write(vmarker[1]+'\n')
		# must end with a newline. byebye!
	m.close()

# def cleanup_version_f(fp_vmarker):
# 
# 	tmp_fp_vmarker = '/home/mao540/tmp_realnvp' + fp_vmarker.replace('/', '')
# 
# 	with open(fp_vmarker, 'r') as v:
# 		with open(tmp_fp_vmarker, 'w') as t:
# 			for l in v:
# 				stripped = l.strip()
# 				if stripped == 'V-' or stripped == '':
# 					continue
# 				else:
# 					t.write(l)
# 	shutil.move(tmp_fp_vmarker, fp_vmarker)


if __name__ == '__main__':

	conf_name = 'config/ffhq128_c.yml'
	parser = ArgumentParser(description='RealNVP training on various datasets.')
	parser.add_argument('--config', '-c', default=conf_name)
	parser.parse_args()

	C = ConfWrap(fn=parser.config)
	# C = ConfWrap(fn='config/ffhq128_c.yml')
	# C = ConfWrap(fn='config/glow_celeba.yml')
	# C.version = 'V-1.1' # remove?

	C.track_y = False
	main(C)

