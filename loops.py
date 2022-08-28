
def track_z_celeba(net, device, img_size, loader=None, track_x=False, n_bits=5,
	                 y_all_p = 'data/1_den_celeba/attr_y.pkl', stats=['Z'], **kwC):

	if any([s.islower() for s in stats]):
		raise ValueError

	net.eval()
	if type(net.module).__name__ == 'Glow':
		n_bins = 2. ** n_bits

	# arrays for concatenation
	STD = np.array([])
	MEAN = np.array([])
	Z = np.array([], dtype=np.float32).reshape(0, 3, img_size, img_size)
	X = np.array([], dtype=np.float32).reshape(0, 3, img_size, img_size)
	axes = [1, 2, 3]
	if track_x:
	with tqdm(total=len(loader.dataset)) as progress:
		with torch.no_grad():
			for x, y in loader:
				x = x.to(device)
				if type(net.module).__name__ == 'Glow':
					log_p, logdet, z = net(x + torch.rand_like(x) / n_bins)
					z = net(z, reverse=True, reconstruct=True)
				if 'MEAN' in stats:
					MEAN = np.concatenate(( MEAN, z.mean(axis=axes)
					                              .to('cpu').detach().numpy()))
				if 'STD' in stats:
					stds = z.std(axis=axes) # dim=[1, 2, 3])
					STD = np.concatenate(( STD, stds.to('cpu').detach().numpy() ))
				if 'Z' in stats:
					Z = np.concatenate(( Z, z.to('cpu').detach().numpy() ))
				if 'X' in stats:
					X = np.concatenate(( X, x.to('cpu').detach().numpy() ))
				progress.update(x.size(0))
	for s in stats:
		exec(f'stats[s] = {s}')

	if 'x' in stats:
		return {'std': STD, 'mean': MEAN, 'z': Z}, {'x': X}
	else:
		return {'std': STD, 'mean': MEAN, 'z': Z}
