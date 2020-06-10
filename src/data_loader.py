import numpy as np
import pandas as pd
import ast
import os
from PIL import Image, ExifTags#, ImageOps

import constants as const

VERIFY_BATCHES = False
PLOT_SAMPLE_DATA = True
NUMERIC_COLS = ['craving', 'minutes_since_last_smoke']


def main():

	dl = DataLoader()

	print('Total images:', len(dl.data['all']))

	print('Train data:')
	print(dl.data['train'][const.OUTCOMES])

	print('Val data:')
	print(dl.data['val'][const.OUTCOMES])

	print('Test data:')
	print(dl.data['test'][const.OUTCOMES])

	if VERIFY_BATCHES:

		for part in ['train', 'val', 'test']:

			print('Verifying %s batches:' % part)

			for i, (batch_images, batch_x, batch_y) in enumerate(
				dl.get_batch(part, 100)):
				
				print(
					'Batch %i: imagefiles shape' % i,
					np.shape(batch_images),
					'and features shape',
					np.shape(batch_x),
					'and outcomes shape',
					np.shape(batch_y))

	if PLOT_SAMPLE_DATA:

		print('Displaying random sample:')

		import matplotlib.pyplot as plt

		images, features, y = dl.sample_data(
			normalize_images=False, n=8)

		fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))

		for i, (xi, yi) in enumerate(zip(images, y)):

			ax[i // 4, i % 4].imshow(xi.astype(int))
			outcomes_text = ['%s: %s' % (o, str(v)) for o, v in zip(const.OUTCOMES, yi)]
			ax[i // 4, i % 4].set_title('\n'.join(outcomes_text))
			ax[i // 4, i % 4].axis('off')

		plt.tight_layout()
		plt.show()


class DataLoader:

	def __init__(
		self, n_folds=5, val_fold=3, test_fold=4,
		partition_method='participant',
		pid_features=True,
		nrows=None, **kwargs):

		self.datadir = os.path.join(
			const.DATA_DIR,
			'deeplearning',
			'prepdata')

		self.imagedir = os.path.join(
			const.DATA_DIR,
			'deeplearning',
			'EMA data and Photos')

		self.partition_method = partition_method
		self.is_categorical = np.array(
			[const.VARTYPES[o] == 'categorical' for o in const.OUTCOMES])

		self.n_out = len(const.OUTCOMES)

		all_data = pd.read_csv(os.path.join(
			self.datadir,
			'all_data.csv')).set_index(['pid', 'prompt', 'index'])

		na_rows = all_data[const.OUTCOMES + ['filename']].isna().any(axis=1)
		all_data = all_data[~na_rows]
		print('Removed %i rows with missing filenames or outcomes' % sum(na_rows))

		all_data['filename'] = all_data['filename'].apply(lambda x: os.path.join(self.imagedir, x))

		all_data = self._remove_missing_files(all_data)

		if pid_features:

			pid_series = all_data.index.get_level_values('pid')

			for pid in pid_series.unique():

				all_data[pid] = (pid_series == pid).astype(float)

		self.feature_cols = [x for x in all_data.columns if not x in const.OUTCOMES]
		self.n_features = len(self.feature_cols)

		self.data = dict()
		self.data['all'] = all_data.sample(frac=1, random_state=0)# shuffle rows

		if partition_method == 'participant':

			pids = self.data['all'].index.get_level_values('pid').unique()
			fold_idx = get_fold_indices(n_folds, len(pids))

			val_pidx = fold_idx[val_fold]
			test_pidx = fold_idx[test_fold]
			train_pidx = ~val_pidx & ~test_pidx

			val_pids = pids[val_pidx]
			test_pids = pids[test_pidx]
			train_pids = pids[train_pidx]

			val_idx = self.data['all'].index.get_level_values('pid').isin(val_pids)
			test_idx = self.data['all'].index.get_level_values('pid').isin(test_pids)
			train_idx = self.data['all'].index.get_level_values('pid').isin(train_pids)

		elif partition_method == 'image':

			fold_idx = get_fold_indices(n_folds, len(self.data['all']))

			val_idx = fold_idx[val_fold]
			test_idx = fold_idx[test_fold]
			train_idx = ~val_idx & ~test_idx

		elif partition_method == 'longitudinal':

			test_idx = self.data['all'].index.get_level_values('Day') >= 10

			np.random.seed(0)
			train_or_val_idx = np.random.rand(len(self.data['all'])) < .8
			np.random.seed()

			train_idx = ~test_idx & train_or_val_idx
			val_idx = ~test_idx & ~ train_or_val_idx

		else:

			assert False, 'Invalid partition method'		

		self.data['train'] = self.data['all'][train_idx]
		self.n_train = len(self.data['train'])

		print('There are %i images in the training set' % self.n_train)

		self.data['val'] = self.data['all'][val_idx]
		self.n_val = len(self.data['val'])

		print('There are %i images in the validation set' % self.n_val)

		self.data['test'] = self.data['all'][test_idx]
		self.n_test = len(self.data['test'])

		print('There are %i images in the test set' % self.n_test)

		self.train_mean = self.data['train'][const.OUTCOMES].mean(axis=0)
		self.train_std = self.data['train'][const.OUTCOMES].std(axis=0)

		self._normalize_numeric()


	def get_batch(self, part, batch_size, imgfmt='array', normalize=True):

		assert part in ['all', 'train', 'val', 'test']
		assert imgfmt in ['name', 'array']
		
		l = len(self.data[part])

		for ndx in range(0, l, batch_size):

			endx = min(ndx + batch_size, l)

			data = self.data[part].iloc[ndx:endx, :]

			fns, x, y = self._split_data(data)

			if imgfmt == 'name':

				yield fns, x, y

			elif imgfmt == 'array':

				yield np.squeeze(images_from_files(fns, (224, 224)), axis=1), x, y


	def sample_data(self, part='all', n=1, imgfmt='array', normalize_images=True):

		assert part in ['all', 'train', 'val', 'test']
		assert imgfmt in ['name', 'array']

		if n == -1:

			s = self.data[part].sample(frac=1)

		else:

			s = self.data[part].sample(n=n)

		fns, x, y = self._split_data(s)

		if imgfmt == 'name':

			return fns, x, y

		elif imgfmt == 'array':

			return np.squeeze(images_from_files(fns, (224, 224), normalize=normalize_images)), x, y


	def _remove_missing_files(self, df):

		exists = df['filename'].apply(lambda x: os.path.exists(str(x)))

		print('%i files were not found:' % sum(~exists))
		print(df['filename'][~exists])

		df['filename'][~exists].to_csv('../results/not_found.csv')

		return df[exists]


	def _normalize_numeric(self):

		for col in NUMERIC_COLS:

			train_mean = self.data['train'][col].mean(axis=0)
			train_std = self.data['train'][col].std(axis=0)

			self.data['all'].loc[:, col] = ((self.data['all'][col] - train_mean) / train_std).fillna(0)
			self.data['train'].loc[:, col] = ((self.data['train'][col] - train_mean) / train_std).fillna(0)
			self.data['val'].loc[:, col] = ((self.data['val'][col] - train_mean) / train_std).fillna(0)
			self.data['test'].loc[:, col] = ((self.data['test'][col] - train_mean) / train_std).fillna(0)


	def _split_data(self, df):

		filenames = df[['filename']].values
		features = df[self.feature_cols].values
		outcomes = df[const.OUTCOMES].values

		return filenames, features, outcomes


def listdir_by_ext(directory, extension=None):

	if extension is None:

		return os.listdir(directory)

	else:

		return [x for x in os.listdir(directory)
				if os.path.splitext(x)[-1] == extension]


def get_filecols(df):

	return [x for x in df.columns.values if (x[-7:] == 'FILE_ID')]


def image_from_file_or_npy(fn, shape=(224, 224), normalize=True):

	fn_npy = os.path.splitext(fn)[0] + '.npy'

	if os.path.isfile(fn_npy):

		imagearr = np.load(fn_npy)

		if not normalize:
			imagearr = (imagearr + 1) * 127.5

		return imagearr

	elif os.path.isfile(fn):

		imagearr = image_from_file(fn, shape=shape, normalize=True)
		np.save(fn_npy, imagearr)

		if not normalize:
			imagearr = (imagearr + 1) * 127.5

		return imagearr

	else:

		print(fn, 'not found; returning array of zeros')
		return np.zeros(shape + (3, ))


def image_from_file(fn, shape=(224, 224), normalize=True):
	"""Get raw image from filename"""
	
	with Image.open(fn) as image:

		try:

			#image = ImageOps.exif_transpose(image)
			image = correct_orientation(image)

		except:

			pass

		image = image.resize(shape, resample=Image.BILINEAR)
		
		imagearr = np.array(image, dtype='f')

		if np.shape(imagearr)[-1] == 4:

			imagearr = imagearr[:, :, :3]
	
	if normalize:
		
		return imagearr / 127.5 - 1
	
	else:
		
		return imagearr


def images_from_files(fns, shape, normalize=True, use_npy=True):
	"""Get stack of images from filenames"""
	if len(np.shape(fns)) == 1:
		if use_npy:
			return np.stack([image_from_file_or_npy(
				fn, shape, normalize=normalize) for fn in fns])
		else:
			return np.stack([image_from_file(
				fn, shape, normalize=normalize) for fn in fns])
	else:
		return np.stack([images_from_files(
			fn, shape, normalize=normalize) for fn in fns])


for ORIENTATION_TAG in ExifTags.TAGS.keys():
	if ExifTags.TAGS[ORIENTATION_TAG]=='Orientation':
		break


def correct_orientation(image):

	try:
		
		exif=dict(image._getexif().items())

		if exif[ORIENTATION_TAG] == 3:
			image=image.rotate(180, expand=True)
		elif exif[ORIENTATION_TAG] == 6:
			image=image.rotate(270, expand=True)
		elif exif[ORIENTATION_TAG] == 8:
			image=image.rotate(90, expand=True)

	except (AttributeError, KeyError, IndexError):
		# cases: image don't have getexif
		pass

	return image


def get_fold_indices(n_folds, l):

	pos = np.linspace(0, l, n_folds + 1, dtype=int)
	indices =[]

	for i in range(n_folds):
		idx = np.array([False] * l)
		idx[pos[i]:pos[i + 1]] = True
		indices.append(idx)

	return indices


if __name__ == '__main__':
	main()

