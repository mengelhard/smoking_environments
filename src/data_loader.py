import numpy as np
import pandas as pd
import ast
import os
from PIL import Image, ExifTags#, ImageOps

import constants as const

VERIFY_BATCHES = False
PLOT_OUTCOMES = True


def main():

	dl = DataLoader()

	print('Total participants:', len(dl.data['all']))

	print('Train data:')
	print(dl.data['train'][const.OUTCOMES])

	print('Val data:')
	print(dl.data['val'][const.OUTCOMES])

	print('Test data:')
	print(dl.data['test'][const.OUTCOMES])

	print('Value counts in training set:')

	if PLOT_OUTCOMES:

		import matplotlib.pyplot as plt

		for o in const.OUTCOMES:

			plt.hist(dl.data['train'][o].astype(float))
			plt.savefig('../results/%s_hist.png' % o)
			plt.close()


	if VERIFY_BATCHES:

		print('Verifying training batches:')

		for i, (batch_x, batch_y) in enumerate(dl.get_batch('train', 10)):
			
			print(
				'Batch %i: imagefiles shape' % i,
				np.shape(batch_x),
				'and outcomes shape',
				np.shape(batch_y))

		print('Verifying validation batches:')

		for i, (batch_x, batch_y) in enumerate(dl.get_batch('val', 10)):
			
			print(
				'Batch %i: imagefiles shape' % i,
				np.shape(batch_x),
				'and outcomes shape',
				np.shape(batch_y))

		print('Verifying test batches:')

		for i, (batch_x, batch_y) in enumerate(dl.get_batch('test', 10)):
			
			print(
				'Batch %i: imagefiles shape' % i,
				np.shape(batch_x),
				'and outcomes shape',
				np.shape(batch_y))

	print('Displaying random sample:')

	import matplotlib.pyplot as plt

	x, y = dl.sample_data(normalize_images=False)

	y = np.squeeze(y, axis=0)

	print('Outcomes for displayed images:')
	print(list(zip(const.OUTCOMES, y)))

	if SINGLE_IMAGE:
		plt.imshow(np.squeeze(x).astype(int))
		plt.axis('off')
		plt.tight_layout()
		plt.show()

	else:

		fig, ax = plt.subplots(
			ncols=5,
			nrows=int(np.ceil(len(x) / 5)),
			figsize=(15, 6))

		for i, img in enumerate(x):
			a = ax[i // 5, i % 5]
			a.imshow(x[i, :, :, :].astype(int))
			a.axis('off')

		plt.tight_layout()
		plt.show()


class DataLoader:

	def __init__(
		self, n_folds=5, val_fold=3, test_fold=4,
		dichotomize=None, nrows=None, **kwargs):

		self.datadir = os.path.join(
			check_directories(const.DATA_DIRS),
			'deeplearning')

		df_smok, smok_coldicts = self._get_datafile('smok', nrows=nrows)
		df_smok['smoke'] = 'Yes'

		df_non, non_coldicts = self._get_datafile('non', nrows=nrows)
		df_non['smoke'] = 'No'

		assert const.ITEMS['smok'].keys() == const.ITEMS['non'].keys()

		for o in const.ITEMS['smok'].keys():

			if o == 'smoking':
				continue

			smok_items = const.ITEMS['smok'][o]
			non_items = const.ITEMS['non'][o]

			for smok_item, non_item in zip(smok_items, non_items):

				v0 = smok_coldicts[0].get(smok_item, np.nan)
				v1 = smok_coldicts[1].get(smok_item, np.nan)
				v2 = non_coldicts[0].get(non_item, np.nan)
				v3 = non_coldicts[1].get(non_item, np.nan)

				vstr = '\n'.join(['V%i: %s' % (i, str(v)) for i, v in enumerate(
					[v0, v1, v2, v3])])

				assert (v0 == v1) and (v1 == v2) and (v2 == v3), vstr

		self.dichotomize = dichotomize
		self.is_categorical = np.array(
			[const.VARTYPES[o] == 'categorical' for o in const.OUTCOMES])

		data_smok = self._get_image_filenames(df_smok).join(
			self._get_outcomes(df_smok, 'smok', dichotomize=dichotomize))

		data_non = self._get_image_filenames(df_non).join(
			self._get_outcomes(df_non, 'non', dichotomize=dichotomize))

		self.n_out = len(const.OUTCOMES)
		self.n_images = len(const.IMAGES)

		self.data = dict()

		self.data['all'] = pd.concat([data_smok, data_non], axis=0).sample(
			frac=1, random_state=0) # shuffle rows

		if single_image:
			frames = []
			for img in const.IMAGES:
				frame = self.data['all'][[img] + const.OUTCOMES]
				frame.columns = ['image'] + const.OUTCOMES
				frames.append(frame)
			self.data['all'] = pd.concat(frames, axis=0).sample(
				frac=1, random_state=0)

		fold_idx = get_fold_indices(n_folds, len(self.data['all']))

		val_idx = fold_idx[val_fold]
		test_idx = fold_idx[test_fold]
		train_idx = ~val_idx & ~test_idx

		self.data['train'] = self.data['all'][train_idx]
		self.n_train = len(self.data['train'])

		print('There are %i subjects in the training set' % self.n_train)

		self.data['val'] = self.data['all'][val_idx]
		self.n_val = len(self.data['val'])

		print('There are %i subjects in the validation set' % self.n_val)

		self.data['test'] = self.data['all'][test_idx]
		self.n_test = len(self.data['test'])

		print('There are %i subjects in the test set' % self.n_test)

		self.train_mean = self.data['train'][const.OUTCOMES].mean(axis=0)
		self.train_std = self.data['train'][const.OUTCOMES].std(axis=0)


	def get_batch(self, part, batch_size, imgfmt='array', normalize=True):

		assert part in ['all', 'train', 'val', 'test']
		assert imgfmt in ['name', 'array']
		
		l = len(self.data[part])

		for ndx in range(0, l, batch_size):

			endx = min(ndx + batch_size, l)

			data = self.data[part].iloc[ndx:endx, :]

			fns, y = self._split_images_and_outcomes(data)

			if imgfmt == 'name':

				yield fns, self._normalize_outcomes(y)

			elif imgfmt == 'array':

				yield np.squeeze(images_from_files(fns, (224, 224))), self._normalize_outcomes(y)


	def _normalize_outcomes(self, outcomes):

		if self.dichotomize == True:

			return outcomes

		elif self.dichotomize == False:

			return (outcomes - self.train_mean[np.newaxis, :]) / self.train_std[np.newaxis, :]

		else:

			normalized = (outcomes - self.train_mean[np.newaxis, :]) / self.train_std[np.newaxis, :]
			normalized[:, self.is_categorical] = outcomes[:, self.is_categorical]

			return normalized


	def sample_data(
		self, part='all', n=1, imgfmt='array',
		normalize_outcomes=True, normalize_images=True):

		assert part in ['all', 'train', 'val', 'test']
		assert imgfmt in ['name', 'array']

		if n == -1:

			s = self.data[part].sample(frac=1)

		else:

			s = self.data[part].sample(n=n)

		x, y = self._split_images_and_outcomes(s)

		if normalize_outcomes:

			y = self._normalize_outcomes(y)

		if imgfmt == 'name':

			return x, y

		elif imgfmt == 'array':

			return np.squeeze(images_from_files(x, (224, 224), normalize=normalize_images)), y


	def _split_images_and_outcomes(self, df):

		filenames = df.drop(const.OUTCOMES, axis=1).values
		outcomes = df[const.OUTCOMES].values

		return filenames, outcomes


	def _get_datafile(self, group, nrows=None):

		subdirs = const.DATA_SUBDIRS[group]

		frames = []
		coldicts = []

		for subdir in subdirs:

			d = os.path.join(self.datadir, subdir)

			fns = listdir_by_ext(d, '.csv')

			print('Found %i .csv files in' % len(fns), subdir)
			print('Reading', fns[-1])

			df, coldict, reversedict = self._read_imageturk_csv(
				os.path.join(d, fns[-1]), nrows)

			df = self._filter_imageturk_csv(df, reversedict)

			df['base_dir'] = d

			frames.append(df)
			coldicts.append(coldict)

		return pd.concat(frames, axis=0, sort=True), coldicts


	def _read_imageturk_csv(self, fn, nrows=None):

		df = pd.read_csv(fn, header=[0, 1, 2], nrows=nrows)
		
		colnames = [ast.literal_eval(x[2])['ImportId'] for x in df.columns.values]

		assert len(colnames) == len(set(colnames))

		coltext = [x[1] for x in df.columns.values]
		coldict = {x:y for x, y in zip(colnames, coltext)}
		reversedict = {y:x for x, y in zip(colnames, coltext)}

		df.columns = colnames

		return df.set_index('_recordId'), coldict, reversedict


	def _filter_imageturk_csv(self, df, coldict):

		imagecols = get_filecols(df)
		sizecols = [x.split('_')[0] + '_FILE_SIZE' for x in imagecols]
		imagesizes = np.array([df[x].fillna(1e6).values for x in sizecols])

		exclusion_criteria = [
			df['distributionChannel'] == 'preview',
			~df['finished'],
			df[coldict['At birth, were you described as:']].isna(),
			np.any(imagesizes < 1e5, axis=0)
		]

		fdf = df[~np.any(exclusion_criteria, axis=0)]

		#print('The following columns have null values:')
		#print(fdf.columns[fdf.isnull().any()].values)
		#print(fdf.isna().sum().reset_index().values)

		return fdf


	def _get_outcomes(self, df, group, dichotomize=None):
		return pd.DataFrame(
			{o: const.score_outcome(
				df, group, o, dichotomize=dichotomize) for o in const.OUTCOMES},
			index=df.index)


	def _get_image_filenames(self, df, verify=True):

		cols = const.IMAGES

		fn_dict = {c: imageturk_fn_from_qcol(
			df, c, verify=verify) for c in cols}

		return pd.DataFrame(fn_dict, index=df.index)


def imageturk_fn_from_qcol(df, qcol, verify=True):
	filename = replace_all(
		df[qcol + '_FILE_NAME'],
		['-', ' ', '(', ')', '[', ']', '~', '#'],
		'_')
	filename = replace_all(filename, ['.heic', '.HEIC'], '.jpg')
	basename = df.index + '~' + filename
	fns = df['base_dir'] + '/' + qcol + '_FILE_ID/' + basename
	if verify:
		for fn in fns.values:
			if not os.path.isfile(fn):
				print('%s not found' % fn)
	return fns


def replace_all(series, pattern_list, replacement):
	s = series.copy()
	for pattern in pattern_list:
		s = s.str.replace(pattern, replacement)
	return s


def check_directories(dirlist):

	for i, d in enumerate(dirlist):

		if os.path.exists(d):

			print('Found data directory', d)

			break

		if (i + 1) == len(dirlist):

			print('No data directory found')

			assert False

	return d


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

