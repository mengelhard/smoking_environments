import numpy as np
import pandas as pd
import ast
import os
from PIL import Image, ExifTags#, ImageOps

import constants as const

VERIFY_BATCHES = True
PLOT_OUTCOMES = True
PLOT_SAMPLE_DATA = True


def main():

	dl = DataLoader()

	print('Total images:', len(dl.data['all']))

	print('Train data:')
	print(dl.data['train'][const.OUTCOMES])

	print('Val data:')
	print(dl.data['val'][const.OUTCOMES])

	print('Test data:')
	print(dl.data['test'][const.OUTCOMES])

	# print('Day counts:')
	# print(dl.data['all'].index.get_level_values('Day').value_counts())

	if PLOT_OUTCOMES:

		import matplotlib.pyplot as plt

		for o in const.OUTCOMES:

			plt.hist(dl.data['train'][o].astype(float))
			plt.savefig('../results/%s_hist.png' % o)
			plt.close()

	if VERIFY_BATCHES:

		for part in ['train', 'val', 'test']:

			print('Verifying %s batches:' % part)

			for i, (batch_x, batch_xi, batch_xs, batch_y) in enumerate(
				dl.get_batch(part, 100)):
				
				print(
					'Batch %i: imagefiles shape' % i,
					np.shape(batch_x),
					'and image features shape',
					np.shape(batch_xi),
					'and subject features shape',
					np.shape(batch_xs),
					'and outcomes shape',
					np.shape(batch_y))

	if PLOT_SAMPLE_DATA:

		print('Displaying random sample:')

		import matplotlib.pyplot as plt

		images, image_features, subject_features, y = dl.sample_data(
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
		partition_method='longitudinal',
		pid_features=True,
		dichotomize=None, nrows=None, **kwargs):

		self.datadir = os.path.join(
			check_directories(const.DATA_DIRS),
			'deeplearning', 'EMA data and Photos')

		self.dichotomize = dichotomize
		self.partition_method = partition_method
		self.is_categorical = np.array(
			[const.VARTYPES[o] == 'categorical' for o in const.OUTCOMES])

		self.n_out = len(const.OUTCOMES)

		subject_features = self._load_subject_features()
		self.subject_feature_cols = subject_features.columns.tolist()
		self.n_subject_features = len(self.subject_feature_cols)

		self.n_image_features = 5
		self.image_feature_cols = ['weekend', 'night', 'morning', 'afternoon', 'evening']

		folders = [f for f in os.listdir(self.datadir) if os.path.isdir(
			os.path.join(self.datadir, f))]
		data = [self._read_subject_data(d) for d in folders]

		all_data = self._validate_data(pd.concat(data, axis=0))
		all_data = all_data.join(subject_features, how='left')

		if pid_features:

			pid_series = all_data.index.get_level_values('pid')

			for pid in pid_series.unique():

				all_data[pid] = (pid_series == pid).astype(float)

			self.subject_feature_cols += list(pid_series.unique())
			self.n_subject_features = len(self.subject_feature_cols)

			# pid_onehot = pd.get_dummies(
			# 	all_data.index.get_level_values('pid').astype('category'),
			# 	drop_first=False,
			# 	dtype=float)

			# self.subject_feature_cols += pid_onehot.columns.tolist()

			# all_data = pd.concat(
			# 	[all_data, pid_onehot.reset_index(drop=True)],
			# 	axis=1)

		self.data = dict()
		self.data['all'] = all_data.sample(frac=1, random_state=0)# shuffle rows

		self.data['all'].to_csv('../results/all_data.csv')

		print(
			'Removing nan values:',
			self.data['all'][const.OUTCOMES + self.subject_feature_cols].isna().sum())

		na_rows = self.data['all'][const.OUTCOMES + self.subject_feature_cols].isna().any(axis=1)
		self.data['all'] = self.data['all'][~na_rows]

		#TODO add time of day features

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

		if partition_method == 'participant':

			for part in ['train', 'val', 'test']:

				print(
					part + ' participants:',
					self.data[part].index.get_level_values('pid').unique().values)


	def get_batch(self, part, batch_size, imgfmt='array', normalize=True):

		assert part in ['all', 'train', 'val', 'test']
		assert imgfmt in ['name', 'array']
		
		l = len(self.data[part])

		for ndx in range(0, l, batch_size):

			endx = min(ndx + batch_size, l)

			data = self.data[part].iloc[ndx:endx, :]

			fns, xi, xs, y = self._split_images_and_outcomes(data)

			if imgfmt == 'name':

				yield fns, xi, xs, self._normalize_outcomes(y)

			elif imgfmt == 'array':

				yield np.squeeze(images_from_files(fns, (224, 224)), axis=1), \
					xi, xs, self._normalize_outcomes(y)


	def _load_subject_features(self):

		f1 = pd.read_csv(os.path.join(
			self.datadir,
			'baseline_data.csv')).set_index('pid')

		f2 = pd.read_excel(os.path.join(
			self.datadir,
			'DL Demographics.xlsx'))

		f2['race'] = ((f2['Race'] == 'White') & (f2['Ethnicity'] != 'Hispanic')).astype(float)
		f2['sex'] = (f2['Gender'] == 'Female').astype(float)
		f2['pid'] = f2['Subject ID assigned']

		f2 = f2[['pid', 'sex', 'race']].set_index('pid')

		return f1.join(f2, how='left')


	def _normalize_outcomes(self, outcomes):

		if self.dichotomize == True:

			return outcomes

		elif self.dichotomize == False:

			return (outcomes - self.train_mean[np.newaxis, :]) / \
				self.train_std[np.newaxis, :]

		else:

			normalized = (outcomes - self.train_mean[np.newaxis, :]) / \
				self.train_std[np.newaxis, :]
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

		fns, xi, xs, y = self._split_images_and_outcomes(s)

		if normalize_outcomes:

			y = self._normalize_outcomes(y)

		if imgfmt == 'name':

			return fns, xi, xs, y

		elif imgfmt == 'array':

			return np.squeeze(images_from_files(
				fns, (224, 224), normalize=normalize_images)), xi, xs, y


	def _read_subject_data(self, d):

		pid = int(d.split('_')[1].strip('{}'))

		# if pid in [3064]:
		# 	return None

		data_subdir = os.path.join(self.datadir, d)

		assert os.path.isdir(data_subdir), 'Not a directory: %s' % data_subdir

		print('Reading', data_subdir)

		ema_random = pd.read_excel(
			os.path.join(
				data_subdir,
				'%i_EMA_data' % pid,
				'Random.xlsx'),
			parse_dates=[['Survey Submitted Date', 'Survey Submitted Time']],
			dayfirst=True)

		ema_smoking = pd.read_excel(
			os.path.join(
				data_subdir,
				'%i_EMA_data' % pid,
				'Smoking.xlsx'),
			parse_dates=[['Survey Submitted Date', 'Survey Submitted Time']],
			dayfirst=True)

		date_col = 'Survey Submitted Date_Survey Submitted Time'

		ema_random['Day'] = (ema_random[date_col] - ema_random[date_col][0]).dt.days.astype(int)
		ema_smoking['Day'] = (ema_smoking[date_col] - ema_smoking[date_col][0]).dt.days.astype(int)

		ema_smoking = ema_smoking.rename(
			columns={'How long ago did you use a tobacco product?': \
				'When did you last use a tobacco product?'})

		craving_col = 'When you decided to use the tobacco product, how strong was your urge to use the tobacco product?'

		ema_random.loc[ema_random[craving_col] == 'CONDITION_SKIPPED', craving_col] = \
			ema_random['How strong is your current urge to smoke a cigarette?'][ema_random[craving_col] == 'CONDITION_SKIPPED']		

		ema_random['imagedir'] = os.path.join(
			data_subdir, 'Random_folder')

		ema_smoking['imagedir'] = os.path.join(
			data_subdir, 'Smoking_folder')

		df_random = self._get_outcomes(
			ema_random,
			dichotomize=self.dichotomize)

		df_smoking = self._get_outcomes(
			ema_smoking,
			dichotomize=self.dichotomize)

		df_random['prompt'] = 'random'
		df_smoking['prompt'] = 'smoking'

		df_random['Day'] = ema_random['Day']
		df_smoking['Day'] = ema_smoking['Day']

		df_random['datetime'] = ema_random[date_col]
		df_smoking['datetime'] = ema_smoking[date_col]

		df_random['filename'] = ema_random['imagedir'].str.cat(
			ema_random[const.IMAGECOL].str.split('/').str[-1],
			sep='/')

		df_smoking['filename'] = ema_smoking['imagedir'].str.cat(
			ema_smoking[const.IMAGECOL].str.split('/').str[-1],
			sep='/')

		df = pd.concat([df_random, df_smoking], axis=0)

		df['weekend'] = (df['datetime'].dt.weekday > 4).astype(float)
		df['part_of_day'] = df['datetime'].dt.hour // 6
		df['night'] = (df['part_of_day'] == 0).astype(float)
		df['morning'] = (df['part_of_day'] == 1).astype(float)
		df['afternoon'] = (df['part_of_day'] == 2).astype(float)
		df['evening'] = (df['part_of_day'] == 3).astype(float)

		df = df.drop(['datetime', 'part_of_day'], axis=1)

		df['pid'] = pid

		return df.set_index(['pid', 'prompt', 'Day', df.index])


	def _validate_data(self, df):

		exists = df['filename'].apply(lambda x: os.path.exists(str(x)))

		print('The following files were not found:')
		print(df['filename'][~exists])

		df['filename'][~exists].to_csv('../results/not_found.csv')

		return df[exists]


	def _split_images_and_outcomes(self, df):

		filenames = df[['filename']].values
		image_features = df[self.image_feature_cols].values
		subject_features = df[self.subject_feature_cols].values
		outcomes = df[const.OUTCOMES].values

		return filenames, image_features, subject_features, outcomes


	def _get_outcomes(self, df, dichotomize=None):
		return pd.DataFrame(
			{o: const.score_outcome(
				df, o, dichotomize=dichotomize) for o in const.OUTCOMES},
			index=df.index)


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

