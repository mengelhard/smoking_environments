import pandas as pd
import numpy as np
import os

import constants as const

PLOT_OUTCOMES = True


def main():

	sdss = get_sdss_data()
	dem = get_dem_data()

	subject_data_fn = os.path.join(
		const.DATA_DIR,
		'deeplearning',
		'prepdata',
		'subject_data.csv')

	subject_data = dem.join(sdss)
	subject_data.to_csv(subject_data_fn)

	ema_data_fn = os.path.join(
		const.DATA_DIR,
		'deeplearning',
		'prepdata',
		'ema_data.csv')

	ema = get_ema_data()
	ema.to_csv(ema_data_fn)

	all_data_fn = os.path.join(
		const.DATA_DIR,
		'deeplearning',
		'prepdata',
		'all_data.csv')

	all_data = ema.join(subject_data, how='left')
	all_data.to_csv(all_data_fn)

	if PLOT_OUTCOMES:

		import matplotlib.pyplot as plt

		for o in const.OUTCOMES:

			plt.hist(all_data[o].astype(float))
			plt.savefig('../results/%s_hist.png' % o)
			plt.close()


def get_sdss_data():

	sdss_fn = os.path.join(
		const.DATA_DIR,
		'deeplearning',
		'rawdata',
		'DeepLearning-SDSS_DATA_LABELS_2020-05-08_0941.csv'
	)

	q_str = 'b. Do you smoke in this room?'

	bedroom_cols = [('%i%s' % (i + 1, q_str)) for i in range(5)]
	bathroom_cols = [('%i%s' % (i + 6, q_str)) for i in range(3)]
	other_rooms = {('%i%s' % (i + 9, q_str)): ('room_%i' % i) for i in range(12)}

	other_places = {
		'1. Open (non-shelter) Bus Stop': 'open_bus_stop',
		'2. Sheltered Bus Stop': 'sheltered_bus_stop',
		'3. Outside Central Bus Terminal': 'bus_terminal',
		'4. Train Station': 'train_station',
		'5. Outside Restaurants': 'restaurant',
		'6. Outside Bars': 'bar',
		'7. Outside Library': 'library',
		'8. Outside School': 'school',
		'9. Outside Religious Venue': 'church',
		'10. Sidewalks': 'sidewalk',
		'11. Greenways/Trails': 'trail',
		'12. Park': 'park',	
		'Do you smoke while you are at work?': 'work'
	}

	df = pd.read_csv(sdss_fn)

	frame = df[['Subject ID']].rename({'Subject ID': 'pid'}, axis=1)

	frame['bedroom'] = (df[bedroom_cols] == 'Yes').any(
		axis=1).astype(int).fillna(0)
	frame['bathroom'] = (df[bathroom_cols] == 'Yes').any(
		axis=1).astype(int).fillna(0)

	for k, v in {**other_rooms, **other_places}.items():
		frame[v] = (df[k] == 'Yes').astype(int).fillna(0)

	return frame.set_index('pid')


def get_dem_data():

	df = pd.read_excel(os.path.join(
		const.DATA_DIR,
		'deeplearning',
		'rawdata',
		'DL Demographics.xlsx'))

	df['race'] = ((df['Race'] == 'White') & (df['Ethnicity'] != 'Hispanic')).astype(float)
	df['sex'] = (df['Gender'] == 'Female').astype(float)
	df['pid'] = df['Subject ID assigned']

	return df[['pid', 'sex', 'race']].set_index('pid')


def get_ema_data():

	ema_dir = os.path.join(
		const.DATA_DIR,
		'deeplearning',
		'EMA data and Photos')

	folders = [f for f in os.listdir(ema_dir) if os.path.isdir(
		os.path.join(ema_dir, f))]
	
	data = [read_subject_data(d) for d in folders]

	return pd.concat(data, axis=0)


def read_subject_data(d):

	pid = int(d.split('_')[1].strip('{}'))

	subject_ema_dir = os.path.join(
		const.DATA_DIR,
		'deeplearning',
		'EMA data and Photos',
		d)

	assert os.path.isdir(subject_ema_dir), 'Not found: %s' % subject_ema_dir

	print('Reading', subject_ema_dir)

	ema_random = pd.read_excel(
		os.path.join(
			subject_ema_dir,
			'%i_EMA_data' % pid,
			'Random.xlsx'),
		parse_dates=[['Survey Submitted Date', 'Survey Submitted Time']],
		dayfirst=True)

	ema_smoking = pd.read_excel(
		os.path.join(
			subject_ema_dir,
			'%i_EMA_data' % pid,
			'Smoking.xlsx'),
		parse_dates=[['Survey Submitted Date', 'Survey Submitted Time']],
		dayfirst=True)

	datecol = {'Survey Submitted Date_Survey Submitted Time': 'datetime'}

	ema_random = ema_random.rename(datecol, axis=1)
	ema_smoking = ema_smoking.rename(datecol, axis=1)

	ema_smoking = ema_smoking.rename(
		columns={'How long ago did you use a tobacco product?': \
			'When did you last use a tobacco product?'})

	craving_col = 'When you decided to use the tobacco product, how strong was your urge to use the tobacco product?'

	ema_random.loc[ema_random[craving_col] == 'CONDITION_SKIPPED', craving_col] = \
		ema_random['How strong is your current urge to smoke a cigarette?'][ema_random[craving_col] == 'CONDITION_SKIPPED']		

	ema_random['imagedir'] = os.path.join(
		subject_ema_dir.split('/')[-1], 'Random_folder')

	ema_smoking['imagedir'] = os.path.join(
		subject_ema_dir.split('/')[-1], 'Smoking_folder')

	df_random = get_outcomes(ema_random)
	df_smoking = get_outcomes(ema_smoking)

	df_random['prompt'] = 'random'
	df_smoking['prompt'] = 'smoking'

	df_random['datetime'] = ema_random['datetime']
	df_smoking['datetime'] = ema_smoking['datetime']

	df_random['filename'] = ema_random['imagedir'].str.cat(
		ema_random[const.IMAGECOL].str.split('/').str[-1],
		sep='/')

	df_smoking['filename'] = ema_smoking['imagedir'].str.cat(
		ema_smoking[const.IMAGECOL].str.split('/').str[-1],
		sep='/')

	df_random['pid'] = pid
	df_smoking['pid'] = pid

	df_random.index.name = 'index'
	df_smoking.index.name = 'index'

	df_random = df_random.set_index(['pid', 'prompt', df_random.index])
	df_smoking = df_smoking.set_index(['pid', 'prompt', df_smoking.index])

	df = pd.concat([df_random, df_smoking], axis=0).sort_values('datetime')

	df['day'] = (df['datetime'] - df['datetime'][0]).dt.days.astype(int)

	df['weekend'] = (df['datetime'].dt.weekday > 4).astype(float)
	df['part_of_day'] = df['datetime'].dt.hour // 6
	df['night'] = (df['part_of_day'] == 0).astype(float)
	df['morning'] = (df['part_of_day'] == 1).astype(float)
	df['afternoon'] = (df['part_of_day'] == 2).astype(float)
	df['evening'] = (df['part_of_day'] == 3).astype(float)

	#df['minutes_since_last_prompt'] = df['datetime'].diff().dt.total_seconds() / 60
	df['last_smoke_time'] = df['datetime'].where(df['smoking'] == 1).ffill().shift(1)
	df['minutes_since_last_smoke'] = (df['datetime'] - df['last_smoke_time']).dt.total_seconds() / 60

	return df.drop(['part_of_day', 'last_smoke_time', 'datetime'], axis=1)


def get_outcomes(df):
	return pd.DataFrame(
		{o: const.score_outcome(df, o) for o in const.OUTCOMES},
		index=df.index)


if __name__ == '__main__':
	main()
