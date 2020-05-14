import pandas as pd
import os

BASELINE_DIR = '/Users/mme/data/deeplearning'

BASELINE_FN = os.path.join(
	BASELINE_DIR,
	'DeepLearning-SDSS_DATA_LABELS_2020-05-08_0941.csv'
)

BEDROOM_COLS = [('%ib. Do you smoke in this room?' % (i + 1)) for i in range(5)]

BATHROOM_COLS = [('%ib. Do you smoke in this room?' % (i + 6)) for i in range(3)]

OTHER_ROOMS = [('%ib. Do you smoke in this room?' % (i + 9)) for i in range(12)]

OTHER_PLACES = [
	'1. Open (non-shelter) Bus Stop',
	'2. Sheltered Bus Stop',
	'3. Outside Central Bus Terminal',
	'4. Train Station',
	'5. Outside Restaurants',
	'6. Outside Bars',
	'7. Outside Library',
	'8. Outside School',
	'9. Outside Religious Venue',
	'10. Sidewalks',
	'11. Greenways/Trails',
	'12. Park',	
	'Do you smoke while you are at work?'
]

KEY = {'No': 0, 'Yes': 1}


def main():

	df = pd.read_csv(BASELINE_FN)

	frame = (df[OTHER_ROOMS + OTHER_PLACES] == 'Yes').astype(int).fillna(0)

	frame['bedroom'] = (df[BEDROOM_COLS] == 'Yes').any(axis=1).astype(int).fillna(0)
	frame['bathroom'] = (df[BATHROOM_COLS] == 'Yes').any(axis=1).astype(int).fillna(0)

	frame['pid'] = df['Subject ID']

	frame.set_index('pid').to_csv(os.path.join(
		BASELINE_DIR,
		'baseline_data.csv'))


if __name__ == '__main__':
	main()