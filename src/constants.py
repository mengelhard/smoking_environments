import numpy as np
import pandas as pd
import sys, os

'''contains constants and scoring for deeplearning data'''


DATA_DIRS = ['/Users/mme/data', '/scratch/mme4']

MODELS_PATHS = [
	'/Users/mme/projects/models/research/slim',
	'/scratch/mme4/models/research/slim'
]

CHECKPOINT_FILE_PATHS = [
	'/Users/mme/projects/imageturk/mobilenet_checkpoint',
	'/scratch/mme4/mobilenet_checkpoint'
]

MOBILENET_OUTPUT_SIZE = {
	'global_pool': 1280,
	'Logits': 1001
}

IMAGECOL = 'Take a photo of your current environment'

# OUTCOMES = [
# 	'smoking',
# 	'craving',
# 	'smoking_allowed',
# 	'outside'
# ]

OUTCOMES = [
	'smoking',
	'smoking_allowed',
	'outside'
]

VARTYPES = {
	'smoking': 'categorical',
	'craving': 'numeric',
	'smoking_allowed': 'categorical',
	'outside': 'categorical'
}

CUTOFFS = {# DIVIDE BY >CUTOFF
	'smoking': 0,
	'craving': 0,
	'smoking_allowed': 0,
	'outside': 0
}

ITEM = {
	'smoking': 'When did you last use a tobacco product?',
	'craving': 'When you decided to use the tobacco product, how strong was your urge to use the tobacco product?',
	'smoking_allowed': 'Is smoking allowed in your present location?',
	'outside': 'Are you inside or outside?'
}

SMOKING_SCALE = {
	'Smoking/vaping/dipping right now': 1,
	'Less than 10 minutes ago': 0,
	'11-30 minutes ago': 0,
	'31-60 minutes ago': 0,
	'61-120 minutes ago': 0,
	'Greater than 120 minutes ago': 0,
	'CONDITION_SKIPPED': np.nan
}

CRAVING_SCALE = {
	'Very slightly or not at all': 0,
	'A little': 1,
	'Moderately': 2,
	'Moderate': 2,
	'Quite a bit': 3,
	'Extremely': 4,
	'Extreme': 4,
	'CONDITION_SKIPPED': np.nan
}

SCALES = {
	'smoking': SMOKING_SCALE,
	'craving': CRAVING_SCALE,
	'smoking_allowed': {'No': 0, 'Yes': 1, 'CONDITION_SKIPPED': np.nan},
	'outside': {'Inside': 0, 'Outside': 1, 'CONDITION_SKIPPED': np.nan}
}


def score_outcome(df, outcome, dichotomize=None):

	s = df[ITEM[outcome]].apply(lambda x: getv(SCALES[outcome], x))

	if dichotomize == True:

		return (s > CUTOFFS[outcome]).astype(float)

	elif dichotomize == False:

		return s

	elif VARTYPES[outcome] == 'categorical':

		return (s > CUTOFFS[outcome]).astype(float)

	elif VARTYPES[outcome] == 'numeric':

		return s

	else:

		assert False, 'Could not determine variable type for %s' % outcome


def getv(d, key):

	try:

		return d[key]

	except:

		print('Key not found:', key)
		return np.nan
