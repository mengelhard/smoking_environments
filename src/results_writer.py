import numpy as np
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def main():

	rw = ResultsWriter(['param1', 'param2'])

	for i in range(2):

		rw.write(i, {'param1': 11, 'param2': 22})

		train_stats = [(0, 55), (1, 44), (2, 33), (3, 22), (4, 11)]
		val_stats = [(0, 66), (4, 0)]

		y_pred = np.array([[1, 2], [3, 4], [5, 6], [2, 3], [4, 5]])
		y = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [1, 2]])

		outcomes = ['outcome1', 'outcome2']

		score = np.mean((y - y_pred) ** 2, axis=0)

		rw.plot(i, train_stats, val_stats, y_pred, y, outcomes, score)


class ResultsWriter:

	def __init__(self, resultcols):

		self.utc = datetime.datetime.utcnow().strftime('%s')

		self.resultcols = resultcols
		
		self.results_dir = os.path.join(
			os.path.abspath('..'),
			'results/results_' + self.utc)

		os.mkdir(self.results_dir)
		
		self.results_fn = os.path.join(self.results_dir, 'results.csv')

		with open(self.results_fn, 'w+') as results_file:
			print(', '.join(['result_idx'] + self.resultcols), file=results_file)


	def write(self, result_idx, resultsdict):

		values = [str(result_idx)]
		values += [str(resultsdict.get(col, np.nan)) for col in self.resultcols]

		with open(self.results_fn, 'a') as results_file:
			print(', '.join(values), file=results_file)


	def plot(
		self, result_idx, train_stats, val_stats, y_pred, y, outcomes, results_dict,
		var_types, dichotomize=None, **kwargs):

		if dichotomize == True:
			var_types = {o: 'categorical' for o in outcomes}

		elif dichotomize == False:
			var_types = {o: 'numeric' for o in outcomes}

		fig, ax = plt.subplots(
			nrows=len(outcomes) + 1, ncols=1,
			figsize=(5, 5 * (len(outcomes) + 1)))

		ax[0].plot(*list(zip(*train_stats)), label='train')
		ax[0].plot(*list(zip(*val_stats)), label='val')
		ax[0].set_title('Training Plot')
		ax[0].set_xlabel('Iteration')
		ax[0].set_ylabel('Loss')
		ax[0].legend()

		for i, o in enumerate(outcomes):

			if var_types[o] == 'categorical':

				auc = results_dict[('auc_%s' % o)]

				fpr, tpr, _ = roc_curve(y[:, i], y_pred[:, i])
				ax[i + 1].plot(fpr, tpr)
				ax[i + 1].set_title(outcomes[i] + ('(AUC=%.2f)' % auc))
				ax[i + 1].set_xlabel('False Positive Rate')
				ax[i + 1].set_ylabel('True Positive Rate')

			else:

				mse = results_dict[('mse_%s' % o)]
				r2 = results_dict[('r2_%s' % o)]

				ax[i + 1].scatter(y[:, i], y_pred[:, i])
				ax[i + 1].set_title(outcomes[i] + ('(MSE=%.2f)' % mse) + ('(R2=%.2f)' % r2))
				ax[i + 1].set_xlabel('y_true')
				ax[i + 1].set_ylabel('y_pred')

		plt.tight_layout()
		plt.savefig(os.path.join(
			self.results_dir, 'results_' + str(result_idx) + '.png'))
		plt.close()


if __name__ == '__main__':
	main()


