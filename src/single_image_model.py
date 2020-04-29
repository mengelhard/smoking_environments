import numpy as np
import tensorflow as tf
import sys
import os
import datetime
from sklearn.metrics import roc_auc_score 

import constants as const

for p in const.MODELS_PATHS:
	if os.path.exists(p):
		sys.path.append(p)

from nets.mobilenet import mobilenet_v2

for f in const.CHECKPOINT_FILE_PATHS:
	if os.path.exists(f):
		CHECKPOINT_FILE = f + '/mobilenet_v2_1.0_224.ckpt'

NUM_TUNING_RUNS = 20
NUM_ROWS_PER_DATAFILE = None


def main():

	from data_loader import DataLoader
	from results_writer import ResultsWriter

	hyperparam_options = {
		'n_hidden_layers': [1],
		'hidden_layer_sizes': np.arange(50, 300),
		'learning_rate': np.logspace(-4., -6.5),
		'activation_fn': [tf.nn.relu, tf.nn.sigmoid],#[tf.nn.relu, tf.nn.tanh],
		'dropout_pct': [0, .1, .3, .5],
		'train_mobilenet': [True],#, False],
		'mobilenet_endpoint': ['global_pool'],#['global_pool', 'Logits'],
		'max_epochs_no_improve': np.arange(2),
		'batch_size': [100],
		'dichotomize': [None]
	}

	resultcols = ['status']
	resultcols += ['fold']
	resultcols += [('mse_%s' % o) for o in const.OUTCOMES]
	resultcols += [('r2_%s' % o) for o in const.OUTCOMES]
	resultcols += [('auc_%s' % o) for o in const.OUTCOMES]
	resultcols += list(hyperparam_options.keys())

	rw = ResultsWriter(resultcols)
	results_list = []

	# tuning runs

	for i in range(NUM_TUNING_RUNS):

		hyperparams = select_hyperparams(hyperparam_options)

		print('Running with the following hyperparams:')
		print(hyperparams)

		for val_fold in range(1):

			print('Training with val_fold =', val_fold)

			tf.compat.v1.reset_default_graph()
			dl = DataLoader(
				val_fold=val_fold,
				nrows=NUM_ROWS_PER_DATAFILE,
				single_image=True,
				**hyperparams)
			mdl = SingleImageModel(dl, **hyperparams)

			fold_losses = []

			#try:

			with tf.compat.v1.Session() as s:

				train_stats, val_stats = mdl.train(s, **hyperparams)
				y_pred, y, avg_val_loss = mdl.predict(s, 'val', hyperparams['batch_size'])

			results_dict = get_results(y, y_pred, hyperparams['dichotomize'])

			fold_losses.append(avg_val_loss)

			rw.write(i, {
				'status': 'complete',
				'fold': val_fold,
				**results_dict,
				**hyperparams})

			rw.plot(
				'%i_%i' % (i, val_fold),
				train_stats,
				val_stats,
				y_pred,
				y,
				const.OUTCOMES,
				results_dict,
				const.VARTYPES,
				**hyperparams)

			# except:

			# 	rw.write(i, {'status': 'failed', **hyperparams})

		if len(fold_losses) > 0:
			results_list.append((hyperparams, np.mean(fold_losses)))

	# choose final hyperparameters

	hps, results = list(zip(*results_list))
	hyperparams = hps[np.argmin(results)]

	# final run

	tf.compat.v1.reset_default_graph()
	dl = DataLoader(
		val_fold=3,
		nrows=NUM_ROWS_PER_DATAFILE,
		single_image=True,
		**hyperparams)
	mdl = SingleImageModel(dl, **hyperparams)

	try:

		with tf.compat.v1.Session() as s:

			train_stats, val_stats = mdl.train(s, **hyperparams)
			y_pred, y, avg_loss = mdl.predict(s, 'test', hyperparams['batch_size'])

		results_dict = get_results(y, y_pred, hyperparams['dichotomize'])

		rw.write('final', {
			'status': 'complete',
			'fold': 4,
			**results_dict,
			**hyperparams})

		rw.plot(
			'final',
			train_stats,
			val_stats,
			y_pred,
			y,
			const.OUTCOMES,
			results_dict,
			const.VARTYPES,
			**hyperparams)

	except:

		rw.write('final', {'status': 'failed', **hyperparams})


def get_results(y_true, y_pred, dichotomize=None):

	if dichotomize == True:
		var_types = {o: 'categorical' for o in const.OUTCOMES}

	elif dichotomize == False:
		var_types = {o: 'numeric' for o in const.OUTCOMES}

	else:
		var_types = const.VARTYPES

	result_dict = {}

	for o, yt, yp in zip(const.OUTCOMES, y_true.T, y_pred.T):

		if var_types[o] == 'categorical':
			result_dict[('auc_%s' % o)] = roc_auc_score(yt, yp)

		elif var_types[o] == 'numeric':
			result_dict[('mse_%s' % o)] = np.mean((yp - yt) ** 2)
			result_dict[('r2_%s' % o)] = np.corrcoef(yp, yt)[0][1] ** 2

	return result_dict


class SingleImageModel:

	def __init__(
		self, dataloader,
		n_hidden_layers=0,
		hidden_layer_sizes=50,
		learning_rate=1e-3,
		activation_fn=tf.nn.relu,
		dropout_pct=.5,
		train_mobilenet=False,
		mobilenet_endpoint='global_pool',
		**kwargs):

		self.dataloader = dataloader

		self.n_out = dataloader.n_out

		self.hidden_layer_sizes = [hidden_layer_sizes] * n_hidden_layers

		self.learning_rate = learning_rate
		self.activation_fn = activation_fn
		self.dropout_pct = dropout_pct

		self.train_mobilenet = train_mobilenet

		self.mobilenet_endpoint = mobilenet_endpoint

		self._build_placeholders()
		self._build_mobilenet()
		self._build_model()
		self._build_train_step()


	def train(
		self, sess,
		max_epochs=30,
		max_epochs_no_improve=2,
		batch_size=100,
		**kwargs):

		sess.run(tf.compat.v1.global_variables_initializer())
		self.mobilenet_saver.restore(sess, CHECKPOINT_FILE)

		batches_per_epoch = int(np.ceil(
			self.dataloader.n_train / batch_size))

		train_stats = []
		val_stats = []

		# initial validation batch

		val_batch_sizes, val_loss = self._run_batches(
			sess,
			[self.loss],
			'val',
			batch_size,
			train=False)

		avg_val_loss = np.sum(val_batch_sizes * val_loss) / np.sum(val_batch_sizes)
		val_stats.append((0, avg_val_loss))

		print('Initial val loss: %.2f' % val_stats[-1][1])

		# initialize early stopping conditions

		best_val_loss = avg_val_loss
		n_epochs_no_improve = 0

		for epoch_idx in range(max_epochs):

			# training batches

			train_batch_sizes, train_loss = self._run_batches(
				sess,
				[self.loss],
				'train',
				batch_size,
				train=True)

			train_indices = np.arange(len(train_loss)) + epoch_idx * batches_per_epoch
			train_stats.extend(list(zip(train_indices, train_loss)))

			# validation batches

			val_batch_sizes, val_loss = self._run_batches(
				sess,
				[self.loss],
				'val',
				batch_size,
				train=False)

			train_idx = (epoch_idx + 1) * batches_per_epoch
			avg_val_loss = np.sum(val_batch_sizes * val_loss) / np.sum(val_batch_sizes)
			val_stats.append((train_idx, avg_val_loss))

			print('Completed Epoch %i' % epoch_idx)
			print('Val loss: %.2f, Train loss: %.2f' % (avg_val_loss, np.mean(train_loss)))

			# check for early stopping

			if avg_val_loss < best_val_loss:
				best_val_loss = avg_val_loss
				n_epochs_no_improve = 0
			else:
				n_epochs_no_improve += 1

			if n_epochs_no_improve > max_epochs_no_improve:
				break

		return train_stats, val_stats


	def predict(self, sess, part, batch_size):

		assert part in ['all', 'train', 'val', 'test']

		batch_sizes, y_pred, y_prob_pred, y, loss = self._run_batches(
			sess,
			[self.y_pred, self.y_prob_pred, self.y, self.loss],
			part,
			batch_size,
			train=False)

		avg_loss = (loss * batch_sizes) / np.sum(batch_sizes)

		if self.dataloader.dichotomize == True:

			return y_prob_pred, y, avg_loss

		elif self.dataloader.dichotomize == False:

			return y_pred, y, avg_loss

		else:

			cat_cols = self.dataloader.is_categorical
			y_pred[:, cat_cols] = y_prob_pred[:, cat_cols]

			return y_pred, y, avg_loss


	def _run_batches(self, sess, tensors, part, batch_size, train=False):

		results = [[] for t in tensors]
		batch_sizes = []

		for batch_idx, (xb, yb) in enumerate(self.dataloader.get_batch(
			part, batch_size)):

			print('Starting %s batch %i' % (part, batch_idx))

			if train:

				results_ = sess.run(
					tensors + [self.train_step],
					feed_dict={self.x: xb, self.y: yb, self.is_training: True})

			else:

				results_ = sess.run(
					tensors,
					feed_dict={self.x: xb, self.y: yb, self.is_training: False})

			batch_sizes.append(len(xb))

			for i in range(len(tensors)):
				results[i].append(results_[i])

		return (np.array(batch_sizes), ) + tuple(try_concat(r, axis=0) for r in results)


	def _build_placeholders(self):

		self.x = tf.compat.v1.placeholder(
			dtype=tf.float32,
			shape=(None, 224, 224, 3))

		self.y = tf.compat.v1.placeholder(
			dtype=tf.float32,
			shape=(None, self.n_out))

		self.is_training = tf.compat.v1.placeholder(
			dtype=tf.bool,
			shape=())


	def _build_mobilenet(self):

		if self.train_mobilenet:
			is_training = self.is_training
		else:
			is_training = False

		with tf.contrib.slim.arg_scope(
			mobilenet_v2.training_scope(is_training=is_training)):
			
			logits, endpoints = mobilenet_v2.mobilenet(self.x)

		ema = tf.train.ExponentialMovingAverage(0.999)
		self.mobilenet_saver = tf.compat.v1.train.Saver(
			ema.variables_to_restore())

		features_flat = endpoints[self.mobilenet_endpoint]

		if self.mobilenet_endpoint == 'global_pool':
			features_flat = tf.squeeze(features_flat, [1, 2])

		self.image_features = features_flat


	def _build_model(self):

		### NOTE: mobilenet v2 says logits should be LINEAR from here

		with tf.compat.v1.variable_scope('outcomes'):

			with tf.compat.v1.variable_scope('mlp'):

				hidden_layer = mlp(
					self.image_features,
					self.hidden_layer_sizes,
					dropout_pct=self.dropout_pct,
					activation_fn=self.activation_fn,
					training=self.is_training)

			with tf.compat.v1.variable_scope('linear'):

				self.y_pred = mlp(
					hidden_layer,
					[self.n_out],
					dropout_pct=0.,
					activation_fn=None,
					training=self.is_training)

			self.y_prob_pred = tf.nn.sigmoid(self.y_pred)


	def _build_train_step(self):

		if self.dataloader.dichotomize == True:

			self.loss = tf.compat.v1.losses.sigmoid_cross_entropy(
				multi_class_labels=self.y,
				logits=self.y_pred)

		elif self.dataloader.dichotomize == False:

			self.loss = tf.compat.v1.losses.mean_squared_error(
				self.y,
				self.y_pred)

		else:

			self.loss_mse = tf.compat.v1.losses.mean_squared_error(
				self.y,
				self.y_pred,
				reduction='none')

			self.loss_ce = tf.compat.v1.losses.sigmoid_cross_entropy(
				multi_class_labels=self.y,
				logits=self.y_pred,
				reduction='none')

			mse_mask = (~self.dataloader.is_categorical).astype(float)[np.newaxis, :]
			ce_mask = (self.dataloader.is_categorical).astype(float)[np.newaxis, :]

			self.loss = tf.reduce_mean(
				mse_mask * self.loss_mse + ce_mask * self.loss_ce)

		if self.train_mobilenet:

			self.train_step = tf.compat.v1.train.AdamOptimizer(
				self.learning_rate).minimize(self.loss)

		else:

			myvars = tf.compat.v1.get_collection(
				tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
				scope='image_features')

			myvars += tf.compat.v1.get_collection(
				tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
				scope='outcomes')

			self.train_step = tf.compat.v1.train.AdamOptimizer(
				self.learning_rate).minimize(self.loss, var_list=myvars)


def mlp(x, hidden_layer_sizes,
		dropout_pct=0.,
		activation_fn=tf.nn.relu,
		training=True,
		reuse=False):

	hidden_layer = x

	with tf.compat.v1.variable_scope('mlp', reuse=reuse):

		for i, layer_size in enumerate(hidden_layer_sizes):

			hidden_layer = tf.layers.dense(
				hidden_layer, layer_size,
				activation=activation_fn,
				name='fc_%i' % i,
				reuse=reuse)

			if dropout_pct > 0:
				hidden_layer = tf.layers.dropout(
					hidden_layer, rate=dropout_pct,
					training=training,
					name='dropout_%i' % i)

	return hidden_layer


def select_hyperparams(hpdict):

	return {k: v[np.random.randint(len(v))] for k, v in hpdict.items()}


def try_concat(arr, axis=0):

	try:

		return np.concatenate(arr, axis=axis)

	except:

		return np.array(arr)


if __name__ == '__main__':
	main()
