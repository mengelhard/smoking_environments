import numpy as np
import tensorflow as tf
import sys
import os
import datetime
from sklearn.metrics import roc_auc_score
import pickle

import constants as const

for p in const.MODELS_PATHS:
	if os.path.exists(p):
		sys.path.append(p)

from nets.mobilenet import mobilenet_v2

for f in const.CHECKPOINT_FILE_PATHS:
	if os.path.exists(f):
		CHECKPOINT_FILE = f + '/mobilenet_v2_1.0_224.ckpt'

NUM_TUNING_RUNS = 30
NUM_ROWS_PER_DATAFILE = None
WRITE_BOTTLENECKS = True
SAVE_AS_TFLITE = True
USE_FEATURES = True


def main():

	from data_loader import DataLoader
	from results_writer import ResultsWriter

	OUTCOMES = ['smoking', 'craving_binary', 'smoking_allowed', 'outside']

	hyperparam_options = {
		'n_hidden_layers': [0, 1],
		'hidden_layer_sizes': np.arange(50, 1000),
		'learning_rate': np.logspace(-3., -6.),
		'activation_fn': [tf.nn.relu, tf.sigmoid, tf.elu],#[tf.nn.relu, tf.nn.tanh],
		'dropout_pct': [0, .1, .3, .5],
		'train_mobilenet': [True],#, False],
		'mobilenet_endpoint': ['global_pool'],#['global_pool', 'Logits'],
		'max_epochs_no_improve': np.arange(2),
		'batch_size': [50],
	}

	resultcols = ['status']
	resultcols += [('auc_%s' % o) for o in OUTCOMES]
	resultcols += list(hyperparam_options.keys())

	rw = ResultsWriter(resultcols)
	results_list = []

	# tuning runs

	for i in range(NUM_TUNING_RUNS):

		hyperparams = select_hyperparams(hyperparam_options)

		print('Running with the following hyperparams:')
		print(hyperparams)

		tf.compat.v1.reset_default_graph()
		dl = DataLoader(
			partition_method='longitudinal',
			outcomes=OUTCOMES,
			pid_features=False,
			**hyperparams)
		mdl = SmokingNet(dl, use_features=USE_FEATURES, **hyperparams)

		fold_losses = []

		#try:

		with tf.compat.v1.Session() as s:

			train_stats, val_stats = mdl.train(s, **hyperparams)
			y_pred, y, avg_val_loss, _ = mdl.predict(s, 'val', hyperparams['batch_size'])

		results_dict = get_results(y, y_pred, OUTCOMES)

		fold_losses.append(avg_val_loss)

		rw.write(i, {
			'status': 'complete',
			**results_dict,
			**hyperparams})

		rw.plot(
			'%i' % i,
			train_stats,
			val_stats,
			y_pred,
			y,
			OUTCOMES,
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
		partition_method='longitudinal',
		outcomes=OUTCOMES,
		pid_features=False,
		**hyperparams)
	mdl = SmokingNet(dl, use_features=USE_FEATURES, **hyperparams)

	with tf.compat.v1.Session() as s:

		train_stats, val_stats = mdl.train(s, **hyperparams)
		y_pred, y, avg_loss, _ = mdl.predict(
			s, 'test', hyperparams['batch_size'])

		if WRITE_BOTTLENECKS:

			y_pred_all, y_all, _, image_features = mdl.predict(
				s, 'all', hyperparams['batch_size'])

		if SAVE_AS_TFLITE: # save model as tflite

			if USE_FEATURES:
				in_tensors = [mdl.xi, mdl.xf, mdl.is_training]
			else:
				in_tensors = [mdl.xi, mdl.xf]
			
			out_tensors = [mdl.y]

			try:

				converter = tf.lite.TFLiteConverter.from_session(
					s, in_tensors, out_tensors)
				tflite_model = converter.convert()
				open('../saved_models/quiteye.tflite', 'wb').write(
					tflite_model)

			except:
				pass

	if WRITE_BOTTLENECKS:

		df_all = dl.data['all']
		df_all['image_features'] = list(image_features)

		for i, out in enumerate(OUTCOMES):

			pred_name = out + '_predicted'
			df_all[pred_name] = y_pred_all[:, i]

		df_all.drop('image_features', axis=1).to_csv(
			os.path.join(rw.results_dir, 'predictions.csv'))

		data_dict = dl.data['all'].reset_index().to_dict(orient='index')

		with open(os.path.join(rw.results_dir, 'ddict.pickle'), 'wb') as handle:
			pickle.dump(data_dict, handle)

	results_dict = get_results(y, y_pred, OUTCOMES)

	rw.write('final', {
		'status': 'complete',
		**results_dict,
		**hyperparams})

	rw.plot(
		'final',
		train_stats,
		val_stats,
		y_pred,
		y,
		OUTCOMES,
		results_dict,
		const.VARTYPES,
		**hyperparams)

	# except:

	# 	rw.write('final', {'status': 'failed', **hyperparams})


def get_results(y_true, y_pred, outcomes):

	var_types = const.VARTYPES

	result_dict = {}

	for o, yt, yp in zip(outcomes, y_true.T, y_pred.T):

		if var_types[o] == 'categorical':
			result_dict[('auc_%s' % o)] = roc_auc_score(yt, yp)

		elif var_types[o] == 'numeric':
			result_dict[('mse_%s' % o)] = np.mean((yp - yt) ** 2)
			result_dict[('r2_%s' % o)] = np.corrcoef(yp, yt)[0][1] ** 2

	return result_dict


class SmokingNet:

	def __init__(
		self, dataloader,
		n_hidden_layers=0,
		hidden_layer_sizes=50,
		learning_rate=1e-3,
		logit_learning_rate=1e-3,
		activation_fn=tf.nn.relu,
		dropout_pct=.5,
		train_mobilenet=False,
		use_features=True,
		mobilenet_endpoint='global_pool',
		**kwargs):

		self.dataloader = dataloader

		self.n_out = dataloader.n_out
		self.n_features = dataloader.n_features

		self.hidden_layer_sizes = [hidden_layer_sizes] * n_hidden_layers

		self.learning_rate = learning_rate
		self.logit_learning_rate = logit_learning_rate
		self.activation_fn = activation_fn
		self.dropout_pct = dropout_pct

		self.train_mobilenet = train_mobilenet
		self.use_features

		self.mobilenet_endpoint = mobilenet_endpoint

		self._build_placeholders()
		self._build_mobilenet()
		self._build_model()
		self._build_train_step()


	def train(
		self, sess,
		max_epochs=100,
		max_epochs_no_improve=2,
		batch_size=100,
		burn_in_epochs=3,
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
			batch_size)

		avg_val_loss = np.sum(val_batch_sizes * val_loss) / np.sum(val_batch_sizes)
		val_stats.append((0, avg_val_loss))

		print('Initial val loss: %.2f' % val_stats[-1][1])

		# initialize early stopping conditions

		best_val_loss = avg_val_loss
		n_epochs_no_improve = 0

		for epoch_idx in range(max_epochs):

			train_batch_sizes, train_loss = self._run_batches(
				sess,
				[self.loss],
				'train',
				batch_size,
				train_step=self.train_step)

			train_indices = np.arange(len(train_loss)) + epoch_idx * batches_per_epoch
			train_stats.extend(list(zip(train_indices, train_loss)))

			# validation batches

			val_batch_sizes, val_loss = self._run_batches(
				sess,
				[self.loss],
				'val',
				batch_size)

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

		batch_sizes, y_pred, y_prob_pred, y, loss, image_features = self._run_batches(
			sess,
			[self.y_pred, self.y_prob_pred, self.y, self.loss, self.image_features],
			part,
			batch_size)

		avg_loss = (loss * batch_sizes) / np.sum(batch_sizes)

		cat_cols = self.dataloader.is_categorical
		y_pred[:, cat_cols] = y_prob_pred[:, cat_cols]

		return y_pred, y, avg_loss, image_features


	def _run_batches(self, sess, tensors, part, batch_size, train_step=None):

		results = [[] for t in tensors]
		batch_sizes = []

		for batch_idx, (xib, xfb, yb) in enumerate(self.dataloader.get_batch(
			part, batch_size)):

			print('Starting %s batch %i' % (part, batch_idx))

			if train_step is not None:

				results_ = sess.run(
					tensors + [train_step],
					feed_dict={
						self.xi: xib,
						self.y: yb,
						self.is_training: True})

			else:

				results_ = sess.run(
					tensors,
					feed_dict={
						self.xi: xib,
						self.y: yb,
						self.is_training: False})

			batch_sizes.append(len(xib))

			for i in range(len(tensors)):
				results[i].append(results_[i])

		return (np.array(batch_sizes), ) + tuple(try_concat(r, axis=0) for r in results)


	def _build_placeholders(self):

		self.xi = tf.compat.v1.placeholder(
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
			
			logits, endpoints = mobilenet_v2.mobilenet(self.xi)

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

		self.loss_se = (self.y - self.y_pred) ** 2

		self.loss_ce = tf.compat.v1.losses.sigmoid_cross_entropy(
			multi_class_labels=self.y,
			logits=self.y_pred,
			reduction='none')

		se_mask = (~self.dataloader.is_categorical).astype(float)[np.newaxis, :]
		ce_mask = (self.dataloader.is_categorical).astype(float)[np.newaxis, :]

		self.loss = tf.reduce_mean(
			se_mask * self.loss_se + ce_mask * self.loss_ce)

		self.full_train_step = tf.compat.v1.train.AdamOptimizer(
			self.learning_rate).minimize(self.loss)

		myvars = tf.compat.v1.get_collection(
			tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
			scope='outcomes')

		self.logit_train_step = tf.compat.v1.train.AdamOptimizer(
			self.logit_learning_rate).minimize(self.loss, var_list=myvars)


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
