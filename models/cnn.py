import os
import time

import numpy as np
import np_utils
import tensorflow as tf

from models.early_stop import EarlyStopping
from utils.log import log


class ConvNN(object):
	def __init__(self,target,batch_size=16,nb_classes=2,epochs=2,mode='cv'):
		self.target = target
		self.batch_size = batch_size
		self.nb_classes = nb_classes
		self.epochs = epochs
		self.mode = mode

	def setup(self,X_shape):
		tf.reset_default_graph()
		print ('X_train shape', X_shape)
		# Input shape = (None,22,59,114,1)
		self.x = tf.placeholder(tf.float32,
								[None, X_shape[1], X_shape[2], X_shape[3], X_shape[4]])
		self.y_ = tf.placeholder(tf.float32, [None, 2])

		self.y_conv, self.training = base_cnn(self.x, X_shape)
		self.predictions = tf.nn.softmax(self.y_conv)

		with tf.name_scope('loss'):
			cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.y_, logits=self.y_conv)
		self.cross_entropy = tf.reduce_mean(cross_entropy)

		with tf.name_scope('adam_optimizer'):
			self.train_step = tf.train.AdamOptimizer(5e-4).minimize(self.cross_entropy)

		with tf.name_scope('accuracy'):
			correct_prediction = tf.equal(
                tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
			correct_prediction = tf.cast(correct_prediction, tf.float32)
		self.accuracy = tf.reduce_mean(correct_prediction)

		self.saver = tf.train.Saver(max_to_keep=14)
		return self


	def fit(self,X_train,Y_train,X_val=None, y_val=None, batch_size=100, steps=2000, every_n_step=100):
		print ('Start training using batch_size=%d, steps=%d, check model every %d steps'
		       %(batch_size, steps, every_n_step))
		#Y_train = np_utils.to_categorical(Y_train, self.nb_classes)
		#y_val = np_utils.to_categorical(y_val, self.nb_classes)
		Y_train = np.eye(2)[Y_train]
		y_val = np.eye(2)[y_val]

		if X_val is None:
			val_size = int(0.25*X_train.shape[0])
		else:
			val_size = 0
		def next_batch(batch_size):
			indices = np.random.permutation(X_train.shape[0] - val_size)
			xb = X_train[indices[:batch_size]]
			yb = Y_train[indices[:batch_size]]
			return [xb, yb]

		early_stop = EarlyStopping(patience=12, crit='min')

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			start_time = time.time()
			train_accuracy = 0
			train_loss = 0
			for i in range(1,steps+1):
				batch = next_batch(batch_size)
				if i%every_n_step == 0:
					print ('Executing time is %.1f' %(time.time()-start_time))
					start_time = time.time()
					# train_accuracy /= every_n_step
					train_loss /= every_n_step
					#mbatch = next_batch(X_train.shape[0]-val_size)
					if val_size > 0:
						train_accuracy = self.accuracy.eval(feed_dict={
							self.x: batch[0],
							self.y_: batch[1],
							self.training: False
						})
						val_accuracy = self.accuracy.eval(feed_dict={
							self.x: X_train[-val_size:],
							self.y_: Y_train[-val_size:],
							self.training: False
						})
						# train_loss = self.cross_entropy.eval(feed_dict={
						# 	self.x: mbatch[0],
						# 	self.y_: mbatch[1],
						# 	self.training: False
						# })
						val_loss = self.cross_entropy.eval(feed_dict={
							self.x: X_train[-val_size:],
							self.y_: Y_train[-val_size:],
							self.training: False
						})
					else:
						train_accuracy = self.accuracy.eval(feed_dict={
							self.x: batch[0],
							self.y_: batch[1],
							self.training: False
						})
						val_accuracy = self.accuracy.eval(feed_dict={
							self.x: X_val,
							self.y_: y_val,
							self.training: False
						})
						# train_loss = self.cross_entropy.eval(feed_dict={
						# 	self.x: mbatch[0],
						# 	self.y_: mbatch[1],
						# 	self.training: False
						# })
						val_loss = self.cross_entropy.eval(feed_dict={
							self.x: X_val,
							self.y_: y_val,
							self.training: False
						})
					print('Step %d: training accuracy %g, validation accuracy %g'
					      % (i, train_accuracy, val_accuracy))

					save_path = self.saver.save(sess,
					                            "./cache/%s-model-%d.ckpt" %(self.target,i))
					#print("Model saved in file: %s" % save_path)

					stop_step = early_stop.check(i,train_loss+val_loss)
					if stop_step is not None:
						save_path = "./cache/%s-model-%d.ckpt" %(self.target,stop_step)
						print("Early stopping. Optimum mode saved in file: %s" % save_path)
						log("Early stopping. Optimum mode saved in file: %s" % save_path)
						break

					train_accuracy = 0
					train_loss = 0

				self.train_step.run(feed_dict={
					self.x: batch[0],
					self.y_: batch[1],
					self.training: True})
				# _train_accuracy = self.accuracy.eval(feed_dict={
				# 			self.x: batch[0],
				# 			self.y_: batch[1],
				# 			self.training: False
				# 		})
				# train_accuracy += _train_accuracy
				_train_loss = self.cross_entropy.eval(feed_dict={
							self.x: batch[0],
							self.y_: batch[1],
							self.training: False
						})
				train_loss += _train_loss
			self.save_path = "./cache/%s-model-%d.ckpt" \
			                 %(self.target,early_stop.get_optimum_step())
			log(self.save_path)

	def load_trained_weights(self, filename):
		self.save_path = filename
		with tf.Session() as sess:
			self.saver.restore(sess, self.save_path)
		print ('Loading pre-trained weights from %s.' %filename)
		return self

	def predict_proba(self,X):
		with tf.Session() as sess:
			self.saver.restore(sess, self.save_path)
			predictions = self.predictions.eval(feed_dict={
				self.x: X,
				#self.y_: y,
				self.training: True
			})
		return predictions

	def evaluate(self, X, y):
		y = np.eye(2)[y]
		with tf.Session() as sess:
			self.saver.restore(sess, self.save_path)
			predictions = self.predictions.eval(feed_dict={
				self.x: X,
				self.y_: y,
				self.training: True
			})[:,1]
			from sklearn.metrics import roc_auc_score
			auc_test = roc_auc_score(y[:,1], predictions)
			print('Test AUC is:', auc_test)
			log('Test AUC is: %.2f' % auc_test)


def base_cnn(input,X_shape):
	print (X_shape)

	normal1 = tf.layers.batch_normalization(
		inputs=input,
		axis=1, # normalized along EEG channels (not the virtual channel)
		name='norm1'
	)

	conv1 = tf.layers.conv3d(
		inputs=normal1,
		filters=16,
		kernel_size=(X_shape[1], 5, 5),
		padding='valid',strides=(1,2,2),
		data_format='channels_last',
		activation=tf.nn.relu,
		name='conv1'
	)

	pool1 = tf.layers.max_pooling3d(
		inputs = conv1,
		pool_size=(1,2,2),
		strides=(1,2,2),
		padding='same',
		data_format='channels_last',
		name='pool1'
	)

	ts = int(np.round(np.floor((X_shape[2]-4)/2) / 2 + 0.1))
	fs = int(np.round(np.floor((X_shape[3]-4)/2) / 2 + 0.1))

	reshape1 = tf.reshape(pool1, shape=[-1, ts, fs, 16])

	normal2 = tf.layers.batch_normalization(
		inputs=reshape1,
		axis=-1,
		name='norm2'
	)

	conv2 = tf.layers.conv2d(
		inputs=normal2,
		filters=32,
		kernel_size=(3, 3),
		padding='valid',strides=(1,1),
		data_format='channels_last',
		activation=tf.nn.relu,
		name='conv2'
	)

	pool2 = tf.layers.max_pooling2d(
		inputs = conv2,
		pool_size=(2,2),
		strides=(2,2),
		padding='same',
		data_format='channels_last',
		name='pool2'
	)

	normal3 = tf.layers.batch_normalization(
		inputs=pool2,
		axis=-1,
		name='normal3'
	)

	conv3 = tf.layers.conv2d(
		inputs=normal3,
		filters=64,
		kernel_size=(3, 3),
		padding='valid',strides=(1,1),
		data_format='channels_last',
		activation=tf.nn.relu,
		name='conv3'
	)

	pool3 = tf.layers.max_pooling2d(
		inputs = conv3,
		pool_size=(2,2),
		strides=(2,2),
		padding='same',
		data_format='channels_last',
		name='pool3'
	)

	flat = tf.layers.flatten(
		inputs=pool3
	)

	training = tf.placeholder(tf.bool)

	drop1 = tf.layers.dropout(
		inputs=flat,
		rate=0.5,
		training=training
	)

	dens1 = tf.layers.dense(
		inputs=drop1,
		units=128,
		activation=tf.nn.sigmoid,
		name='dens1'
	)

	drop2 = tf.layers.dropout(
		inputs=dens1,
		rate=0.5,
		training=training
	)

	dens2 = tf.layers.dense(
		inputs=drop2,
		units=2,
		name='dens2'
	)

	return dens2,training


