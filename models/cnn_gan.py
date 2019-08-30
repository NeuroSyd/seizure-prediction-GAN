import os
import glob
import time
import shutil

import numpy as np
import np_utils
import tensorflow as tf

from models.early_stop import EarlyStopping
from utils.log import log

from dcgan.ops import lrelu, linear, concat, conv2d, batch_norm, conv3d

conv1_GAN = None

class CNNGAN():
    def __init__(self,
                 target,
                 batch_size=16,
                 nb_classes=2,
                 epochs=2,
                 mode='cv',
                 dataset='Kaggle2014Pred',
                 sph=5,
                 cache='./cache',
                 checkpoint='./checkpoint',
                 result_dir='./results'):
        self.target = target
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.epochs = epochs
        self.mode = mode
        self.dataset = dataset
        self.cache = cache
        self.checkpoint = checkpoint
        self.result_dir = result_dir
        self.sph = sph
        

    def setup(self, X_shape):
        tf.reset_default_graph()
        self.x = tf.placeholder(tf.float32,[None, X_shape[1], X_shape[2], X_shape[3]])
        self.y_ = tf.placeholder(tf.float32, [None, 2])

        self.y_conv, self.training = base_cnn(self.x, X_shape, dataset=self.dataset, checkpoint=self.checkpoint, sph=self.sph)
        self.predictions = tf.nn.softmax(self.y_conv)

        print ('SETUP y_conv', self.y_conv.get_shape())

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.y_, logits=self.y_conv)
        self.cross_entropy = tf.reduce_mean(cross_entropy)

        # with tf.name_scope('adam_optimizer'):
        #     self.train_step = tf.train.AdamOptimizer(
        #         1e-4).minimize(self.cross_entropy)

        with tf.name_scope('rmsprop_optimizer'):
            self.train_step = tf.train.RMSPropOptimizer(
                1e-4).minimize(self.cross_entropy)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(
                tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(correct_prediction)

        self.saver = tf.train.Saver(max_to_keep=14)
        return self

    def fit(self, X_train, Y_train, X_val=None, y_val=None, batch_size=100, steps=2000, every_n_step=100):
        global conv1_GAN
        #print (conv1_GAN[0])


        print('Start training using batch_size=%d, steps=%d, check model every %d steps'
              % (batch_size, steps, every_n_step))
        #Y_train = np_utils.to_categorical(Y_train, self.nb_classes)
        #y_val = np_utils.to_categorical(y_val, self.nb_classes)
        Y_train = np.eye(2)[Y_train]
        y_val = np.eye(2)[y_val]

        if X_val is None:
            val_size = min(int(0.25*X_train.shape[0]),5000) # work around for OOM
        else:
            val_size = 0

        def next_batch(batch_size):
            indices = np.random.permutation(X_train.shape[0] - val_size)
            xb = X_train[indices[:batch_size]]
            yb = Y_train[indices[:batch_size]]
            return [xb, yb]

        early_stop = EarlyStopping(patience=12, crit='min')
        


        dum_graph = tf.Graph()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

        #     imported_meta = tf.train.import_meta_graph(
        # #"./checkpoint/model_dir_STFT_%s/DCGAN.model-61501.meta" %dataset)  # FB
        # "./checkpoint/model_dir_STFT_%s/DCGAN.model-6501.meta" %self.dataset)    #Kaggle2014Pred
        #
        #
        #     imported_meta.restore(sess,
        #                     tf.train.latest_checkpoint('./checkpoint/model_dir_STFT_%s/' %self.dataset))
            conv1 = sess.run('discriminator/d_h1_conv/w:0')
            print (conv1[0])

            start_time = time.time()

            train_accuracy = 0
            train_loss = 0
            for i in range(1, steps+1):
                batch = next_batch(batch_size)
                if i % every_n_step == 0:
                    print('Executing time is %.1f' % (time.time()-start_time))
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
                            self.x: X_val[:5000], # work around for OOM
                            self.y_: y_val[:5000],
                            self.training: False
                        })
                        # train_loss = self.cross_entropy.eval(feed_dict={
                        # 	self.x: mbatch[0],
                        # 	self.y_: mbatch[1],
                        # 	self.training: False
                        # })
                        val_loss = self.cross_entropy.eval(feed_dict={
                            self.x: X_val[:5000], # work around for OOM
                            self.y_: y_val[:5000],
                            self.training: False
                        })
                    print('Step %d: training accuracy %g, validation accuracy %g'
                          % (i, train_accuracy, val_accuracy))


                    # check if GAN trained weights were properly loaded and not changing
                    d_h1_conv = sess.run('discriminator/d_h1_conv/w:0')
                    assert (d_h1_conv==conv1_GAN).all()
                    print ('ALL GOOD....')

                    save_path = self.saver.save(
                        sess, self.cache + "/%s-model-%d.ckpt" % (self.target, i))
                    #print("Model saved in file: %s" % save_path)

                    stop_step = early_stop.check(i, train_loss+val_loss)
                    if stop_step is not None:
                        save_path = self.cache + "/%s-model-%d.ckpt" % (self.target, stop_step)
                        print(
                            "Early stopping. Optimum mode saved in file: %s" % save_path)
                        log("Early stopping. Optimum mode saved in file: %s" %
                            save_path)
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
            self.save_path = self.cache + "/%s-model-%d.ckpt" \
                % (self.target, early_stop.get_optimum_step())
            for suff in ['.index', '.meta', '.data-00000-of-00001']:
                shutil.copy2(
                    self.save_path + suff,
                    self.result_dir)
            log(self.save_path)

    def load_trained_weights(self, filename):
        self.save_path = filename
        with tf.Session() as sess:
            self.saver.restore(sess, self.save_path)
        print('Loading pre-trained weights from %s.' % filename)
        return self

    def predict_proba(self, X):
        batch_size = 1000
        n_batch = int(X.shape[0]/batch_size) + 1
        predictions_ = []
        with tf.Session() as sess:
            self.saver.restore(sess, self.save_path)
            for ie in range(n_batch):
                x_ = X[ie*batch_size: min((ie+1)*batch_size, X.shape[0])]
                if x_.shape[0]>0:
                    predictions_.append(self.predictions.eval(feed_dict={
                        self.x: x_,
                        self.training: False
                    }))
            predictions = np.concatenate(predictions_)
            # predictions = self.predictions.eval(feed_dict={
            #     self.x: X,
            #     # self.y_: y,
            #     self.training: False
            # })
        return predictions

    def evaluate(self, X, y):
        y = np.eye(2)[y]
        batch_size = 1000
        n_batch = int(y.shape[0]/batch_size) + 1
        predictions_ = []
        with tf.Session() as sess:
            self.saver.restore(sess, self.save_path)
            for ie in range(n_batch):
                x_ = X[ie*batch_size: min((ie+1)*batch_size, X.shape[0])]
                if x_.shape[0]>0:
                    predictions_.append(self.predictions.eval(feed_dict={
                        self.x: x_,
                        self.y_: y[ie*batch_size: min((ie+1)*batch_size, X.shape[0])],
                        self.training: False
                    })[:, 1])
            predictions = np.concatenate(predictions_)
            # predictions = self.predictions.eval(feed_dict={
            #     self.x: X,
            #     self.y_: y,
            #     self.training: False
            # })[:, 1]
            from sklearn.metrics import roc_auc_score
            auc_test = roc_auc_score(y[:, 1], predictions)
            print('Test AUC is:', auc_test)
            return auc_test


def base_cnn(input, X_shape, dataset='FB', checkpoint='./checkpoint',sph=5):
    global conv1_GAN

    new_graph = tf.Graph()
    meta_list = glob.glob(checkpoint + "/model_dir/*.meta")
    meta_path = max(meta_list, key=os.path.getctime)
    print (meta_path)

    # if dataset=='Kaggle2014Pred':
    #     meta_path = checkpoint + "/model_dir_STFT_%s/DCGAN.model-6501.meta" %dataset
    # elif dataset=='FB':
    #     meta_path = checkpoint + "/model_dir_STFT_%s/DCGAN.model-61501.meta" %dataset
    # elif dataset=='CHBMIT':
    #     meta_path = checkpoint + "/model_dir_STFT_%s/DCGAN.model-49501.meta" %dataset
    with tf.Session(graph=new_graph) as sess:

        imported_meta = tf.train.import_meta_graph(meta_path)  

        imported_meta.restore(
            sess,
            tf.train.latest_checkpoint(
                checkpoint + "/model_dir/"))
        #print (tf.global_variables())
        d_h0_conv = sess.run('discriminator/d_h0_conv/w:0')
        d_h1_conv = sess.run('discriminator/d_h1_conv/w:0')
        d_h2_conv = sess.run('discriminator/d_h2_conv/w:0')

        conv1_GAN = d_h1_conv

        print ('X_shape', X_shape)

        #asd

    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE) as scope:
        training = tf.placeholder(tf.bool)

        df_dim = 16

        d_bn0 = batch_norm(name='d_bn0')
        d_bn1 = batch_norm(name='d_bn1')
        d_bn2 = batch_norm(name='d_bn2')

        h0 = lrelu(d_bn0(conv2d(input, df_dim, name='d_h0_conv', restore=True, trained_weights=d_h0_conv), train=False))

        h1 = lrelu(d_bn1(conv2d(h0, df_dim*2, name='d_h1_conv', restore=True, trained_weights=d_h1_conv), train=False))
        h2 = lrelu(d_bn2(conv2d(h1, df_dim*4, name='d_h2_conv', restore=True, trained_weights=d_h2_conv), train=False))
        #h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))


        fl = tf.layers.flatten(inputs=h2)
        dr1 = tf.layers.dropout(inputs=fl, rate=0.5, training=training)
        d1 = linear(dr1, 256, 'd_d1_lin')
        dr2 = tf.layers.dropout(inputs=d1, rate=0.5, training=training)
        d2 = linear(tf.nn.sigmoid(dr2), 2, 'd_d2_lin')
        print ('base_cnn d2', d2.get_shape())

        return d2, training


class CNNGAN_infer(CNNGAN):
    def setup(self, X_shape):
        tf.reset_default_graph()
        print ('X_shape', X_shape)
        self.x = tf.placeholder(tf.float32,[1, X_shape[1], X_shape[2], X_shape[3]], name='input')

        # Build the graph for the deep net
        y_conv = base_cnn_infer(self.x)
        output = tf.nn.softmax( y_conv, name='output' )
        self.predictions = output
        self.training = tf.placeholder(tf.bool) # not used, just to make compatible with training code
        # saver = tf.train.Saver(tf.global_variables())

        # meta_path = self.cache + "/%s-model" %self.target
        #
        # with tf.Session() as sess:
        #   sess.run(tf.global_variables_initializer())
        #   sess.run(tf.local_variables_initializer())
        #   saver.restore(sess, meta_path )
        #   saver.save(sess, meta_path + '-inference' )
        #   #print ([n.name for n in tf.get_default_graph().as_graph_def().node])

        return self

    def load_trained_weights(self, filename):
        self.save_path = filename
        self.saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            self.saver.restore(sess, filename)
            self.saver.save(sess, filename + '-inference')
        print ('Save inference model to %s-inference.' %filename)
        return self

def base_cnn_infer(input):

    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE) as scope:

        df_dim = 16

        d_bn0 = batch_norm(name='d_bn0')
        d_bn1 = batch_norm(name='d_bn1')
        d_bn2 = batch_norm(name='d_bn2')

        h0 = lrelu(d_bn0(conv2d(input, df_dim, name='d_h0_conv'), train=False))

        h1 = lrelu(d_bn1(conv2d(h0, df_dim*2, name='d_h1_conv'), train=False))
        h2 = lrelu(d_bn2(conv2d(h1, df_dim*4, name='d_h2_conv'), train=False))


        fl = tf.layers.flatten(inputs=h2)

        d1 = linear(fl, 256, 'd_d1_lin')

        d2 = linear(tf.nn.sigmoid(d1), 2, 'd_d2_lin')

        return d2
