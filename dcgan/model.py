# adapt from https://github.com/shaohua0116/DCGAN-Tensorflow
from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from dcgan.ops import *
from utils import *

from models.early_stop import EarlyStopping

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
  def __init__(self, sess, target, input_height=56, input_width=112, crop=False,
         batch_size=64, sample_num = 64, output_height=56, output_width=112,
         y_dim=None, z_dim=100, gf_dim=16, df_dim=16,
         gfc_dim=1024, dfc_dim=1024, dataset_dir=None,
         checkpoint_dir="checkpoint", sample_dir=None):
   
    self.sess = sess
    self.crop = crop

    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.y_dim = y_dim
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn0 = batch_norm(name='d_bn0')
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')

    if not self.y_dim:
      self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')

    if not self.y_dim:
      self.g_bn3 = batch_norm(name='g_bn3')

    self.dataset_dir = dataset_dir
    self.input_fname_pattern = "%s_*.npy" % target
    # self.input_fname_pattern = "*.npy"
    self.checkpoint_dir = checkpoint_dir

    self.data = []
    for data_dir in self.dataset_dir:
      self.data += glob(os.path.join(data_dir, self.input_fname_pattern))
      print (len(self.data))
    #self.data = glob(os.path.join(self.dataset_dir, self.input_fname_pattern))
    #print (self.data)
    print ('Total files', len(self.data), self.input_fname_pattern)
    
    tmp = np.load(self.data[0])
    self.c_dim = tmp.shape[-3]
    # for idx in range(len(self.data)):
    #   tmp2 = np.load(self.data[idx])
    #   if tmp2.shape != tmp.shape:
    #     print (tmp2.shape, self.data[i])

    self.grayscale = (self.c_dim == 1)

    self.build_model()

  def build_model(self):
    if self.y_dim:
      self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
    else:
      self.y = None

    if self.crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      self.output_height = self.input_height
      self.output_width = self.input_width
      image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')

    inputs = self.inputs

    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')
    self.z_sum = histogram_summary("z", self.z)

    self.G                  = self.generator(self.z, self.y)
    self.D, self.D_logits   = self.discriminator(inputs, self.y, reuse=False)   
    self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)
    
    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.G_sum = image_summary("G", self.G[:,:,:,:3])
    
    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    self.d_loss_real = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
    self.g_loss = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
                          
    self.d_loss = self.d_loss_real + self.d_loss_fake

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

    self.model_dir = 'model_dir'

  def train(self, config):
    d_optim = tf.train.AdamOptimizer(config["learning_rate"], beta1=config["beta1"]) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config["learning_rate"], beta1=config["beta1"]) \
              .minimize(self.g_loss, var_list=self.g_vars)
    # d_optim = tf.train.RMSPropOptimizer(config["learning_rate"]) \
    #           .minimize(self.d_loss, var_list=self.d_vars)
    # g_optim = tf.train.RMSPropOptimizer(config["learning_rate"]) \
    #           .minimize(self.g_loss, var_list=self.g_vars)
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.g_sum = merge_summary([self.z_sum, self.d__sum,
      self.G_sum,
                                self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = merge_summary(
        [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    self.writer = SummaryWriter("./logs", self.sess.graph)   
  
    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    earlystopping = EarlyStopping(patience=20, crit='gan')
    stopping_check = None

    for epoch in xrange(config["epoch"]):
      if stopping_check is None:
        print ('Epoch', epoch)
        batch_idxs = min(len(self.data), config["train_size"]) // config["batch_size"]      

        for idx in xrange(0, batch_idxs):          
          batch_files = self.data[idx*config["batch_size"]:(idx+1)*config["batch_size"]]
          
          batch = [
              np.load(batch_file) for batch_file in batch_files]

          #print ('batch', len(batch), batch[0].shape)

          batch_images = np.array(batch, dtype=np.float32)#.astype(np.float32)
          batch_images = np.transpose(batch_images, (0,2,3,1))

          batch_z = np.random.uniform(-1, 1, [config["batch_size"], self.z_dim])#.astype(np.float32)


          # Update D network
          _, summary_str = self.sess.run([d_optim, self.d_sum],
            feed_dict={ self.inputs: batch_images, self.z: batch_z })
          self.writer.add_summary(summary_str, counter)

          # Update G network
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={ self.z: batch_z })
          self.writer.add_summary(summary_str, counter)

          # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={ self.z: batch_z })
          self.writer.add_summary(summary_str, counter)

          errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
          errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
          errG = self.g_loss.eval({self.z: batch_z})

          loss = "%d,%d,%.4f,%.4f" %(epoch,idx,errD_fake+errD_real, errG) 
          log.log_loss(loss)

          counter += 1
          stopping_check = earlystopping.check(counter, errG-errD_fake-errD_real)
          if stopping_check is not None:
            self.save(config["checkpoint_dir"], counter)
            break
          print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
            % (epoch, idx, batch_idxs,
              time.time() - start_time, errD_fake+errD_real, errG))          

          if np.mod(counter, 200) == 1:            
            self.save(config["checkpoint_dir"], counter)


  def discriminator(self, image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      if not self.y_dim:
        in_ = image
        print ('IMAGE SHAPE', image.shape)
        h0 = lrelu(self.d_bn0(conv2d(in_, self.df_dim, name='d_h0_conv')))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))        
        h3 = linear(tf.reshape(h2, [self.batch_size, -1]), 1, 'd_h3_lin')    
        return tf.nn.sigmoid(h3), h3      

  def generator(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      if not self.y_dim:
        
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)       

        # project `z` and reshape
        self.z_, self.h0_w, self.h0_b = linear(
            z, self.gf_dim*4*s_h8*s_w8, 'g_h0_lin', with_w=True)

        self.h0 = tf.reshape(
            self.z_, [-1, s_h8, s_w8, self.gf_dim * 4])
        h0 = tf.nn.relu(self.g_bn0(self.h0))

        self.h1, self.h1_w, self.h1_b = deconv2d(
            h0, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))

        h2, self.h2_w, self.h2_b = deconv2d(
            h1, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))        

        h3, self.h3_w, self.h3_b = deconv2d(
            h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3', with_w=True)

        print ('generator h3', h3.get_shape())
        return tf.nn.tanh(h3)    
 


  def save(self,checkpoint_dir='checkpoint', step=2):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    return self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)


  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...", checkpoint_dir, self.model_dir)
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
    print (checkpoint_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
