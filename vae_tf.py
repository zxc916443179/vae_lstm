import os
import platform

import numpy as np
import scipy
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.examples.tutorials.mnist import input_data
from model import Layers, Utils
import utils
from vae import xavier_init
tf.flags.DEFINE_integer('epoch', 10000, "steps")
tf.flags.DEFINE_string('device', '0', 'cuda visible devices')
flags = tf.flags.FLAGS

if 'Linux' in platform.system():
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.device
class VAE(object):
    def __init__(self, input_h, input_w, k_w, k_h):
        self.input_h = input_h
        self.input_w = input_w
        self.k_h = k_h
        self.k_w = k_w
        self.input_x = tf.placeholder(tf.float32, shape=[None, 784], name="input_x")
        self.training = tf.placeholder(tf.bool, name="training")
        self.X = self.input_x

        self.h, _ = Layers.RNN.LSTM(tf.reshape(self.input_x, shape=(-1, 28, 28)), num_units=[20, 10])
        self.Q = Layers.dense(self.X, 128, activation=tf.nn.relu, name="encoder")

        with tf.name_scope('res_block_0'):
            ha = Layers.dense(self.Q, 128, activation=tf.nn.relu)
            ha_bn = Layers.batch_norm(ha, is_training=self.training)
            hb = Layers.dense(ha_bn, 128, activation=None)
            hb_bn = Layers.batch_norm(hb, is_training=self.training)

            hc = Layers.dense(self.Q, 128, activation=tf.nn.relu)
            hc_bn = Layers.batch_norm(hc, is_training=self.training)
            
            out = tf.nn.relu(hb_bn + hc_bn)

        self.mean = Layers.dense(out, 100, activation=None, name="encoder_mean")
        self.var = Layers.dense(out, 100, activation=None, name="encoder_variance")

        with tf.name_scope('latent_vector'):
            sampled = Utils.sample(self.mean, self.var)
            sampled = tf.concat(values=[sampled, self.h], axis=1)

        h1 = Layers.dense(sampled, 128, activation=tf.nn.relu, name="decoder_1")

        with tf.name_scope('res_block_1'):
            ha = Layers.dense(h1, 128, activation=tf.nn.relu)
            ha_bn = Layers.batch_norm(ha, is_training=self.training)
            hb = Layers.dense(ha_bn, 128, activation=None)
            hb_bn = Layers.batch_norm(hb, is_training=self.training)

            hc = Layers.dense(h1, 128, activation=tf.nn.relu)
            hc_bn = Layers.batch_norm(hc, is_training=self.training)
            
            out = tf.nn.relu(hb_bn + hc_bn)

        self.out = Layers.dense(out, 784, None, name="decoder")

        with tf.name_scope('score'):
            # self.recon_loss = tf.reduce_sum((self.out - self.input_x) ** 2)
            # self.recon_loss = -tf.reduce_sum(self.input_x * tf.log(1e-8 + self.out) + (1 - self.input_x) * tf.log(1e-8 + 1 - self.out))
            self.recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out, labels=self.input_x), 1)
            # self.recon_loss = tf.reduce_mean(tf.image.psnr(tf.reshape(self.out, shape=(-1, 28, 28)), tf.reshape(self.input_x, shape=(-1, 28, 28)), max_val=1.0))
            # self.recon_loss = tf.losses.mean_pairwise_squared_error(self.input_x, self.out)
            self.kl_loss = 0.5 * tf.reduce_sum(tf.exp(self.var) + self.mean**2 - 1. - self.var, 1)
            self.loss = tf.reduce_mean(self.recon_loss + self.kl_loss)
    
if __name__ == '__main__':
    mnist = input_data.read_data_sets('./MNIST', one_hot=True)
    train_data = mnist.train.images[0:8]
    test_data = mnist.train.images[9]
    # x_train = mnist.train.images
    vae = VAE(input_h=28, input_w=28, k_w=3, k_h=3)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.AdamOptimizer(0.01)
    # optimizer = tf.train.GradientDescentOptimizer(0.001)
    # optimizer = tf.train.MomentumOptimizer(0.01, 0.9)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = optimizer.minimize(vae.loss, global_step=global_step)

    session_conf = tf.ConfigProto(
        allow_soft_placement=True)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    sess.run(tf.initialize_all_variables())
    for i in range(flags.epoch):
        for x in utils.batch_iter(train_data, batch_size=128, shuffle=True):
            x = np.asarray(x)
            fetches = [train_op, vae.loss]
            fetch = sess.run(fetches, feed_dict={
                vae.input_x: x, vae.training: True
            })
            current_step = tf.train.global_step(sess, global_step)
            if current_step % 50 == 0:
                print('epoch:%3d \t step:%d \t loss:%5f' % (i, current_step, fetch[1]))
            if current_step % 100 == 0:
                loss = sess.run(vae.loss, feed_dict={
                    vae.input_x: test_data, vae.training: False
                })
                print("Evaluation:")
                print("loss:%.5f" % loss)

    recon = sess.run(vae.out, feed_dict={
        vae.input_x: test_data, vae.training: False
    })
    recon = recon[0:200]
    print(recon.shape)
    inputs = mnist.test.images[0:200]
    # inputs = np.squeeze(inputs, -1)
    recon = np.reshape(recon, (200, 28, 28))
    inputs = np.reshape(inputs, (200, 28, 28))
    scipy.misc.imsave('./generate.jpg', utils.montage(recon))
    scipy.misc.imsave('./inputs.jpg', utils.montage(inputs))
