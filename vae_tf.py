import os
import platform

import numpy as np
import scipy
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.examples.tutorials.mnist import input_data
from model import Layers, Utils
import utils
tf.flags.DEFINE_integer('epoch', 10000, "steps")
tf.flags.DEFINE_string('device', '0', 'cuda visible devices')
tf.flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
tf.flags.DEFINE_float('alpha', 0.2, 'alpha between kl_loss and recon_loss')
tf.flags.DEFINE_integer('batch_size', 128, 'batch size')
flags = tf.flags.FLAGS

if 'Linux' in platform.system():
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.device
class VAE(object):
    def __init__(self, input_h, input_w, k_w, k_h, batch_size,
        learning_rate=0.01, alpha=0.2):
        self.input_h = input_h
        self.input_w = input_w
        self.k_h = k_h
        self.k_w = k_w
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.input_x = tf.placeholder(tf.float32, shape=[None, 784], name="input_x")
        self.training = tf.placeholder(tf.bool, name="training")
        self.X = self.input_x


        self.h, _ = Layers.RNN.LSTM(tf.reshape(self.input_x, shape=(-1, 28, 28)), num_units=[50, 20, 10])

        h1 = Layers.dense(self.X, 512, activation=tf.nn.relu, name='encoder_0')
        h1_res = Layers.res_block(h1, 512, name='res_block_0', is_training=self.training)

        h2 = Layers.dense(h1_res, 256, activation=tf.nn.relu, name='encoder_1')
        h2_res = Layers.res_block(h2, 256, name="res_block_1", is_training=self.training)

        h3 = Layers.dense(h2_res, 128, activation=tf.nn.relu, name="encoder")
        h3_res = Layers.res_block(h3, 128, name='res_block_2', is_training=self.training)

        self.mean = Layers.dense(h3_res, 100, activation=None, name="encoder_mean")
        self.mean = Layers.res_block(self.mean, 100, name='res_block_mean', is_training=self.training)

        self.var = Layers.dense(h3_res, 100, activation=None, name="encoder_variance")
        self.var = Layers.res_block(self.var, 100, name='res_block_var', is_training=self.training)

        with tf.name_scope('latent_vector'):
            sampled = Utils.sample(self.mean, self.var)
            sampled = tf.concat(values=[sampled, self.h], axis=1)

        o1 = Layers.dense(sampled, 128, activation=tf.nn.relu, name="decoder_0")
        o1_res = Layers.res_block(o1, 128, name='res_block_3', is_training=self.training)

        o2 = Layers.dense(o1_res, 256, activation=tf.nn.relu, name="decoder_1")
        o2_res = Layers.res_block(o2, 256, name="res_block_4", is_training=self.training)

        o3 = Layers.dense(o2_res, 512, activation=tf.nn.relu, name='decoder_2')
        o3_res = Layers.res_block(o3, 512, name='res_block_5', is_training=self.training)

        self.out = Layers.dense(o3_res, 784, None, name="decoder")

        with tf.name_scope('score'):
            # self.recon_loss = tf.reduce_sum((self.out - self.input_x) ** 2)
            # self.recon_loss = -tf.reduce_sum(self.input_x * tf.log(1e-8 + self.out) + (1 - self.input_x) * tf.log(1e-8 + 1 - self.out))
            self.recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out, labels=self.input_x), 1)
            # self.recon_loss = tf.reduce_mean(tf.image.psnr(tf.reshape(self.out, shape=(-1, 28, 28)), tf.reshape(self.input_x, shape=(-1, 28, 28)), max_val=1.0))
            # self.recon_loss = tf.losses.mean_pairwise_squared_error(self.input_x, self.out)
            self.kl_loss = 0.5 * tf.reduce_sum(tf.exp(self.var) + self.mean**2 - 1. - self.var, 1)
            self.recon_loss = tf.reduce_mean(self.recon_loss)
            self.kl_loss = tf.reduce_mean(self.kl_loss)
            self.loss = self.recon_loss * self.alpha + self.kl_loss
    def preprocess_mnist(self):
        mnist = input_data.read_data_sets('./MNIST', one_hot=False)
        train_data = []
        test_data = []
        for i, l in enumerate(mnist.train.labels):
            if l != 9:
                train_data.append(mnist.train.images[i])
            else:
                test_data.append(mnist.train.images[i])
        return train_data, test_data

    def train_mnist(self):
        train_data, test_data = self.preprocess_mnist()
        global_step = tf.Variable(0, trainable=False, name='global_step')
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(self.loss, global_step=global_step)
        session_conf = tf.ConfigProto(
            allow_soft_placement=True)
        session_conf.gpu_options.allow_growth = True
        self.sess = tf.Session(config=session_conf)
        self.sess.run(tf.initialize_all_variables())
        for i in range(flags.epoch):
            for x in utils.batch_iter(train_data, batch_size=self.batch_size, shuffle=True):
                x = np.asarray(x)
                fetches = [train_op, self.recon_loss, self.kl_loss]
                fetch = self.sess.run(fetches, feed_dict={
                    self.input_x: x, self.training: True
                })
                current_step = tf.train.global_step(self.sess, global_step)
                if current_step % 50 == 0:
                    print('epoch:%3d \t step:%d \t reon_loss:%.5f \t kl_loss:%.5f' % (i, current_step, fetch[1], fetch[2]))
                if current_step % 100 == 0:
                    loss, kl_loss = self.sess.run([self.recon_loss, self.kl_loss], feed_dict={
                        self.input_x: test_data, self.training: False
                    })
                    print("Evaluation:")
                    print("loss:%.5f, kl_loss:%.5f" % (loss, kl_loss))
    def test_mnist(self):
        mnist = input_data.read_data_sets('./MNIST', one_hot=False)
        test_data = mnist.test.images
        recon = self.sess.run(self.out, feed_dict={
            self.input_x: test_data, self.training: False
        })
        recon = recon[0:200]
        inputs = mnist.test.images[0:200]
        # inputs = np.squeeze(inputs, -1)
        recon = np.reshape(recon, (200, 28, 28))
        inputs = np.reshape(inputs, (200, 28, 28))
        scipy.misc.imsave('./generate.jpg', utils.montage(recon))
        scipy.misc.imsave('./inputs.jpg', utils.montage(inputs))
if __name__ == '__main__':
    vae = VAE(input_h=28, input_w=28, k_w=3, k_h=3, batch_size=flags.batch_size, learning_rate=flags.learning_rate, alpha=flags.alpha)
    vae.train_mnist()
    vae.test_mnist()