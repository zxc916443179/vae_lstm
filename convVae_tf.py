import os
import platform

import numpy as np
import scipy
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.examples.tutorials.mnist import input_data
from model import Layers, Utils
import utils
tf.flags.DEFINE_integer('epoch', 10000, "training epoches")
tf.flags.DEFINE_string('device', '0', 'cuda visible devices')
tf.flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
tf.flags.DEFINE_float('alpha', 0.2, 'alpha between kl_loss and recon_loss')
tf.flags.DEFINE_integer('batch_size', 128, 'batch size')
tf.flags.DEFINE_integer('input_size', 45, 'size of images(default:28*28)')
tf.flags.DEFINE_string('dataset_path', './UCSDped_patch/ped1', 'path to dataset')
flags = tf.flags.FLAGS

if 'Linux' in platform.system():
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.device
class VAE(object):
    def __init__(self, input_h, input_w, batch_size,
        learning_rate=0.01, alpha=0.2, dataset_path=None):
        self.input_h = input_h
        self.input_w = input_w
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dataset_path = dataset_path

        self.input_x_ = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_h, self.input_w], name="input_x")
        self.training = tf.placeholder(tf.bool, name="training")
        
        self.h, _ = Layers.RNN.LSTM(utils.generate_lstm_input(self.input_x_), num_units=[512, 256, 128, 10])
        self.input_x = tf.expand_dims(self.input_x_, -1)


        h1 = Layers.conv2d(self.input_x, 128, 3, 2, activation=tf.nn.leaky_relu, padding='VALID')
        with tf.name_scope('res_block_0'):
            h2 = Layers.conv2d(h1, 128, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            h2_bn = Layers.batch_norm(h2, is_training=self.training)
            h3 = Layers.conv2d(h2_bn, 128, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            h3_bn = Layers.batch_norm(h3, is_training=self.training)

            
            h4 = Layers.conv2d(h1, 128, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            h4_bn = Layers.batch_norm(h4, is_training=self.training)

            h5 = h4_bn + h3_bn

        h6 = Layers.conv2d(h5, 256, 3, 2, activation=tf.nn.leaky_relu, padding='VALID')
        with tf.name_scope('res_block_1'):
            h7 = Layers.conv2d(h6, 256, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            h7_bn = Layers.batch_norm(h7, is_training=self.training)
            h8 = Layers.conv2d(h7_bn, 256, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            h8_bn = Layers.batch_norm(h8, is_training=self.training)

            
            h4 = Layers.conv2d(h6, 256, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            h4_bn = Layers.batch_norm(h4, is_training=self.training)

            h5 = h4_bn + h8_bn
        h = Layers.conv2d(h5, 1, 1, 1, activation=tf.nn.sigmoid, padding="VALID")
        h = tf.reshape(tf.squeeze(h, -1), (self.batch_size, 10 * 10))

        mean = Layers.dense(h , 90, activation=tf.nn.sigmoid)
        self.mean = Layers.res_block(mean, 90, fn=Layers.dense, is_training=self.training)
        # var = Layers.dense(h, 90, activation=tf.nn.sigmoid)
        # self.var = Layers.res_block(var, 90, fn=Layers.dense, is_training=self.training)
        # sampled = Utils.sample(self.mean, self.var)
        sampled = tf.expand_dims(tf.reshape(tf.concat(values=[self.mean, self.h], axis=1), (self.batch_size, 10, 10)), axis=-1)
        
        # estimator
        e1 = Layers.dense(self.mean, 128, activation=tf.nn.leaky_relu)
        e2 = Layers.dense(e1, 256, activation=tf.nn.leaky_relu)
        e3 = Layers.dense(e2, 128, activation=tf.nn.leaky_relu)
        e_out = Layers.dense(e3, 1, activation=tf.nn.sigmoid)

        with tf.name_scope('res_block_3'):
            d1 = Layers.conv2d(sampled, 256, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            d1_bn = Layers.batch_norm(d1, is_training=self.training)
            d2 = Layers.conv2d(d1_bn, 256, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            d2_bn = Layers.batch_norm(d2, is_training=self.training)

            
            d3 = Layers.conv2d(sampled, 256, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            d3_bn = Layers.batch_norm(d3, is_training=self.training)

            d = d3_bn + d2_bn
        
        o1 = Layers.conv2d_transpose(d, 128, 3, 2, padding='VALID', output_shape=[128, h1.get_shape()[1].value, h1.get_shape()[2].value, h1.get_shape()[3].value], activation=tf.nn.leaky_relu)
        with tf.name_scope('res_block_3'):
            d1 = Layers.conv2d(o1, 128, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            d1_bn = Layers.batch_norm(d1, is_training=self.training)
            d2 = Layers.conv2d(d1_bn, 128, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            d2_bn = Layers.batch_norm(d2, is_training=self.training)

            
            d3 = Layers.conv2d(o1, 128, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            d3_bn = Layers.batch_norm(d3, is_training=self.training)

            d = d3_bn + d2_bn
        
        self.out = Layers.conv2d_transpose(d, 1, 3, 2, padding="VALID", output_shape=[128, self.input_x.get_shape()[1].value, self.input_x.get_shape()[2].value, self.input_x.get_shape()[3].value], activation=tf.nn.sigmoid)
        
        
        with tf.name_scope('score'):
            # self.recon_loss = tf.reduce_sum((self.out - self.input_x) ** 2, (1, 2, 3))
            # self.recon_loss = -tf.reduce_sum(self.input_x * tf.log(1e-8 + self.out) + (1 - self.input_x) * tf.log(1e-8 + 1 - self.out))
            # self.recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out, labels=self.input_x), (1, 2, 3))
            self.recon_loss = - utils.psnr(tf.reshape(self.input_x, shape=(-1, 45, 45, 1)), tf.reshape(self.out, shape=(-1, 45, 45, 1)))
            # self.recon_loss = tf.losses.mean_pairwise_squared_error(self.input_x, self.out)
            # self.kl_loss = 0.5 * tf.reduce_sum(1.0 + tf.log(self.var ** 2) - self.mean ** 2 - self.var ** 2, 1)
            self.kl_loss = - tf.reduce_sum(tf.log(e_out), 1)
            self.recon_loss = tf.reduce_mean(self.recon_loss)
            self.kl_loss = tf.reduce_mean(self.kl_loss)
            self.loss = self.recon_loss * self.alpha + self.kl_loss
            self.loss = tf.tuple([self.loss], control_inputs=tf.get_collection(tf.GraphKeys.UPDATE_OPS))[0]
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

    def train(self):
        # train_data, test_data = self.preprocess_mnist()
        train_data = utils.read_data_UCSD('UCSDped_patch/ped1', shuffle=True, reshape=False)
        # split train / validation
        validate_data = train_data[-128:]
        train_data = train_data[:-128]
        global_step = tf.Variable(0, trainable=False, name='global_step')
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step)
        # optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        # learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, 4000, 0.999, staircase=True)
        # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss, global_step)
        session_conf = tf.ConfigProto(
            allow_soft_placement=True)
        session_conf.gpu_options.allow_growth = True
        self.sess = tf.Session(config=session_conf)
        self.sess.run(tf.initialize_all_variables())
        for i in range(flags.epoch):
            recon_sum = 0
            kl_sum = 0
            with tf.device('/cpu:0'):
                batcher = utils.batch_iter(train_data, batch_size=self.batch_size, shuffle=True)
            for x in batcher:
                x = np.asarray(x)
                fetches = [train_op, self.recon_loss, self.kl_loss]
                fetch = self.sess.run(fetches, feed_dict={
                    self.input_x_: x, self.training: True
                })
                recon_sum += fetch[1]
                kl_sum += fetch[2]
                current_step = tf.train.global_step(self.sess, global_step)
                if current_step % 50 == 0:
                    print('epoch:%3d \t step:%d \t reon_loss:%.5f \t kl_loss:%.5f' % (i, current_step, recon_sum / 50, - kl_sum / 50))
                    recon_sum = 0
                    kl_sum = 0
                if current_step % 1000 == 0:
                    loss, kl_loss = self.sess.run([self.recon_loss, self.kl_loss], feed_dict={
                        self.input_x_: validate_data, self.training: False
                    })
                    print("Evaluation:")
                    print("loss:%.5f, kl_loss:%.5f" % (loss, kl_loss))
            self.test(test_data=validate_data, img_size=self.input_h, num_show=128)

    def test(self, test_data, img_size, num_show):
        recon = self.sess.run(self.out, feed_dict={
            self.input_x_: test_data, self.training: False
        })
        recon = recon[0:num_show]
        inputs = test_data[0:num_show]
        # inputs = np.squeeze(inputs, -1)
        recon = np.reshape(recon, (num_show, img_size, img_size))
        inputs = np.reshape(inputs, (num_show, img_size, img_size))
        scipy.misc.imsave('./generate.jpg', utils.montage(recon))
        scipy.misc.imsave('./inputs.jpg', utils.montage(inputs))

if __name__ == '__main__':
    vae = VAE(
        input_h=flags.input_size, input_w=flags.input_size, 
        batch_size=flags.batch_size, learning_rate=flags.learning_rate, alpha=flags.alpha, dataset_path=flags.dataset_path)
    vae.train()