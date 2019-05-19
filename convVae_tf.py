import os
import platform

import numpy as np
import scipy
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.examples.tutorials.mnist import input_data
from model import Layers, Utils
import utils
import sys
from error import ModeNotDefinedError
tf.flags.DEFINE_integer('epoch', 10000, "training epoches")
tf.flags.DEFINE_string('device', '0', 'cuda visible devices')
tf.flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
tf.flags.DEFINE_float('alpha', 0.2, 'alpha between kl_loss and recon_loss')
tf.flags.DEFINE_integer('batch_size', 128, 'batch size')
tf.flags.DEFINE_integer('input_size', 45, 'size of images(default:28*28)')
tf.flags.DEFINE_string('dataset_path', './UCSDped_patch/ped1', 'path to dataset')
tf.flags.DEFINE_bool('fixed_lr', True, 'whether to use fixed learning rate (default: False)')
tf.flags.DEFINE_bool('use_pickle', False, 'use image data directly or load data from pickle file (default: False)')
tf.flags.DEFINE_string('checkpoint_dir', None, 'directory to checkpoint when finetuning (default: None)')
tf.flags.DEFINE_string('mode', 'train', 'define the runing mode (options: train, finetune, default: train)')
flags = tf.flags.FLAGS

if 'Linux' in platform.system():
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.device
class VAE(object):
    def __init__(self, input_h, input_w, batch_size,
        learning_rate=0.01, alpha=0.2, dataset_path=None, use_pickle=False, checkpoint_dir=None, mode='train'):
        # training params
        self.input_h = input_h
        self.input_w = input_w
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        # dataset params
        self.dataset_path = dataset_path
        self.use_pickle = use_pickle
        #finetune params
        self.mode = mode
        self.checkpoint_dir = checkpoint_dir
        print(self.mode)
        print(type(self.mode))
        assert self.mode == 'finetune'
        assert self.mode is 'train'
        self.input_x_ = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_h, self.input_w], name="input_x")
        self.training = tf.placeholder(tf.bool, name="training")
        
        self.h, _ = Layers.RNN.LSTM(self.input_x_, num_units=[512, 256, 128])
        self.input_x = tf.expand_dims(self.input_x_, -1)


        h1 = Layers.conv2d(self.input_x, 128, 3, 1, activation=tf.nn.leaky_relu, padding='VALID') # 43 x 43 x 128
        with tf.name_scope('res_block_0'):
            r1 = Layers.conv2d(h1, 128, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            r1_bn = Layers.batch_norm(r1, is_training=self.training)
            r2 = Layers.conv2d(r1_bn, 128, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            r2_bn = Layers.batch_norm(r2, is_training=self.training)

            
            r3 = Layers.conv2d(h1, 128, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            r3_bn = Layers.batch_norm(r3, is_training=self.training)

            r = r2_bn + r3_bn
        # 43 x 43 x 128
        h2 = Layers.conv2d(r, 256, 5, 2, activation=tf.nn.leaky_relu, padding='VALID') # 20 x 20 x 256
        with tf.name_scope('res_block_1'):
            r1 = Layers.conv2d(h2, 256, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            r1_bn = Layers.batch_norm(r1, is_training=self.training)
            r2 = Layers.conv2d(r1_bn, 256, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            r2_bn = Layers.batch_norm(r2, is_training=self.training)

            
            r3 = Layers.conv2d(h2, 256, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            r3_bn = Layers.batch_norm(r3, is_training=self.training)

            r = r2_bn + r3_bn
        
        h3 = Layers.conv2d(r, 256, 3, 1, activation=tf.nn.leaky_relu, padding='VALID') # 18 x 18 x 256
        with tf.name_scope('res_block_2'):
            r1 = Layers.conv2d(h3, 256, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            r1_bn = Layers.batch_norm(r1, is_training=self.training)
            r2 = Layers.conv2d(r1_bn, 256, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            r2_bn = Layers.batch_norm(r2, is_training=self.training)

            
            r3 = Layers.conv2d(h3, 256, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            r3_bn = Layers.batch_norm(r3, is_training=self.training)

            r = r2_bn + r3_bn

        # 18 x 18 x 256
        h4 = Layers.conv2d(r, 128, 3, 1, activation=tf.nn.leaky_relu, padding='VALID') # 16 x 16 x 128
        with tf.name_scope('res_block_3'):
            r1 = Layers.conv2d(h4, 128, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            r1_bn = Layers.batch_norm(r1, is_training=self.training)
            r2 = Layers.conv2d(r1_bn, 128, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            r2_bn = Layers.batch_norm(r2, is_training=self.training)

            
            r3 = Layers.conv2d(h4, 128, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            r3_bn = Layers.batch_norm(r3, is_training=self.training)

            r = r2_bn + r3_bn

        # 16 x 16 x 128
        
        h = Layers.conv2d(r, 1, 1, 1, activation=tf.nn.sigmoid, padding="VALID")
        h = tf.reshape(tf.squeeze(h, -1), (self.batch_size, 16 * 16))
        self.mean = Layers.dense(h , 128, activation=tf.nn.sigmoid)
        # self.mean = Layers.res_block(mean, 128, fn=Layers.dense, is_training=self.training)
        self.var = Layers.dense(h, 128, activation=tf.nn.sigmoid)
        # self.var = Layers.res_block(var, 128, fn=Layers.dense, is_training=self.training)
        sampled = Utils.sample(self.mean, self.var)
        sampled = tf.expand_dims(tf.reshape(tf.concat(values=[sampled, self.h], axis=1), (self.batch_size, 16, 16)), axis=-1)
        
        # estimator
        # e1 = Layers.dense(self.mean, 128, activation=tf.nn.leaky_relu)
        # e2 = Layers.dense(e1, 256, activation=tf.nn.leaky_relu)
        # e3 = Layers.dense(e2, 128, activation=tf.nn.leaky_relu)
        # e_out = Layers.dense(e3, 1, activation=tf.nn.sigmoid)

        with tf.name_scope('res_block_4'):
            d1 = Layers.conv2d(sampled, 128, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            d1_bn = Layers.batch_norm(d1, is_training=self.training)
            d2 = Layers.conv2d(d1_bn, 128, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            d2_bn = Layers.batch_norm(d2, is_training=self.training)

            
            d3 = Layers.conv2d(sampled, 128, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            d3_bn = Layers.batch_norm(d3, is_training=self.training)

            d = d3_bn + d2_bn
        
        o1 = Layers.conv2d_transpose(d, 256, 3, 1, padding='VALID', output_shape=[128, h3.get_shape()[1].value, h3.get_shape()[2].value, h3.get_shape()[3].value], activation=tf.nn.leaky_relu)
        with tf.name_scope('res_block_5'):
            d1 = Layers.conv2d(o1, 256, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            d1_bn = Layers.batch_norm(d1, is_training=self.training)
            d2 = Layers.conv2d(d1_bn, 256, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            d2_bn = Layers.batch_norm(d2, is_training=self.training)

            
            d3 = Layers.conv2d(o1, 256, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            d3_bn = Layers.batch_norm(d3, is_training=self.training)

            d = d3_bn + d2_bn
        # 18 x 18 x 256
        o2 = Layers.conv2d_transpose(d, 256, 3, 1, padding='VALID', output_shape=[128, h2.get_shape()[1].value, h2.get_shape()[2].value, h2.get_shape()[3].value], activation=tf.nn.leaky_relu)
        with tf.name_scope('res_block_6'):
            d1 = Layers.conv2d(o2, 256, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            d1_bn = Layers.batch_norm(d1, is_training=self.training)
            d2 = Layers.conv2d(d1_bn, 256, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            d2_bn = Layers.batch_norm(d2, is_training=self.training)

            
            d3 = Layers.conv2d(o2, 256, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            d3_bn = Layers.batch_norm(d3, is_training=self.training)

            d = d3_bn + d2_bn
        # 20 x 20 x 256
        o3 = Layers.conv2d_transpose(d, 128, 5, 2, padding='VALID', output_shape=[128, h1.get_shape()[1].value, h1.get_shape()[2].value, h1.get_shape()[3].value], activation=tf.nn.leaky_relu)
        with tf.name_scope('res_block_7'):
            d1 = Layers.conv2d(o3, 128, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            d1_bn = Layers.batch_norm(d1, is_training=self.training)
            d2 = Layers.conv2d(d1_bn, 128, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            d2_bn = Layers.batch_norm(d2, is_training=self.training)

            
            d3 = Layers.conv2d(o3, 128, 3, 1, activation=tf.nn.leaky_relu, padding="SAME")
            d3_bn = Layers.batch_norm(d3, is_training=self.training)

            d = d3_bn + d2_bn
        # 43 x 43 x 128
        self.out = Layers.conv2d_transpose(d, 1, 3, 1, padding="VALID", output_shape=[128, self.input_x.get_shape()[1].value, self.input_x.get_shape()[2].value, self.input_x.get_shape()[3].value], activation=tf.nn.sigmoid)
        # 45 x 45 x 1
        
        self.h = [h1, h2, h3, h4, self.var, o1, o2, o3, self.out]
        
        with tf.name_scope('score'):
            # self.recon_loss = tf.reduce_sum((self.out - self.input_x) ** 2, (1, 2, 3))
            # self.recon_loss = -tf.reduce_sum(self.input_x * tf.log(1e-8 + self.out) + (1 - self.input_x) * tf.log(1e-8 + 1 - self.out))
            # self.recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out, labels=self.input_x), (1, 2, 3))
            self.recon_loss = - utils.psnr(tf.reshape(self.input_x, shape=(-1, 45, 45, 1)), tf.reshape(self.out, shape=(-1, 45, 45, 1)))
            # self.recon_loss = tf.losses.mean_pairwise_squared_error(self.input_x, self.out)
            self.kl_loss = - 0.5 * tf.reduce_sum(1.0 + tf.log(self.var ** 2) - self.mean ** 2 - self.var ** 2, 1)
            # self.kl_loss = - tf.reduce_sum(tf.log(e_out), 1)
            self.recon_loss = tf.reduce_mean(self.recon_loss)
            self.kl_loss = tf.reduce_mean(self.kl_loss)
            self.loss = self.recon_loss * self.alpha + self.kl_loss
            self.loss = tf.tuple([self.loss], control_inputs=tf.get_collection(tf.GraphKeys.UPDATE_OPS))[0]

    def train(self):
        # train_data, test_data = self.preprocess_mnist()
        if not self.use_pickle:
            train_data = utils.read_data_UCSD(self.dataset_path, shuffle=True, reshape=False)
        else:
            train_data = utils.read_pickle_data_UCSD(self.dataset_path, shuffle=False)
        # split train / validation
        validate_data = train_data[-128:]
        train_data = train_data[:-128]
        global_step = tf.Variable(0, trainable=False, name='global_step')
        # optimizer definition
        if flags.fixed_lr:
            learning_rate = self.learning_rate
        else:
            learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, 4000, 0.999, staircase=True)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step)
        # optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss, global_step)
        session_conf = tf.ConfigProto(
            allow_soft_placement=True)
        session_conf.gpu_options.allow_growth = True
        print('start session')
        self.sess = tf.Session(config=session_conf)
        if self.mode is 'finetune':
            print('loading checkpint from %s' % (self.checkpoint_dir))
            checkpoint_file = tf.train.latest_checkpoint(self.checkpoint_dir)
            saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
            saver.restore(self.sess, checkpoint_file)
            print('load success')
        elif self.mode is 'train':
            print('training')
            self.sess.run(tf.initialize_all_variables())
            # Saver
            saver = tf.train.Saver(max_to_keep=10, var_list=tf.global_variables())
        else:
            print('given mode not found, expect (train or finetune) but get %s' % self.mode)
            pass
        recon_sum = 0
        kl_sum = 0
        previous_recon = 0
        ckpt_dir = 'ckpt_lr_%f_alpha_%f/model.ckpt' % (self.learning_rate, self.alpha) if self.mode is 'train' else 'ckpt_lr_%f_alpha_%f_finetune/model.ckpt' % (self.learning_rate, self.alpha)
        for i in range(flags.epoch):
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
                    print('epoch:%3d \t step:%d \t reon_loss:%.5f \t kl_loss:%.5f' % (i, current_step, recon_sum / 50, kl_sum / 50))
                    recon_sum = 0
                    kl_sum = 0
                if current_step % 1000 == 0:
                    loss, kl_loss = self.sess.run([self.recon_loss, self.kl_loss], feed_dict={
                        self.input_x_: validate_data, self.training: False
                    })
                    print("Evaluation:")
                    print("loss:%.5f, kl_loss:%.5f" % (loss, kl_loss))
                    if previous_recon < fetch[1]:
                        saver.save(self.sess, ckpt_dir, global_step=global_step)
                        print('model is saved to %s \t current psnr loss is: %.5f' % (ckpt_dir, fetch[1]))
                        previous_recon = fetch[1]
            self.test(test_data=validate_data, img_size=self.input_h, num_show=128)

    def test(self, test_data, img_size, num_show):
        recon = self.sess.run(self.out, feed_dict={
            self.input_x_: test_data, self.training: False
        })
        recon = recon[:num_show]
        inputs = test_data[:num_show]
        # inputs = np.squeeze(inputs, -1)
        recon = np.reshape(recon, (num_show, img_size, img_size))
        inputs = np.reshape(inputs, (num_show, img_size, img_size))
        scipy.misc.imsave('./generate.jpg', utils.montage(recon))
        scipy.misc.imsave('./inputs.jpg', utils.montage(inputs))

    def save_model(self):
        '''
            TODO:
                save model
        '''
        pass
    
if __name__ == '__main__':
    vae = VAE(
        input_h=flags.input_size, input_w=flags.input_size, 
        batch_size=flags.batch_size, learning_rate=flags.learning_rate, alpha=flags.alpha, dataset_path=flags.dataset_path, use_pickle=flags.use_pickle,
        checkpoint_dir=flags.checkpoint_dir, mode=flags.mode)
    try:
        vae.train()
    except ModeNotDefinedError as e:
        print(e)