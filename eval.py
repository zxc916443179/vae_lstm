import tensorflow as tf
import os, platform
from model import Utils
import utils
import numpy as np
import scipy
import sys
import auroc
tf.flags.DEFINE_string("dataset_path", './UCSDped_patch/ped1', "dataset path")
tf.flags.DEFINE_string("checkpoint_dir", "none", "loading latest checkpoint")
tf.flags.DEFINE_string('label_dir', './label/label15.p', "dir of label")
tf.flags.DEFINE_integer('batch_size', 128, 'batch size')
tf.flags.DEFINE_string('device', '0', 'cuda visible devices')
tf.flags.DEFINE_bool('train', False, 'use train(True) or test(False)')
flags = tf.flags.FLAGS
if 'Linux' in platform.system():
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.device
def main(argv=None):
    graph = tf.Graph()
    f = open('loss.log', 'w')
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            print('loading checkpoint in dir: %s' % flags.checkpoint_dir)
            checkpoint_file = tf.train.latest_checkpoint(flags.checkpoint_dir)
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            print('load success')
            # for operation in graph.get_operations():
            #     print(operation)
            psnr = graph.get_operation_by_name('score/Mean').outputs[0]
            
            kl = graph.get_operation_by_name('score/Mean_1').outputs[0]
            input_x = graph.get_operation_by_name('input_x').outputs[0]
            training = graph.get_operation_by_name('training').outputs[0]
            data = utils.read_data_UCSD(flags.dataset_path, shuffle=False, reshape=False, training=flags.train)
            out = graph.get_operation_by_name('conv2d_transpose_2/Sigmoid').outputs[0]
            # input_x_ = graph.get_operation_by_name('score/Reshape').outputs[0]
            # loss = graph.get_operation_by_name('score/mul_2/y').outputs[0] * graph.get_operation_by_name('score/Neg').outputs[0] + graph.get_operation_by_name('score/mul_1').outputs[0]
            # recon = tf.reduce_sum(tf.square(input_x_ - out), (1, 2, 3))
            recon = graph.get_operation_by_name('score/Sum').outputs[0]
            score = []
            out_all = []
            for batch in utils.batch_iter(data, 128, shuffle=False):
                x = np.asarray(batch)
                if len(x) < 128:
                    continue
                psnr_loss, kl_loss, recon_loss, recon_out = sess.run([psnr, kl, recon, out], feed_dict={
                    input_x: x, training: True
                })
                log = 'psnr:%.5f \t kl:%.5f \t recon:%.5f' % (psnr_loss, kl_loss, np.mean(recon_loss))
                print(log)
                score = np.concatenate((score, recon_loss), -1)
                # print(recon_out.shape())
                out_all = np.concatenate((out_all, recon_out))
            fpr, tpr, threshold, acc = auroc.auroc(score, flags.label_dir)
            print(acc)
            print(threshold)
            # auroc.plot_roc(fpr, tpr)
            fpr = np.array(fpr)
            tpr = np.array(tpr)
            fpr.tofile('fpr.bin')
            tpr.tofile('tpr.bin')

            psnr_loss, kl_loss, recon_loss, recon_out = sess.run([psnr, kl, recon, out], feed_dict={
                input_x: x, training: True
            })
            log = 'psnr:%.5f \t kl:%.5f \t recon:%.5f' % (psnr_loss, kl_loss, np.mean(recon_loss))
                
            f.writelines(str(i) + '\n' for i in recon_loss)
            print(log)
            scipy.misc.imsave('./input.jpg', utils.montage(x))
            recon_out = np.squeeze(recon_out, -1)
            scipy.misc.imsave('./generate.jpg', utils.montage(recon_out))

if __name__ == '__main__':
    tf.app.run()