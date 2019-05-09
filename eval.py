import tensorflow as tf
import os, platform
from model import Utils
import utils
import numpy as np
tf.flags.DEFINE_string("dataset_path", './UCSDped_patch/ped1', "dataset path")
tf.flags.DEFINE_integer("max", 10000, "max number of dataset")
tf.flags.DEFINE_string("checkpoint_dir", "none", "loading latest checkpoint")
tf.flags.DEFINE_string('label_dir', './label/label15.p', "dir of label")
tf.flags.DEFINE_integer('batch_size', 128, 'batch size')
tf.flags.DEFINE_string('device', '0', 'cuda visible devices')

flags = tf.flags.FLAGS
if 'Linux' in platform.system():
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.device

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
        psnr = graph.get_operation_by_name('score/Mean_1').outputs[0]
        
        kl = graph.get_operation_by_name('score/Mean_2').outputs[0]
        input_x = graph.get_operation_by_name('input_x').outputs[0]
        training = graph.get_operation_by_name('training').outputs[0]
        data = utils.read_data_UCSD(flags.dataset_path, shuffle=True, reshape=False)

        for batch in utils.batch_iter(data, 128, shuffle=True):
            x = np.asarray(batch)
            psnr_loss, kl_loss = sess.run([psnr, kl], feed_dict={
                input_x: x, training: False
            })
            log = 'psnr:%.5f \t kl:%.5f' % (psnr_loss, kl_loss)
            f.writelines(log)
            print(log)
