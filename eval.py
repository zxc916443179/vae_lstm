import tensorflow as tf
from model import Utils

tf.flags.DEFINE_string("dataset_path", './UCSDped_patch/ped1', "dataset path")
tf.flags.DEFINE_integer("max", 10000, "max number of dataset")
tf.flags.DEFINE_string("checkpoint_dir", "none", "loading latest checkpoint")
tf.flags.DEFINE_string('label_dir', './label/label15.p', "dir of label")
tf.flags.DEFINE_integer('batch_size', 128, 'batch size')

flags = tf.flags.FLAGS

graph = tf.Graph()
graph.as_default()
sess = tf.Session()
with sess.as_default():
    print('loading checkpoint in dir: %s' % flags.checkpoint_dir)
    checkpoint_file = tf.train.latest_checkpoint(flags.checkpoint_dir)
    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
    saver.restore(sess, checkpoint_file)
    print('load success')
    print(graph.get_all_collection_keys())