import tensorflow as tf
import tensorflow.contrib.rnn as rnn
class Layers:
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)
    @staticmethod
    def conv2d(inputs, filters, kernel_size, stride, activation=None, name="conv2d"):
        with tf.variable_scope(name):
            in_dim = inputs.get_shape()[-1].value
            filter_shape = [kernel_size, kernel_size, in_dim, filters]
            W = tf.Variable(tf.random_normal(shape=filter_shape), name='W')
            b = tf.Variable(tf.constant(0.0, shape=[filters]), name='b')
            return Layers.forward(inputs, W, b, tf.nn.conv2d, stride, activation)
    @staticmethod
    def conv2d_transpose(inputs, filters, kernel_size, strides, padding, output_shape, activation=None, name="conv2d_transpose"):
        '''
            TODO:
                ddd
        '''
        with tf.name_scope(name):
            filter_shape = [kernel_size, kernel_size, filters, inputs.get_shape()[-1].value]
            W = tf.Variable(tf.random_normal(filter_shape), name="W")
            b = tf.Variable(tf.constant(0.0, shape=[filters]), name="b")
            out = tf.nn.conv2d_transpose(inputs, W, output_shape, strides=[1, strides, strides, 1])
            if activation is not None:
                return activation(tf.nn.bias_add(out, b))
            else:
                return tf.nn.bias_add(out, b)
    @staticmethod
    def dense(inputs, units, activation=None, name="dense"):
        with tf.variable_scope(name):
            in_dim = inputs.get_shape()[-1].value
            W = tf.Variable(xavier_init([in_dim, units]), name="W")
            b = tf.Variable(tf.constant(0.0, shape=[units]), name="b")
            return Layers.forward(inputs, W, b, tf.matmul, None, activation)
    @staticmethod
    def batch_norm(x, epsilon=1e-5, momentum=0.9, is_training=True):
        return tf.layers.batch_normalization(
            x, momentum=momentum, epsilon=epsilon, scale=True, is_training=is_training
        )
    @staticmethod
    def forward(inputs, W, b, op=None, stride=None, activation=None):
        if op is tf.nn.conv2d:
            out = op(inputs, W, strides=[1, stride, stride, 1])
        elif op is tf.matmul:
            out = op(inputs, W)
        out = tf.nn.bias_add(out, b)
        if activation:
            return activation(out)
        else:
            return out
    class RNN:
        def __init__(self, *args, **kwargs):
                return super().__init__(*args, **kwargs)
        @staticmethod
        def LSTM(inputs, num_units):
            with tf.name_scope("lstm"):
                lstms = []
                for _, unit in enumerate(num_units):
                    h = rnn.BasicLSTMCell(unit)
                    lstms.append(h)
                lstm = rnn.MultiRNNCell(lstms)
                init_state = lstm.zero_state(batch_size=tf.shape(inputs)[0], dtype=tf.float32)
                h, state = tf.nn.dynamic_rnn(cell=lstm, inputs=inputs, dtype=tf.float32, initial_state=init_state)
            return h[:, -1, :], state
class Utils:
    def __init__(self, *args, **kwargs):
                return super().__init__(*args, **kwargs)
    @staticmethod
    def sample(m, v):
        eps = tf.random_normal(shape=tf.shape(m))
        return m + eps * v
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)