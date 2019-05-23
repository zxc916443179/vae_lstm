import pickle
import numpy as np
import os
import cv2
from sklearn import utils
import tensorflow as tf
import math
# from copy import deepcopy
def montage(images, saveto='montage.png'):
    """
	Draw all images as a montage separated by 1 pixel borders.
    Also saves the file to the destination specified by `saveto`.
    """
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    if len(images.shape) == 4 and images.shape[3] == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 3)) * 0.5
    else:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1)) * 0.5
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                  1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
    # plt.imsave(arr=m, fname=saveto)
    return m
def batch_iter(data, batch_size, shuffle=False):
    '''
        generate a batch of training data
    '''
    data = np.array(data)
    data_size = len(data)
    num_batches = int((len(data) - 1) / batch_size + 1)
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, data_size)
        shuffled_data = data[start_index: end_index]
        if len(shuffled_data) < batch_size:
            continue
        if shuffle:
            shuffled_data = utils.shuffle(shuffled_data)
        yield shuffled_data

def read_data_UCSD(path, shuffle=False, training=True, reshape=True):
    data = []
    if training:
        dirs = os.listdir(os.path.join(path, 'train'))
        for d in dirs:
            for img_dir in os.listdir(os.path.join(path, 'train', d, 'box_img')):
                img = cv2.imread(os.path.join(path, 'train', d, 'box_img', img_dir))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if reshape:
                    img = img.flatten()
                img = cv2.normalize(img.astype(float), img.astype(float), alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
                data.append(img)
    else:
        test_dirs = sorted(os.listdir(os.path.join(path, 'test')))
        for test_dir in test_dirs:
            for img_dir in sorted(os.listdir(os.path.join(path, 'test', test_dir, 'box_img'))):
                img = cv2.imread(os.path.join(path, 'test', test_dir, 'box_img', img_dir))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if reshape:
                    img = img.flatten()
                img = cv2.normalize(img.astype(float), img.astype(float), alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
                data.append(img)
    print('total load data:%d' % len(data))
    if shuffle:
        data = utils.shuffle(data)
    return data

def generate_lstm_input(inputs, input_size=45, batch_size=128, time_steps=10, stride=2):
    inputs = tf.reshape(inputs, (-1, input_size * input_size))
    t = inputs
    print("inputs:{}".format(inputs.get_shape()))
    for i in range(stride):
        t = tf.concat([t, inputs], axis=0)
    print("t:{}".format(t.get_shape()))
    l = [t[0: 10]]
    for i in range(batch_size - 1):
        start_index =  (i + 1) * stride
        end_index = start_index + 10
        l.append(t[start_index: end_index])
    l = tf.convert_to_tensor(l)
    print("l:{}".format(l.get_shape()))
    return l

def psnr(im_true, im_test, max_val=1.0):
    target_data = np.array(im_true)
    
    ref_data = np.array(im_test)
 
    diff = ref_data - target_data
    diff = tf.reshape(diff, (128, -1))
    rmse = tf.sqrt(tf.reduce_mean(diff ** 2., 1))
    return 20 * (tf.log(max_val / rmse) / tf.log(10.0))

def read_pickle_data_UCSD(path, shuffle=False, reshape=True, width=45, height=45):
    opendataset = open(path, 'rb')
    dataset = []
    cnt = 0
    while True:
        try:
            tmp = pickle.load(opendataset)
            if reshape:
                tmp = np.reshape(tmp, (width, height))
            dataset.append(tmp)
            cnt += 1
        except Exception as e:
            print(e)
            break
    print('totally load data: %d' % cnt)
    opendataset.close()
    return dataset

def load_label(path, flatten=False):
    '''
        params:
            path: path to label's father dir
            flatten: flatten the label array
        return:
            labels -> ndarray
            cnt -> int


    '''
    label_dirs = sorted(os.listdir(path))
    labels = []
    cnt = 0
    for label_dir in label_dirs:
        if not os.path.isdir(os.path.join(path, label_dir)):
            continue
        print(label_dir)
        for label_file in sorted(os.listdir(os.path.join(path, label_dir))):
            label = np.fromfile(os.path.join(path, label_dir, label_file), dtype=int, sep='\n')
            labels.append(label)
            cnt += 1
    labels = np.asarray(labels)
    if flatten:
        label_t = []
        for label in labels:
            label_t = np.concatenate((label_t, label), -1)
        labels = label_t
    print('totally load %d' % cnt)
    return labels, cnt