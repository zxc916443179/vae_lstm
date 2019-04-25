import numpy as np
import os
import cv2
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
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(end_index - start_index))
            shuffled_data = shuffled_data[shuffle_indices]
        yield shuffled_data

def read_data_UCSD(path):
    train_data = []
    test_data = []
    train_dirs = os.listdir(os.path.join(path, 'train'))
    for train_dir in train_dirs:
        for img_dir in os.listdir(os.path.join(path, 'train', train_dir)):
            img = cv2.imread(img_dir)
            train_data.append(img)

    test_dirs = os.listdir(os.path.join(path, 'test'))
    for test_dir in test_dirs:
        for img_dir in os.listdir(os.path.join(path, 'test', test_dir)):
            img = cv2.imread(img_dir)
            test_data.append(img)
    print('train data length:%d' % len(train_data))
    print('test data length:%d' % len(test_data))
    return train_data, test_data