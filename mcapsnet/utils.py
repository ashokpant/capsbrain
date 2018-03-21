import os
import scipy
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

import data.smallNORB as norb
from keras.datasets import cifar10, cifar100
from keras import backend as K

import logging
import daiquiri

from mcapsnet.config import cfg

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)


def load_mnist1(batch_size, is_training=True):
    path = os.path.join('data', 'mnist')
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((60000)).astype(np.int32)

        trX = trainX[:55000] / 255.
        trY = trainY[:55000]

        valX = trainX[55000:, ] / 255.
        valY = trainY[55000:]

        num_tr_batch = 55000 // batch_size
        num_val_batch = 5000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch


def load_fashion_mnist1(batch_size, is_training=True):
    path = os.path.join('data', 'fashion-mnist')
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((60000)).astype(np.int32)

        trX = trainX[:55000] / 255.
        trY = trainY[:55000]

        valX = trainX[55000:, ] / 255.
        valY = trainY[55000:]

        num_tr_batch = 55000 // batch_size
        num_val_batch = 5000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch


def save_images(imgs, size, path):
    '''
    Args:
        imgs: [batch_size, image_height, image_width]
        size: a list with tow int elements, [image_height, image_width]
        path: the path to save images
    '''
    imgs = (imgs + 1.) / 2  # inverse_transform
    return (scipy.misc.imsave(path, mergeImgs(imgs, size)))


def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs


# For version compatibility
def reduce_sum(input_tensor, axis=None, keepdims=False):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims)
    except:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims)


# For version compatibility
def softmax(logits, axis=None):
    try:
        return tf.nn.softmax(logits, axis=axis)
    except:
        return tf.nn.softmax(logits, dim=axis)


def create_inputs_norb(is_train: bool, epochs: int):
    import re
    if is_train:
        CHUNK_RE = re.compile(r"train\d+\.tfrecords")
    else:
        CHUNK_RE = re.compile(r"test\d+\.tfrecords")

    processed_dir = './data'
    chunk_files = [os.path.join(processed_dir, fname)
                   for fname in os.listdir(processed_dir)
                   if CHUNK_RE.match(fname)]

    image, label = norb.read_norb_tfrecord(chunk_files, epochs)

    if is_train:
        # TODO: is it the right order: add noise, resize, then corp?
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

        image = tf.image.resize_images(image, [48, 48])
        image = tf.random_crop(image, [32, 32, 1])
    else:
        image = tf.image.resize_images(image, [48, 48])
        image = tf.slice(image, [8, 8, 0], [32, 32, 1])

    x, y = tf.train.shuffle_batch([image, label], num_threads=cfg.num_threads, batch_size=cfg.batch_size,
                                  capacity=cfg.batch_size * 64,
                                  min_after_dequeue=cfg.batch_size * 32, allow_smaller_final_batch=False)

    return x, y


def create_inputs_mnist(is_train):
    tr_x, tr_y = load_mnist(is_train)
    data_queue = tf.train.slice_input_producer([tr_x, tr_y], capacity=64 * 8)
    x, y = tf.train.shuffle_batch(data_queue, num_threads=8, batch_size=cfg.batch_size, capacity=cfg.batch_size * 64,
                                  min_after_dequeue=cfg.batch_size * 32, allow_smaller_final_batch=False)
    x = tf.divide(x, 255.)
    x = slim.batch_norm(x, center=False, is_training=True, trainable=True)

    return x, y


def create_inputs_fashion_mnist(is_train):
    tr_x, tr_y = load_mnist(is_train)
    data_queue = tf.train.slice_input_producer([tr_x, tr_y], capacity=64 * 8)
    x, y = tf.train.shuffle_batch(data_queue, num_threads=8, batch_size=cfg.batch_size, capacity=cfg.batch_size * 64,
                                  min_after_dequeue=cfg.batch_size * 32, allow_smaller_final_batch=False)
    x = tf.divide(x, 255.)
    x = slim.batch_norm(x, center=False, is_training=True, trainable=True)
    return x, y


def create_inputs_cifar10(is_train):
    tr_x, tr_y = load_cifar10(is_train)
    tr_x = tr_x.astype(np.float32, copy=False)
    tr_y = tr_y.T[0]
    data_queue = tf.train.slice_input_producer([tr_x, tr_y], capacity=64 * 8)
    x, y = tf.train.shuffle_batch(data_queue, num_threads=8, batch_size=cfg.batch_size, capacity=cfg.batch_size * 64,
                                  min_after_dequeue=cfg.batch_size * 32, allow_smaller_final_batch=False)

    return x, y


def load_cifar10(is_training):
    # https://keras.io/datasets/
    assert (K.image_data_format() == 'channels_last')
    if is_training:
        return cifar10.load_data()[0]
    else:
        return cifar10.load_data()[1]


def create_inputs_cifar100(is_train):
    tr_x, tr_y = load_cifar100(is_train)
    tr_x = tr_x.astype(np.float32, copy=False)
    tr_y = tr_y.T[0]
    data_queue = tf.train.slice_input_producer([tr_x, tr_y], capacity=64 * 8)
    x, y = tf.train.shuffle_batch(data_queue, num_threads=8, batch_size=cfg.batch_size, capacity=cfg.batch_size * 64,
                                  min_after_dequeue=cfg.batch_size * 32, allow_smaller_final_batch=False)
    return x, y


def load_cifar100(is_training):
    # https://keras.io/datasets/
    # https://www.cs.toronto.edu/~kriz/cifar.html:
    # "Each image comes with a 'fine' label (the class to which it belongs)
    # and a 'coarse' label (the superclass to which it belongs)."
    assert (K.image_data_format() == 'channels_last')
    if is_training:
        return cifar100.load_data(label_mode='fine')[0]
    else:
        return cifar100.load_data(label_mode='fine')[1]


def load_mnist(is_training=True):
    path = os.path.join('data', 'mnist')
    fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

    fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.int32)

    fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float32)

    fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.int32)

    # normalization and convert to a tensor [60000, 28, 28, 1]
    # trX = tf.convert_to_tensor(trX, tf.float32)
    # teX = tf.convert_to_tensor(teX, tf.float32)

    # => [num_samples, 10]
    # trY = tf.one_hot(trY, depth=10, axis=1, dtype=tf.float32)
    # teY = tf.one_hot(teY, depth=10, axis=1, dtype=tf.float32)
    if is_training:
        return trX, trY
    else:
        return teX, teY


def load_fashion_mnist(is_training=True):
    path = os.path.join('data', 'fashion_mnist')
    fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

    fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.int32)

    fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float32)

    fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.int32)

    # normalization and convert to a tensor [60000, 28, 28, 1]
    # trX = tf.convert_to_tensor(trX, tf.float32)
    # teX = tf.convert_to_tensor(teX, tf.float32)

    # => [num_samples, 10]
    # trY = tf.one_hot(trY, depth=10, axis=1, dtype=tf.float32)
    # teY = tf.one_hot(teY, depth=10, axis=1, dtype=tf.float32)

    if is_training:
        return trX, trY
    else:
        return teX, teY


def load_data(dataset_name: str, is_train: bool):
    options = {'mnist': lambda: load_mnist(is_train),
               'fashion_mnist': lambda: load_fashion_mnist(is_train),
               'cifar10': lambda: load_cifar10(is_train),
               'cifa100': lambda: load_cifar100(is_train)}
    return options[dataset_name]()


def get_create_inputs(dataset_name: str, is_train: bool, epochs: int):
    options = {'mnist': lambda: create_inputs_mnist(is_train),
               'fashion_mnist': lambda: create_inputs_mnist(is_train),
               'smallNORB': lambda: create_inputs_norb(is_train, epochs),
               'cifar10': lambda: create_inputs_cifar10(is_train),
               'cifar100': lambda: create_inputs_cifar100(is_train)}
    return options[dataset_name]()


if __name__ == '__main__':
    x, y = get_create_inputs("mnist", False, 10)
    print(x, y)

    with tf.Session() as sess:
        threads = tf.train.start_queue_runners(sess=sess)
        sess.run(tf.global_variables_initializer())

        X, Y = sess.run(x, y)
        print(X, Y)
