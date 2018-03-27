import logging
import os

import cv2
import daiquiri
import data.smallNORB as norb
import numpy as np
import scipy
import tensorflow as tf
from keras import backend as K
from keras.datasets import cifar10, cifar100
from scipy.misc import imread
from sklearn.externals.joblib import Parallel, delayed
from tensorflow.contrib import slim

from mcapsnet.config import cfg
from tfrecord.tfrecord_reader import read_tfrecord

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
        CHUNK_RE = re.compile(r"train\d+\.tfrecord")
    else:
        CHUNK_RE = re.compile(r"test\d+\.tfrecord")

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
    x, y = tf.train.shuffle_batch(data_queue, num_threads=cfg.num_threads, batch_size=cfg.batch_size, capacity=cfg.batch_size * 64,
                                  min_after_dequeue=cfg.batch_size * 32, allow_smaller_final_batch=False)
    # x = tf.divide(x, 255.)
    x = slim.batch_norm(x, center=False, is_training=True, trainable=True)

    return x, y


def create_inputs_fashion_mnist(is_train):
    tr_x, tr_y = load_mnist(is_train)
    data_queue = tf.train.slice_input_producer([tr_x, tr_y], capacity=64 * 8)
    x, y = tf.train.shuffle_batch(data_queue, num_threads=cfg.num_threads, batch_size=cfg.batch_size, capacity=cfg.batch_size * 64,
                                  min_after_dequeue=cfg.batch_size * 32, allow_smaller_final_batch=False)
    # x = tf.divide(x, 255.)
    x = slim.batch_norm(x, center=False, is_training=True, trainable=True)
    return x, y


def create_inputs_cifar10(is_train):
    tr_x, tr_y = load_cifar10(is_train)
    tr_x = tr_x.astype(np.float32, copy=False)
    tr_y = tr_y.T[0]

    data_queue = tf.train.slice_input_producer([tr_x, tr_y], capacity=64 * 8)
    x, y = tf.train.shuffle_batch(data_queue, num_threads=cfg.num_threads, batch_size=cfg.batch_size, capacity=cfg.batch_size * 64,
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
    x, y = tf.train.shuffle_batch(data_queue, num_threads=cfg.num_threads, batch_size=cfg.batch_size, capacity=cfg.batch_size * 64,
                                  min_after_dequeue=cfg.batch_size * 32, allow_smaller_final_batch=False)
    return x, y


def create_inputs_att_faces(is_train, size):
    tr_x, tr_y = load_att_faces(is_train, size)
    tr_x = tr_x.astype(np.float32, copy=False)
    data_queue = tf.train.slice_input_producer([tr_x, tr_y], capacity=64 * 8)
    x, y = tf.train.shuffle_batch(data_queue, num_threads=cfg.num_threads, batch_size=cfg.batch_size, capacity=cfg.batch_size * 64,
                                  min_after_dequeue=cfg.batch_size * 32, allow_smaller_final_batch=False)
    return x, y


def create_inputs_att_faces_from_tfrecords(is_training):
    path = os.path.join('data', 'att_faces')
    if is_training:
        filename = os.path.join(path, 'att_faces_train.tfrecords')
        shape = [cfg.input_size, cfg.input_size, cfg.input_channel]
        x, y = read_tfrecord(filename=filename, image_shape=shape, batch_size=cfg.batch_size, num_threads=cfg.num_threads, epoches=cfg.epoch)
    else:
        filename = os.path.join(path, 'att_faces_test.tfrecords')
        shape = [cfg.input_size, cfg.input_size, cfg.input_channel]
        x, y = read_tfrecord(filename=filename, image_shape=shape, batch_size=cfg.batch_size, num_threads=cfg.num_threads, epoches=cfg.epoch)

    x = slim.batch_norm(x, center=False, is_training=True, trainable=True)
    return x, y


def load_att_faces(is_training, size=None):
    path = os.path.join('data', 'att_faces')
    labels = read_file(os.path.join(path, 'labels.txt'), sep=':')

    if is_training:
        filename = os.path.join(path, 'train.txt')
        X, Y = read_file(filename=filename)
        Y = Y.astype(np.int32)
        X_ = read_images(X, size=size)
        return np.array(X_), Y
    else:
        filename = os.path.join(path, 'test.txt')
        X, Y = read_file(filename=filename)
        Y = Y.astype(np.int32)
        X_ = read_images(X, size=size)
        return np.array(X_), Y


def create_inputs_casia_faces(is_training):
    if cfg.dataset_dir is not None:
        path = cfg.dataset_dir
    else:
        path = os.path.join('data', 'casia')
    if is_training:
        filename = os.path.join(path, 'casia_train.tfrecords')
        shape = [cfg.input_size, cfg.input_size, cfg.input_channel]
        x, y = read_tfrecord(filename=filename, image_shape=shape, batch_size=cfg.batch_size, num_threads=cfg.num_threads, epoches=cfg.epoch)
    else:
        filename = os.path.join(path, 'casia_test.tfrecords')
        shape = [cfg.input_size, cfg.input_size, cfg.input_channel]
        x, y = read_tfrecord(filename=filename, image_shape=shape, batch_size=cfg.batch_size, num_threads=cfg.num_threads, epoches=cfg.epoch)

    x = slim.batch_norm(x, center=False, is_training=True, trainable=True)
    return x, y


def read_file(filename, sep=' '):
    X = []
    Y = []
    f = open(filename)
    for line in f.readlines():
        tokens = line.split(sep)
        X.append(tokens[0])
        Y.append(tokens[1])
    return np.array(X), np.array(Y)


def read_images(image_list, size=None):
    images = Parallel(n_jobs=4, verbose=5)(
        delayed(imread)(f, size) for f in image_list
    )
    return images


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


def get_create_inputs(dataset_name: str, is_train: bool, epochs: int, size=None):
    options = {'mnist': lambda: create_inputs_mnist(is_train),
               'fashion_mnist': lambda: create_inputs_mnist(is_train),
               'smallNORB': lambda: create_inputs_norb(is_train, epochs),
               'cifar10': lambda: create_inputs_cifar10(is_train),
               'cifar100': lambda: create_inputs_cifar100(is_train),
               'att_faces': lambda: create_inputs_att_faces(is_train, size=size),
               # 'att_faces': lambda: create_inputs_att_faces_from_tfrecords(is_train),
               'casia': lambda: create_inputs_casia_faces(is_train)}
    return options[dataset_name]()


def imread(filename, size=None):
    try:
        image = cv2.imread(filename)[:, :, [2, 1, 0]].astype(np.float32)
        if size is not None:
            image = imresize(image, size)
        return image
    except Exception as e:
        logger.error("Unable to read image: {}, {}".format(filename, e))
        return None


def imresize(image, size=None):
    if size is None:
        return image
    else:
        return cv2.resize(image, size)


def bgr2gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def show_image(image, text=None, pause=0):
    emmi = None
    if text is not None:
        emmi = np.zeros_like(image)
        add_text(img=emmi, text=text, text_top=np.int32(emmi.shape[0] / 2), text_left=np.int32(emmi.shape[1] / 2),
                 image_scale=1)
    stack = np.hstack((image, np.ones(shape=image.shape), emmi))
    cv2.imshow('Output', stack)
    cv2.waitKey(pause)
    # c = cv2.waitKey(30) & 0xff
    # if c == 27 or c == 113:
    #     cv2.destroyAllWindows()


def add_text(img, text, text_top, text_left=0, image_scale=1):
    """
    Args:
        img (numpy array of shape (width, height, 3): input image
        text (str): text to add to image
        text_top (int): location of top text to add
        image_scale (float): image resize scale

    Summary:
        Add display text to a frame.

    Returns:
        Next available location of top text (allows for chaining this function)
    """
    cv2.putText(
        img=img,
        text=text,
        org=(text_left, text_top),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.45 * image_scale,
        color=(255, 255, 255))
    return text_top + int(5 * image_scale)


if __name__ == '__main__':
    x, y = get_create_inputs("att_faces", False, 10)
    print(x, y)

    with tf.Session() as sess:
        threads = tf.train.start_queue_runners(sess=sess)
        sess.run(tf.global_variables_initializer())

        X, Y = sess.run(x, y)
        print(X, Y)
