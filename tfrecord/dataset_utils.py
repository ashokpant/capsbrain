"""
-- User: Ashok Kumar Pant (asokpant@gmail.com)
-- Date: 3/23/18
-- Time: 11:33 AM
"""

import math
import os
import sys

import numpy as np
import tensorflow as tf
from boto.kinesis.exceptions import InvalidArgumentException

slim = tf.contrib.slim
import cv2

# State the labels filename
LABELS_FILENAME = 'labels.txt'


# ===================================================  Dataset Utils  ===================================================

def int64_feature(values):
    """Returns a TF-Feature of int64s.
  Args:
    values: A scalar or list of values.
  Returns:
    a TF-Feature.
  """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    """Returns a TF-Feature of bytes.
  Args:
    values: A string.
  Returns:
    a TF-Feature.
  """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(image_data, image_format, height, width, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image/class/label': int64_feature(class_id),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
    }))


def write_label_file(labels_to_class_names, dataset_dir,
                     filename=LABELS_FILENAME):
    """Writes a file with the list of class names.
  Args:
    labels_to_class_names: A map of (integer) labels to class names.
    dataset_dir: The directory in which the labels file should be written.
    filename: The filename where the class names are written.
  """
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))


def has_labels(dataset_dir, filename=LABELS_FILENAME):
    """Specifies whether or not the dataset directory contains a label map file.
  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.
  Returns:
    `True` if the labels file exists and `False` otherwise.
  """
    return tf.gfile.Exists(os.path.join(dataset_dir, filename))


def read_label_file(dataset_dir, filename=LABELS_FILENAME):
    """Reads the labels file and returns a mapping from ID to class name.
  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.
  Returns:
    A map from a label (integer) to class name.
  """
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'r') as f:
        lines = f.read().decode()
    lines = lines.split('\n')
    lines = filter(None, lines)

    labels_to_class_names = {}
    for line in lines:
        index = line.index(':')
        labels_to_class_names[int(line[:index])] = line[index + 1:]
    return labels_to_class_names


# =======================================  Conversion Utils  ===================================================

# Create an image reader object for easy reading of the images
class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def get_filenames_and_classes(dataset_dir, max_classes=0, min_samples_per_class=0):
    """Returns a list of filenames and inferred class names.
  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.
  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
    folders = [name for name in os.listdir(dataset_dir) if
               os.path.isdir(os.path.join(dataset_dir, name))]

    if len(folders) == 0:
        raise ValueError(dataset_dir+ " does not contain valid sub directories.")
    directories = []
    for folder in folders:
        directories.append(os.path.join(dataset_dir, folder))

    folders = sorted(folders)
    # label2id = dict(zip(folders, range(len(folders))))
    label2id = {}

    i = 0
    c = 1
    total_files = []
    for folder in folders:
        dir = os.path.join(dataset_dir, folder)
        files = os.listdir(dir)
        if min_samples_per_class > 0 and len(files) < min_samples_per_class:
            continue

        for file in files:
            path = os.path.join(dir, file)
            total_files.append([path, i])
        label2id[folder] = i
        i += 1

        if 0 < max_classes <= c:
            break
        c += 1

    id2label = {v: k for k, v in label2id.items()}
    print("Number of classes: {}".format(c))
    return np.array(total_files), id2label, label2id


def get_dataset_filename(dataset_dir, split_name, shard_id, tfrecord_filename, num_shards):
    if num_shards > 1:
        filename = '{}_{}_{}_of_{}.tfrecords'.format(tfrecord_filename, split_name, shard_id, num_shards)
    else:
        filename = '{}_{}.tfrecords'.format(tfrecord_filename, split_name)
    return os.path.join(dataset_dir, filename)


def convert_dataset(split_name, filenames, dataset_dir, tfrecord_filename, num_shards, size=None):
    """Converts the given filenames to a TFRecord dataset.
  Args:
    split_name: The name of the dataset, either 'train' or 'test'.
    filenames: A list of absolute paths to png or jpg images.
    labels2id: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
    assert split_name in ['train', 'test']

    num_per_shard = int(math.ceil(len(filenames) / float(num_shards)))

    # with tf.Graph().as_default():
    # with tf.Session('') as sess:
    for shard_id in range(num_shards):
        output_filename = get_dataset_filename(
            dataset_dir, split_name, shard_id, tfrecord_filename=tfrecord_filename, num_shards=num_shards)
        print("Writing {} dataset to file: {}".format(split_name, output_filename))
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
            for i in range(start_ndx, end_ndx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                    i + 1, len(filenames), shard_id))
                sys.stdout.flush()

                # Read the filename:
                image = imread(filenames[i][0], size=size)
                label = int(filenames[i][1])
                shape = np.shape(image)
                width = shape[1]
                height = shape[0]
                if len(shape) == 3:
                    channel = shape[2]
                else:
                    channel = 1
                # Create a feature
                feature = {split_name + '/id': int64_feature(label),
                           split_name + '/width': int64_feature(width),
                           split_name + '/height': int64_feature(height),
                           split_name + '/channel': int64_feature(channel),
                           split_name + '/image': bytes_feature(tf.compat.as_bytes(image.tostring()))}
                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def dataset_exists(dataset_dir, filename, num_shards):
    for split_name in ['train', 'test']:
        for shard_id in range(num_shards):
            tfrecord_filename = get_dataset_filename(
                dataset_dir, split_name, shard_id, filename, num_shards)
            if not tf.gfile.Exists(tfrecord_filename):
                return False
    return True


def imread(filename, size=None):
    img = cv2.imread(filename)
    if size is not None:
        img = cv2.resize(img, (size[0], size[1]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img


if __name__ == '__main__':
    dir = '/data/Datasets/face/att_faces/'
    files, id2label, label2id = get_filenames_and_classes(dir)
    print(files, id2label, label2id)
