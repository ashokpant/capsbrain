import numpy as np
import tensorflow as tf

from tfrecord import dataset_utils

flags = tf.app.flags

flags.DEFINE_string('dataset_dir', None, 'Dataset directory')

flags.DEFINE_float('test_ratio', 0.2, 'Validation dataset ratio')

flags.DEFINE_integer('num_shards', 1, 'Number of shards to split the TFRecord files')

flags.DEFINE_integer('random_seed', 1234, 'Random seed to use for repeatability.')
flags.DEFINE_multi_integer('image_size', [64, 64], 'Image size. [None, [width, height]])')

flags.DEFINE_string('tfrecord_file', None, 'Output TFRecord filename')
flags.DEFINE_boolean('force', True, 'Force recreate')

FLAGS = flags.FLAGS


def main():
    if not FLAGS.tfrecord_file:
        raise ValueError('tfrecord filename is empty.')

    if not FLAGS.dataset_dir:
        raise ValueError('Daraset is empty.')

    if not FLAGS.force and dataset_utils.dataset_exists(dataset_dir=FLAGS.dataset_dir, filename=FLAGS.tfrecord_file,
                                                        num_shards=FLAGS.num_shards):
        print('Dataset already created. Exiting ...')
        return

    files, id2labels, labels2id = dataset_utils.get_filenames_and_classes(FLAGS.dataset_dir)

    num_test = int(FLAGS.test_ratio * len(files))

    np.random.seed(FLAGS.random_seed)
    np.random.shuffle(files)
    train_files = files[num_test:]
    val_files = files[:num_test]

    # First, convert the training and validation sets.
    dataset_utils.convert_dataset('train', train_files,
                                  dataset_dir=FLAGS.dataset_dir, tfrecord_filename=FLAGS.tfrecord_file,
                                  num_shards=FLAGS.num_shards, size=FLAGS.image_size)
    dataset_utils.convert_dataset('test', val_files,
                                  dataset_dir=FLAGS.dataset_dir, tfrecord_filename=FLAGS.tfrecord_file,
                                  num_shards=FLAGS.num_shards, size=FLAGS.image_size)

    dataset_utils.write_label_file(id2labels, FLAGS.dataset_dir)

    print('\nDone!')


if __name__ == '__main__':
    FLAGS.dataset_dir = '/data/Datasets/face/att_faces/'
    FLAGS.tfrecord_file = 'att_faces'
    FLAGS.image_size = [28, 28]

    main()
