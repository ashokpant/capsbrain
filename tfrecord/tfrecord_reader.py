import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string('tfrecord_file', None, 'tfrecord_file to read')

flags.DEFINE_multi_integer('image_shape', [100, 100, 3], 'Image size. [None, [width, height, channel]])')

FLAGS = flags.FLAGS


def read_tfrecord(filename, image_shape, batch_size=16, num_threads = 8,  epoches = 1):

    if filename.__contains__('train'):
        split_name = 'train'
    else:
        split_name = 'test'
    k_image = split_name + '/image'
    k_id = split_name + '/id'
    k_width = split_name + '/width'
    k_height = split_name + '/height'
    k_channel = split_name + '/channel'

    feature = {k_image: tf.FixedLenFeature([], tf.string),
               k_id: tf.FixedLenFeature([], tf.int64),
               k_width: tf.FixedLenFeature([], tf.int64),
               k_height: tf.FixedLenFeature([], tf.int64),
               k_channel: tf.FixedLenFeature([], tf.int64),
               }
    filename_queue = tf.train.string_input_producer([filename], num_epochs=epoches)
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features[k_image], tf.float32)

    # Cast label data into int32
    label = tf.cast(features[k_id], tf.int32)
    # Reshape image data into the original shape
    # shape =tf.stack([features[k_width], features[k_height], features[k_channel]], axis=0)
    image = tf.reshape(image, image_shape)

    # # Creates batches by randomly shuffling tensors
    images, labels = tf.train.shuffle_batch([image, label], num_threads=num_threads, batch_size=batch_size,
                                            capacity=batch_size * 64,
                                            min_after_dequeue=batch_size * 32, allow_smaller_final_batch=False)

    return images, labels


def main():
    if not FLAGS.tfrecord_file:
        raise ValueError('tfrecord_filename is empty.')
    images, labels = read_tfrecord(FLAGS.tfrecord_file, image_shape=FLAGS.image_shape, batch_size=16, num_threads=4, epoches=1)

    with tf.Session() as sess:
        # Initialize all global and local variables
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        # Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for batch_index in range(5):
            img, lbl = sess.run([images, labels])
            img = img.astype(np.uint8)
            print(img.shape)
            for j in range(6):
                plt.subplot(2, 3, j + 1)
                plt.imshow(img[j, ...])
                plt.title(lbl)
            plt.show()
        # Stop the threads
        coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)
        sess.close()
    print('\nDone!')


if __name__ == '__main__':
    FLAGS.tfrecord_file = '/data/Datasets/face/att_faces/att_faces_test.tfrecords'
    FLAGS.image_shape = [28, 28, 3]
    main()

