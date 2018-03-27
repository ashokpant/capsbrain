import daiquiri
import tensorflow as tf
import numpy as np
from mcapsnet.config import cfg
from mcapsnet.layers import conv2d, primary_caps, conv_capsule, class_capsules

slim = tf.contrib.slim
logger = daiquiri.getLogger(__name__)


def capsules_net(inputs, num_classes, iterations, batch_size, name='capsule_em'):
    """Define the Capsule Network model
    """
    with tf.variable_scope(name) as scope:
        # ReLU Conv1
        # Images shape (24, 28, 28, 1) -> conv 5x5 filters, 32 output channels, strides 2 with padding, ReLU
        # nets -> (?, 14, 14, 32)
        nets = conv2d(
            inputs,
            kernel=5, out_channels=32, stride=2, padding='SAME',
            activation_fn=tf.nn.relu, name='relu_conv1'
        )

        # PrimaryCaps
        # (?, 14, 14, 32) -> capsule 1x1 filter, 32 output capsule, strides 1 without padding
        # nets -> (poses (?, 14, 14, 32, 4, 4), activations (?, 14, 14, 32))
        nets = primary_caps(
            nets,
            kernel_size=1, out_capsules=32, stride=1, padding='VALID',
            pose_shape=[4, 4], name='primary_caps'
        )

        # ConvCaps1
        # (poses, activations) -> conv capsule, 3x3 kernels, strides 2, no padding
        # nets -> (poses (24, 6, 6, 32, 4, 4), activations (24, 6, 6, 32))
        nets = conv_capsule(
            nets, shape=[3, 3, 32, 32], strides=[1, 2, 2, 1], iterations=iterations,
            batch_size=batch_size, name='conv_caps1'
        )

        # ConvCaps2
        # (poses, activations) -> conv capsule, 3x3 kernels, strides 1, no padding
        # nets -> (poses (24, 4, 4, 32, 4, 4), activations (24, 4, 4, 32))
        nets = conv_capsule(
            nets, shape=[3, 3, 32, 32], strides=[1, 1, 1, 1], iterations=iterations,
            batch_size=batch_size, name='conv_caps2'
        )

        # Class capsules
        # (poses, activations) -> 1x1 convolution, 10 output capsules
        # nets -> (poses (24, 10, 4, 4), activations (24, 10))
        nets = class_capsules(nets, num_classes, iterations=iterations,
                              batch_size=batch_size, name='class_capsules')

        # poses (24, 10, 4, 4), activations (24, 10)
        poses, activations = nets

    return poses, activations


def predictions(outputs, batch_size, name='output'):
    with tf.variable_scope(name) as scope:
        logits_idx = tf.to_int32(tf.argmax(outputs, axis=1))
        logits_idx = tf.reshape(logits_idx, shape=(batch_size,))
        return logits_idx


def accuracy(outputs, targets, batch_size, name='accuracy'):
    with tf.variable_scope(name) as scope:
        correct_prediction = tf.equal(tf.to_int32(targets), outputs)
        acc = tf.reduce_sum(tf.cast(correct_prediction, tf.float32)) / tf.cast(batch_size, tf.float32)
        return acc


def decode(outputs, hot_targets, batch_size, name='decoder'):
    # Reconstruction
    reconstruct = tf.reshape(tf.multiply(outputs, hot_targets), shape=[batch_size, -1])
    tf.logging.info("Decoder input value dimension:{}".format(reconstruct.get_shape()))

    with tf.variable_scope(name) as scope:
        reconstruct = slim.fully_connected(reconstruct, 512, trainable=True,
                                           weights_regularizer=tf.contrib.layers.l2_regularizer(5e-04))
        reconstruct = slim.fully_connected(reconstruct, 1024, trainable=True,
                                           weights_regularizer=tf.contrib.layers.l2_regularizer(5e-04))
        decoded = slim.fully_connected(reconstruct, cfg.input_size * cfg.input_size * cfg.input_channel,
                                       trainable=True, activation_fn=tf.sigmoid,
                                       weights_regularizer=tf.contrib.layers.l2_regularizer(5e-04))
        tf.logging.info("Decoder output value dimension:{}".format(decoded.get_shape()))
        return decoded


def spread_loss(labels, activations, iterations_per_epoch, global_step, name):
    """Spread loss

    :param labels: (24, 10] in one-hot vector
    :param activations: [24, 10], activation for each class
    :param margin: increment from 0.2 to 0.9 during training

    :return: spread loss
    """
    # Margin schedule
    # Margin increase from 0.2 to 0.9 by an increment of 0.1 for every epoch
    margin = tf.train.piecewise_constant(
        tf.cast(global_step, dtype=tf.int32),
        boundaries=[
            (iterations_per_epoch * x) for x in range(1, 8)
        ],
        values=[
            x / 10.0 for x in range(2, 10)
        ]
    )

    activations_shape = activations.get_shape().as_list()

    with tf.variable_scope(name) as scope:
        # mask_t, mask_f Tensor (?, 10)
        mask_t = tf.equal(labels, 1)  # Mask for the true label
        mask_i = tf.equal(labels, 0)  # Mask for the non-true label


        # Activation for the true label
        # activations_t (?, 1)
        activations_t = tf.reshape(
            tf.boolean_mask(activations, mask_t), shape=(tf.shape(activations)[0], 1)
        )

        # Activation for the other classes
        # activations_i (?, 9)
        activations_i = tf.reshape(
            tf.boolean_mask(activations, mask_i), [tf.shape(activations)[0], activations_shape[1] - 1]
        )

        l = tf.reduce_sum(
            tf.square(
                tf.maximum(
                    0.0,
                    margin - (activations_t - activations_i)
                )
            )
        )
        tf.losses.add_loss(l)

        return l


def spread_loss1(hot_labels, activations, images, decoded, m):
    """
    :param hot_labels:
    :param activations:
    :param images:
    :param m:
    :return:
    """
    y = tf.expand_dims(hot_labels, axis=2)
    data_size = int(images.get_shape()[1])

    # spread loss
    output1 = tf.reshape(activations, shape=[cfg.batch_size, 1, -1])
    at = tf.matmul(output1, y)
    """Paper eq(3)."""
    sp_loss = tf.square(tf.maximum(0., m - (at - output1)))
    sp_loss = tf.matmul(sp_loss, 1. - y)
    sp_loss = tf.reduce_mean(sp_loss)

    # reconstruction loss
    tf.logging.info("Decoder input value dimension:{}".format(decoded.get_shape()))
    with tf.variable_scope('decoder'):
        x = tf.reshape(images, shape=[cfg.batch_size, -1])
        reconstruction_loss = tf.reduce_mean(tf.square(decoded - x))

    if cfg.weight_reg:
        # regularization loss
        regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # loss+0.0005*reconstruction_loss+regularization#
        total_loss = tf.add_n([sp_loss] + [0.0005 * data_size * data_size * reconstruction_loss] + regularization)
    else:
        total_loss = tf.add_n([sp_loss] + [0.0005 * data_size * data_size * reconstruction_loss])

    return total_loss, sp_loss, reconstruction_loss


class CapsNet(object):
    def __init__(self, images, labels, batch_size, num_train_batch=None, is_training=True):
        self.graph = tf.get_default_graph()
        with self.graph.as_default():
            self.batch_size = batch_size
            if is_training:
                # images: Tensor (?, 28, 28, 1)
                # labels: Tensor (?)
                # self.images = tf.placeholder(tf.float32,
                #                              shape=(None, cfg.input_size, cfg.input_size, cfg.input_channel))
                # self.labels = tf.placeholder(tf.int32, shape=(None,))
                self.images = images
                self.labels = labels

                self.one_hot_labels = slim.one_hot_encoding(self.labels, cfg.num_class)  # Tensor(?, 10)
                logger.info("Images: {}".format(self.images))
                logger.info("Labels: {}".format(self.labels))
                logger.info("Hot Labels: {}".format(self.one_hot_labels))
                # poses: Tensor(?, 10, 4, 4) activations: (?, 10)
                self.poses, self.activations = capsules_net(self.images, num_classes=cfg.num_class, iterations=3,
                                                            batch_size=self.batch_size, name='capsules_em')
                self.decoded = decode(self.activations, self.one_hot_labels, self.batch_size)
                self.global_step = tf.train.get_or_create_global_step()

                # self.loss = spread_loss(self.one_hot_labels, self.activations, num_train_batch, self.global_step,
                #                         name='spread_loss')
                self.m_op = tf.placeholder(dtype=tf.float32, shape=())
                self.loss, self.sp_loss, self.reconstruction_loss = spread_loss1(self.one_hot_labels, self.activations, self.images,
                                                               self.decoded, self.m_op)
                self.predictions = predictions(self.activations, self.batch_size, 'predictions')
                self.accuracy = accuracy(self.predictions, self.labels, self.batch_size, "accuracy")

                self.lrn_rate = tf.maximum(tf.train.exponential_decay(
                    1e-3, self.global_step, 1000, 0.8), 1e-5)

                self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
                self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

                # self.train_op = tf.learning.create_train_op(self.loss, self.optimizer, global_step=self.global_step,
                # clip_gradient_norm=4.0)
                self.summary_op = self.get_summary_op(scope='train', name_prefix='train/')

                self.saver = tf.train.Saver(max_to_keep=5)

                """Display parameters"""
                total_p = np.sum([np.prod(v.get_shape().as_list()) for v in tf.global_variables()]).astype(np.int32)
                train_p = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]).astype(np.int32)
                logger.info('Total Parameters: {}'.format(total_p))
                logger.info('Trainable Parameters: {}'.format(train_p))
            else:
                # images: Tensor (?, 28, 28, 1)
                # labels: Tensor (?)
                self.images = tf.placeholder(tf.float32,
                                             shape=(self.batch_size, cfg.input_size, cfg.input_size, cfg.input_channel))
                self.labels = tf.placeholder(tf.int32, shape=(batch_size,))

                self.one_hot_labels = slim.one_hot_encoding(self.labels, cfg.num_class)  # Tensor(?, 10)
                logger.info("Images: {}".format(self.images))
                logger.info("Labels: {}".format(self.labels))
                logger.info("Hot Labels: {}".format(self.one_hot_labels))
                # poses: Tensor(?, 10, 4, 4) activations: (?, 10)
                self.poses, self.activations = capsules_net(self.images, num_classes=cfg.num_class, iterations=3,
                                                            batch_size=self.batch_size, name='capsules_em')
                self.decoded = decode(self.activations, self.one_hot_labels, self.batch_size)
                self.global_step = tf.train.get_or_create_global_step()

                self.predictions = predictions(self.activations, self.batch_size, 'predictions')
                self.accuracy = accuracy(self.predictions, self.labels, self.batch_size, "accuracy")

                self.summary_op = self.get_summary_op
                self.saver = tf.train.Saver(max_to_keep=5)

    def loss(self):
        # 1. The margin loss

        # [batch_size, 10, 1, 1]
        # max_l = max(0, m_plus-||v_c||)^2
        max_l = tf.square(tf.maximum(0., cfg.m_plus - self.v_length))
        # max_r = max(0, ||v_c||-m_minus)^2
        max_r = tf.square(tf.maximum(0., self.v_length - cfg.m_minus))
        assert max_l.get_shape() == [cfg.batch_size, 10, 1, 1]

        # reshape: [batch_size, 10, 1, 1] => [batch_size, 10]
        max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))
        max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))

        # calc T_c: [batch_size, 10]
        # T_c = Y, is my understanding correct? Try it.
        T_c = self.Y
        # [batch_size, 10], element-wise multiply
        L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r

        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

        # 2. The reconstruction loss
        orgin = tf.reshape(self.X, shape=(cfg.batch_size, -1))
        squared = tf.square(self.decoded - orgin)
        self.reconstruction_err = tf.reduce_mean(squared)

        # 3. Total loss
        # The paper uses sum of squared error as reconstruction error, but we
        # have used reduce_mean in `# 2 The reconstruction loss` to calculate
        # mean squared error. In order to keep in line with the paper,the
        # regularization scale should be 0.0005*784=0.392
        self.total_loss = self.margin_loss + cfg.regularization_scale * self.reconstruction_err

    def get_summary_op(self, scope, name_prefix=''):
        train_summary = []
        if scope == "train":
            train_summary.append(tf.summary.scalar(name_prefix + 'total_loss', self.loss))
            train_summary.append(tf.summary.scalar(name_prefix + 'spread_loss', self.sp_loss))
            train_summary.append(tf.summary.scalar(name_prefix + 'reconstruction_loss', self.reconstruction_loss))
            recon_img = tf.reshape(self.decoded,
                                   shape=(self.batch_size, cfg.input_size, cfg.input_size, cfg.input_channel))
            train_summary.append(tf.summary.image(name_prefix + 'reconstruction', recon_img))
            train_summary.append(tf.summary.scalar(name_prefix + 'accuracy', self.accuracy))
            train_summary.append(tf.summary.histogram(name_prefix +'activations', self.activations))
            train_summary.append(tf.summary.histogram(name_prefix +'poses', self.poses))
            # train_summary.append(tf.summary.scalar('learning_rate', self.learning_rate))
        elif scope == "test":
            train_summary.append(tf.summary.scalar(name_prefix + 'accuracy', self.accuracy))
            recon_img = tf.reshape(self.decoded,
                                   shape=(self.batch_size, cfg.input_size, cfg.input_size, cfg.input_channel))
            train_summary.append(tf.summary.image(name_prefix + 'reconstruction', recon_img))
            train_summary.append(tf.summary.histogram(name_prefix +'activations', self.activations))
            train_summary.append(tf.summary.histogram(name_prefix +'poses', self.poses))
        return tf.summary.merge(train_summary)
