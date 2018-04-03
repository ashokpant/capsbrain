"""
-- User: Ashok Kumar Pant (asokpant@gmail.com)
-- Date: 4/2/18
-- Time: 12:19 PM
"""
from dcaps import layers

import tensorflow as tf

from config import cfg
from utils import softmax
import numpy as np
import daiquiri

logger = daiquiri.getLogger(__name__)


def predictions(activations, name='output'):
    with tf.variable_scope(name) as scope:
        if len(activations.shape) ==1:
            logits_idx = tf.to_int32(tf.argmax(softmax(activations, axis=0), axis=0))
            return [logits_idx]
        else:
            logits_idx = tf.to_int32(tf.argmax(softmax(activations, axis=1), axis=1))
            return logits_idx


def accuracy(outputs, targets, name='accuracy'):
    with tf.variable_scope(name) as scope:
        correct = tf.equal(tf.to_int32(targets), outputs)
        acc = tf.reduce_sum(tf.cast(correct, tf.float32)) / tf.cast(targets.shape[0], tf.float32)
        return acc


def decode_digicaps(digitcaps, hot_targets, batch_size, name='decoder'):
    # Decoder structure in Fig. 2
    # Reconstructe the MNIST images with 3 FC layers
    # [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]
    with tf.variable_scope(name) as scope:
        masked_caps = tf.multiply(digitcaps, tf.reshape(hot_targets, (-1, cfg.num_class, 1, 1)))
        active_caps = tf.reshape(masked_caps, shape=(batch_size, -1))
        fc1 = tf.contrib.layers.fully_connected(active_caps, num_outputs=512)
        fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
        decoded = tf.contrib.layers.fully_connected(fc2,
                                                    num_outputs=cfg.input_size * cfg.input_size * cfg.input_channel,
                                                    activation_fn=tf.sigmoid)
        return decoded


class CapsNet(object):
    def __init__(self, images, labels, batch_size, is_training=True):
        self.batch_size = batch_size
        self.graph = tf.get_default_graph()
        with self.graph.as_default():
            if is_training:
                self.x, self.labels = images, labels
                self.y = tf.one_hot(self.labels, depth=cfg.num_class, axis=1, dtype=tf.float32)

                self.digitCaps, self.activations = self.build_arch()
                self.decoded = decode_digicaps(self.digitCaps, self.y, self.batch_size)
                self.loss, self.margin_loss, self.reconstruction_loss = self.build_loss()

                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer()
                self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
            else:

                self.x = tf.placeholder(tf.float32,
                                        shape=(self.batch_size, cfg.input_size, cfg.input_size, cfg.input_channel))
                self.labels = tf.placeholder(tf.int32, shape=(self.batch_size,))
                self.y = tf.one_hot(self.labels, depth=cfg.num_class, axis=1, dtype=tf.float32)
                self.digitCaps, self.activations = self.build_arch()
                self.decoded = decode_digicaps(self.digitCaps, self.y, self.batch_size)
                # self.global_step = tf.train.get_or_create_global_step()

            self.predictions = predictions(self.activations)
            self.accuracy = accuracy(self.predictions, self.labels)

            if is_training:
                self.summary_op = self.get_summary_op(scope='train', name_prefix='train/')
            else:
                self.summary_op = self.get_summary_op

            self.saver = tf.train.Saver()

            total_p = np.sum([np.prod(v.get_shape().as_list()) for v in tf.global_variables()]).astype(np.int32)
            train_p = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]).astype(np.int32)
            logger.info('Total Parameters: {}'.format(total_p))
            logger.info('Trainable Parameters: {}'.format(train_p))

    def build_arch(self):
        with tf.variable_scope('conv1_layer'):
            # Conv1, return with shape [batch_size, 20, 20, 256]
            conv1 = tf.contrib.layers.conv2d(self.x, num_outputs=256, kernel_size=9, stride=1, padding='VALID')

        # return primaryCaps: [batch_size, 1152, 8, 1], activation: [batch_size, 1152]
        with tf.variable_scope('primary_caps_layer'):
            primaryCaps, activation = layers.primaryCaps(conv1, filters=32, kernel_size=9, strides=2,
                                                         out_caps_shape=[8, 1])

        # return digitCaps: [batch_size, num_label, 16, 1], activation: [batch_size, num_label]
        with tf.variable_scope('digit_caps_layer'):
            primaryCaps = tf.reshape(primaryCaps, shape=[self.batch_size, -1, 8, 1])
            digitCaps, activation = layers.fully_connected(primaryCaps, activation, num_outputs=cfg.num_class,
                                                           out_caps_shape=[16, 1],
                                                           routing_method='dynamic')

        return digitCaps, activation

    def build_loss(self):
        # 1. Margin loss

        # max_l = max(0, m_plus-||v_c||)^2
        max_l = tf.square(tf.maximum(0., cfg.m_plus - self.activations))
        # max_r = max(0, ||v_c||-m_minus)^2
        max_r = tf.square(tf.maximum(0., self.activations - cfg.m_minus))

        # reshape: [batch_size, num_label, 1, 1] => [batch_size, num_label]
        max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))
        max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))

        # calc T_c: [batch_size, num_label]
        # T_c = Y, is my understanding correct? Try it.
        T_c = self.y
        # [batch_size, num_label], element-wise multiply
        L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r

        margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

        # 2. The reconstruction loss
        orgin = tf.reshape(self.x, shape=(self.batch_size, -1))
        squared = tf.square(self.decoded - orgin)
        reconstruction_loss = tf.reduce_mean(squared)

        # 3. Total loss
        # The paper uses sum of squared error as reconstruction error, but we
        # have used reduce_mean in `# 2 The reconstruction loss` to calculate
        # mean squared error. In order to keep in line with the paper,the
        # regularization scale should be 0.0005*784=0.392
        total_loss = margin_loss + cfg.regularization_scale * reconstruction_loss

        return total_loss, margin_loss, reconstruction_loss

    def update_any(self):
        pass

    def get_summary_op(self, scope, name_prefix=''):
        train_summary = []
        if scope == "train":
            train_summary.append(tf.summary.scalar(name_prefix + 'total_loss', self.loss))
            train_summary.append(tf.summary.scalar(name_prefix + 'margin_loss', self.margin_loss))
            train_summary.append(tf.summary.scalar(name_prefix + 'reconstruction_loss', self.reconstruction_loss))
            recon_img = tf.reshape(self.decoded,
                                   shape=(self.batch_size, cfg.input_size, cfg.input_size, cfg.input_channel))
            train_summary.append(tf.summary.image(name_prefix + 'reconstruction', recon_img))
            train_summary.append(tf.summary.scalar(name_prefix + 'accuracy', self.accuracy))
            train_summary.append(tf.summary.histogram(name_prefix + 'activation', self.activations))
        elif scope == "test":
            train_summary.append(tf.summary.scalar(name_prefix + 'accuracy', self.accuracy))
            recon_img = tf.reshape(self.decoded,
                                   shape=(self.batch_size, cfg.input_size, cfg.input_size, cfg.input_channel))
            train_summary.append(tf.summary.image(name_prefix + 'reconstruction', recon_img))
            train_summary.append(tf.summary.histogram(name_prefix + 'activation', self.activations))
        return tf.summary.merge(train_summary)