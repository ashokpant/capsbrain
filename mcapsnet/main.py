"""
-- User: Ashok Kumar Pant (asokpant@gmail.com)
-- Date: 3/13/18
-- Time: 1:29 PM
"""
import logging
import os
import time

import daiquiri
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from mcapsnet import config, utils
from mcapsnet.config import cfg
from mcapsnet.network import CapsNet
from mcapsnet.utils import get_create_inputs

slim = tf.contrib.slim

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)


def train():
    tf.logging.set_verbosity(tf.logging.INFO)

    np.random.seed(cfg.seed)
    tf.set_random_seed(cfg.seed)

    """Get batches per epoch."""
    train_data = get_create_inputs(cfg.dataset, True, cfg.epoch)
    num_train_batch = int(cfg.train_size / cfg.batch_size)  # 60,000/24 = 2500

    """Set summary writer"""
    if not os.path.exists(cfg.summary_dir):
        os.makedirs(cfg.summary_dir)

    # images: Tensor (?, 28, 28, 1)
    # labels: Tensor (?)
    images = train_data[0]
    labels = train_data[1]

    model = CapsNet(images=images, labels=labels, num_train_batch=num_train_batch)

    with model.graph.as_default():
        """Set Saver."""
        var_to_save = [v for v in tf.global_variables(
        ) if 'Adam' not in v.name]  # Don't save redundant Adam beta/gamma
        saver = tf.train.Saver(var_list=var_to_save, max_to_keep=cfg.epoch)

        """Display parameters"""
        total_p = np.sum([np.prod(v.get_shape().as_list()) for v in var_to_save]).astype(np.int32)
        train_p = np.sum([np.prod(v.get_shape().as_list())
                          for v in tf.trainable_variables()]).astype(np.int32)
        logger.info('Total Parameters: {}'.format(total_p))
        logger.info('Trainable Parameters: {}'.format(train_p))

        """Set summary writer"""
        if not os.path.exists(cfg.summary_dir):
            os.makedirs(cfg.summary_dir)
        summary_writer = tf.summary.FileWriter(cfg.summary_dir)

    sv = tf.train.Supervisor(
        graph=model.graph,
        is_chief=True,
        summary_writer=summary_writer,
        global_step=model.global_step,
        saver=saver)

    """Set Session settings."""
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    with sv.managed_session(config=config) as sess:
        """Start queue runner."""
        threads = tf.train.start_queue_runners(sess=sess, coord=sv.coord)

        # Main loop
        for epoch in range(cfg.epoch):
            logger.info("Training for epoch {}/{}:".format(epoch, cfg.epoch))
            for step in tqdm(range(num_train_batch), total=num_train_batch, ncols=70, leave=False, unit='b'):
                if sv.should_stop():
                    logger.intfo('Session stopped!')
                    break
                tic = time.time()

                g_step = epoch * num_train_batch + step

                if g_step % cfg.train_sum_freq == 0:
                    _, loss_value, train_acc, summary_str = sess.run(
                        [model.train_op, model.loss, model.accuracy, model.summary_op])
                    assert not np.isnan(loss_value), 'Something wrong! loss is nan...'
                    sv.summary_writer.add_summary(summary_str, g_step)
                    logger.info(
                        '{} iteration finises in {} second,  loss={}, train_acc={}'.format(step, (time.time() - tic),
                                                                                           loss_value, train_acc))
                else:
                    _, loss_value, summary_str = sess.run([model.train_op, model.loss, model.summary_op])
                    sv.summary_writer.add_summary(summary_str, g_step)
                    logger.info(
                        '{} iteration finises in {} second,  loss={}'.format(step, time.time() - tic,
                                                                             loss_value))

                if (g_step + 1) % cfg.save_freq == 0:
                    """Save model periodically"""
                    ckpt_file = os.path.join(cfg.ckpt_dir, 'model_{:.4f}.ckpt'.format(loss_value))
                    sv.saver.save(sess, ckpt_file, global_step=g_step)

            """Save model at each epoch periodically"""
            ckpt_file = os.path.join(cfg.ckpt_dir, 'model_{:.4f}.ckpt'.format(loss_value))
            sv.saver.save(sess, ckpt_file, global_step=g_step)
    sess.close()


def evaluation(scope='test'):
    logger.info("Evaluating on {} dataset.".format(scope))
    tf.logging.set_verbosity(tf.logging.INFO)

    np.random.seed(cfg.seed)
    tf.set_random_seed(cfg.seed)

    """Get data."""
    if scope == 'train':
        data = get_create_inputs(cfg.dataset, True, cfg.epoch)
        num_batch = int(cfg.train_size / cfg.batch_size)
    else:
        data = get_create_inputs(cfg.dataset, False, cfg.epoch)
        num_batch = int(cfg.test_size / cfg.batch_size)

    model = CapsNet(images=None, labels=None, num_train_batch=None, batch_size=cfg.batch_size, is_training=False)

    with model.graph.as_default():
        if scope == 'train':
            summary_op = model.summary_op(scope="test", name_prefix='eval/train/')
        else:
            summary_op = model.summary_op(scope="test", name_prefix='eval/test/')

        summary_writer = tf.summary.FileWriterCache.get(cfg.summary_dir)
        var_to_save = [v for v in tf.global_variables(
        ) if 'Adam' not in v.name]  # Don't save redundant Adam beta/gamma
        saver = tf.train.Saver(var_list=var_to_save)

    sv = tf.train.Supervisor(
        graph=model.graph,
        is_chief=True,
        summary_writer=summary_writer,
        global_step=model.global_step,
        saver=saver)

    with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sv.saver.restore(sess, tf.train.latest_checkpoint(cfg.ckpt_dir))
        logger.info("Model is restored successfully: {}".format(cfg.ckpt_dir))
        threads = tf.train.start_queue_runners(sess=sess, coord=sv.coord)

        avg_acc = 0
        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            batch_images, batch_labels = sess.run(data)
            acc, summary_str = sess.run([model.accuracy, summary_op],
                                        feed_dict={model.images: batch_images, model.labels: batch_labels})
            summary_writer.add_summary(summary_str, step)
            logger.info("Batch {}, batch_accuracy: {:.2f}, total_acc: {:.2f}".format(step, acc, avg_acc / (step + 1.0)))
            avg_acc += acc
        avg_acc = avg_acc / num_batch
        logger.info("Total accuracy: {:2f}".format(avg_acc))
        sv.coord.join(threads)


def predict():
    logger.info("Prediction with {} model.".format(cfg.dataset))
    model = CapsNet(images=None, labels=None, num_train_batch=None, batch_size=1, is_training=False)

    with model.graph.as_default():
        vars = [v for v in tf.global_variables(
        ) if 'Adam' not in v.name]  # Don't save redundant Adam beta/gamma
        saver = tf.train.Saver(var_list=vars)

    sv = tf.train.Supervisor(
        graph=model.graph,
        is_chief=True,
        summary_writer=None,
        global_step=model.global_step,
        saver=saver)
    image = utils.read_image(cfg.input_file)
    img = utils.resize_image(image, [cfg.input_size, cfg.input_size])
    with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sv.saver.restore(sess, tf.train.latest_checkpoint(cfg.ckpt_dir))
        logger.info("Model is restored successfully: {}".format(cfg.ckpt_dir))
        poses, activations = sess.run([model.poses, model.activations], feed_dict={model.images: img})

        logits_idx = tf.to_int32(tf.argmax(activations, axis=1))
        logits_idx = tf.reshape(logits_idx, shape=(model.batch_size,))

        idx = sess.run(logits_idx)

        logger.info("Input:{}".format(cfg.input_file))
        logger.info("Output:{}".format(poses, activations))
        logger.info("Output class:{}".format(idx))
        utils.show_image(image=image, text=str(activations))


def main(_):
    config.update_cfg(cfg.dataset)
    logger.info("Config: {}".format(cfg.flag_values_dict()))
    if cfg.mode == 'train':
        tf.logging.info(' Start training...')
        train()
        tf.logging.info('Training done')
    elif cfg.mode == 'eval':
        evaluation(scope='test')
    elif cfg.mode == 'predict':
        predict()
    else:
        logger.warning("Unknown operation mode: {}".format(cfg.mode))


if __name__ == '__main__':
    tf.app.run()
