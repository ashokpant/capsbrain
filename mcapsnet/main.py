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
    train_data = get_create_inputs(cfg.dataset, True, cfg.epoch, size=(cfg.input_size, cfg.input_size))
    num_train_batch = int(cfg.train_size / cfg.batch_size)  # 60,000/24 = 2500

    """Set summary writer"""
    if not os.path.exists(cfg.summary_dir):
        os.makedirs(cfg.summary_dir)

    # images: Tensor (?, 28, 28, 1)
    # labels: Tensor (?)
    images = train_data[0]
    labels = train_data[1]

    model = CapsNet(images=images, labels=labels, batch_size=cfg.batch_size, num_train_batch=num_train_batch)

    with model.graph.as_default():
        """Set summary writer"""
        if not os.path.exists(cfg.summary_dir):
            os.makedirs(cfg.summary_dir)
        summary_writer = tf.summary.FileWriter(cfg.summary_dir)

    sv = tf.train.Supervisor(
        graph=model.graph,
        is_chief=True,
        summary_writer=summary_writer,
        global_step=model.global_step,
        saver=model.saver)

    """Set Session settings."""
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    with sv.managed_session(config=config) as sess:
        """Start queue runner."""
        threads = tf.train.start_queue_runners(sess=sess, coord=sv.coord)

        # Main loop
        m_min = 0.2
        m_max = 0.9
        m = m_min
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
                        [model.train_op, model.loss, model.accuracy, model.summary_op], feed_dict={model.m_op: m})
                    assert not np.isnan(loss_value), 'Something wrong! loss is nan...'
                    sv.summary_writer.add_summary(summary_str, g_step)
                    logger.info(
                        '{} iteration finises in {:.4f} second,  loss={:.4f}, train_acc={:.2f}'.format(step, (
                                time.time() - tic),
                                                                                                       loss_value,
                                                                                                       train_acc))
                else:
                    _, loss_value, summary_str = sess.run([model.train_op, model.loss, model.summary_op],
                                                          feed_dict={model.m_op: m})
                    sv.summary_writer.add_summary(summary_str, g_step)
                    logger.info(
                        '{} iteration finises in {:.4f} second,  loss={:.4f}'.format(step, time.time() - tic,
                                                                                     loss_value))

                if (g_step + 1) % cfg.save_freq == 0:
                    """Save model periodically"""
                    ckpt_file = os.path.join(cfg.ckpt_dir, 'model_{:.4f}.ckpt'.format(loss_value))
                    sv.saver.save(sess, ckpt_file, global_step=g_step)

            """Epoch wise linear annealing."""
            if g_step > 0:
                m += (m_max - m_min) / (cfg.epoch * cfg.m_schedule)
                if m > m_max:
                    m = m_max

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
        data = get_create_inputs(cfg.dataset, True, cfg.epoch, size=(cfg.input_size, cfg.input_size))
        num_batch = int(cfg.train_size / cfg.batch_size)
    else:
        data = get_create_inputs(cfg.dataset, False, cfg.epoch, size=(cfg.input_size, cfg.input_size))
        num_batch = int(cfg.test_size / cfg.batch_size)

    model = CapsNet(images=None, labels=None,  batch_size=cfg.batch_size,  num_train_batch=None,  is_training=False)

    with model.graph.as_default():
        if scope == 'train':
            summary_op = model.summary_op(scope="test", name_prefix='eval/train/')
        else:
            summary_op = model.summary_op(scope="test", name_prefix='eval/test/')

        summary_writer = tf.summary.FileWriterCache.get(cfg.summary_dir)

    sv = tf.train.Supervisor(
        graph=model.graph,
        is_chief=True,
        logdir=cfg.ckpt_dir,
        summary_writer=summary_writer,
        global_step=model.global_step,
        saver=model.saver)

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

    sv = tf.train.Supervisor(
        graph=model.graph,
        is_chief=True,
        summary_writer=None,
        global_step=model.global_step,
        saver=model.saver)
    images = []
    if cfg.input_file.endswith('.txt'):
        images = [line.rstrip('\n') for line in open(cfg.input_file)]
    else:
        images.append(cfg.input_file)

    with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sv.saver.restore(sess, tf.train.latest_checkpoint(cfg.ckpt_dir))
        logger.info("Model is restored successfully: {}".format(cfg.ckpt_dir))

        for filename in images:
            tic = time.time()
            image = utils.imread(filename)
            img = utils.imresize(image, (cfg.input_size, cfg.input_size))
            img = utils.bgr2gray(img)
            img = np.expand_dims(img, 3)
            img = np.expand_dims(img, 0)
            # poses, activations, predictions = sess.run([model.poses, model.activations, model.predictions],
            # feed_dict={model.images: img})
            predictions = sess.run(model.predictions, feed_dict={model.images: img})
            tac = time.time() - tic
            logger.info("Input:{} , Prediction: {}, Time: {:.3f}".format(cfg.input_file, predictions[0], tac))
            utils.show_image(image=image, text=str(predictions[0]), pause=100)


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
