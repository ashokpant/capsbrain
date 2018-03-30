import logging
import os
import sys

import daiquiri
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import config
from capsule.capsNet import CapsNet
from config import cfg
from utils import get_create_inputs

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

    model = CapsNet(images=train_data[0], labels=train_data[1], batch_size=cfg.batch_size)

    with model.graph.as_default():
        """Set summary writer"""
        if not os.path.exists(cfg.summary_dir):
            os.makedirs(cfg.summary_dir)
        summary_writer = tf.summary.FileWriter(cfg.summary_dir)

    supervisor = tf.train.Supervisor(
        graph=model.graph,
        is_chief=True,
        summary_writer=summary_writer,
        global_step=model.global_step,
        saver=model.saver)

    """Set Session settings."""
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    with supervisor.managed_session(config=config) as sess:
        for epoch in range(cfg.epoch):
            print("Training for epoch %d/%d:" % (epoch, cfg.epoch))
            if supervisor.should_stop():
                print('supervisor stoped!')
                break
            for step in tqdm(range(num_train_batch), total=num_train_batch, ncols=70, leave=False, unit='b'):
                global_step = epoch * num_train_batch + step
                try:
                    if global_step % cfg.train_sum_freq == 0:
                        _, loss, train_acc, summary_str = sess.run(
                            [model.train_op, model.total_loss, model.accuracy, model.train_summary])
                        assert not np.isnan(loss), 'Something wrong! loss is nan...'
                        supervisor.summary_writer.add_summary(summary_str, global_step)
                    else:
                        sess.run(model.train_op)
                except KeyboardInterrupt:
                    sess.close()
                    sys.exit()
                except tf.errors.InvalidArgumentError as e:
                    logger.warning('{} iteration contains NaN gradients. Discarding...'.format(step))
                    continue

            if (global_step + 1) % cfg.save_freq == 0:
                """Save model periodically"""
                ckpt_file = os.path.join(cfg.ckpt_dir, 'model_{:.4f}.ckpt'.format(loss))
                supervisor.saver.save(sess, ckpt_file, global_step=global_step)


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

    model = CapsNet(images=None, labels=None, batch_size=cfg.batch_size, is_training=False)

    summary_writer = tf.summary.FileWriterCache.get(cfg.summary_dir)

    sv = tf.train.Supervisor(
        graph=model.graph,
        is_chief=True,
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
            acc = sess.run([model.accuracy],
                           feed_dict={model.X: batch_images, model.labels: batch_labels})
            logger.info("Batch {}, batch_accuracy: {:.2f}, total_acc: {:.2f}".format(step, acc, avg_acc / (step + 1.0)))
            avg_acc += acc
        avg_acc = avg_acc / num_batch
        logger.info("Total accuracy: {:2f}".format(avg_acc))
        sv.coord.join(threads)


def predict():
    raise NotImplementedError()


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
