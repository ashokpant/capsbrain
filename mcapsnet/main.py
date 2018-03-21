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
from datetime import datetime
from tqdm import tqdm

import mcapsnet
from mcapsnet import config
from mcapsnet.config import cfg, get_dataset_size_train, get_dataset_size_test
from mcapsnet.network import CapsNet
from mcapsnet.utils import load_data, get_create_inputs

slim = tf.contrib.slim

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)


def train():
    tf.logging.set_verbosity(tf.logging.INFO)

    np.random.seed(cfg.seed)
    tf.set_random_seed(cfg.seed)

    dataset_size = get_dataset_size_train(cfg.dataset)

    """Get batches per epoch."""
    train_data = get_create_inputs(cfg.dataset, True, 10)
    num_train_batch = int(dataset_size / cfg.batch_size)  # 60,000/24 = 2500

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
        threads = tf.train.start_queue_runners(sess=sess)

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
                    _, loss_value, train_acc, summary_str = sess.run([model.train_op, model.loss, model.accuracy, model.summary_op])
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


def evaluation():
    pass
    # teX, teY, num_te_batch = load_data(cfg.dataset, cfg.batch_size, is_training=False)
    # with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    #     supervisor.saver.restore(sess, tf.train.latest_checkpoint(cfg.logdir))
    #     tf.logging.info('Model restored!')
    #
    #     test_acc = 0
    #     for i in tqdm(range(num_te_batch), total=num_te_batch, ncols=70, leave=False, unit='b'):
    #         start = i * cfg.batch_size
    #         end = start + cfg.batch_size
    #         acc = sess.run(model.accuracy, {model.X: teX[start:end], model.labels: teY[start:end]})
    #         test_acc += acc
    #     test_acc = test_acc / (cfg.batch_size * num_te_batch)
    #     print('Test accuracy :' + test_acc )


def main(_):
    # tf.logging.info(' Loading Graph...')
    # model = MCapsNet()
    # tf.logging.info(' Graph loaded')

    # sv = tf.train.Supervisor(graph=model.graph, logdir=cfg.logdir, save_model_secs=0)
    config.update_cfg(cfg.dataset)
    logger.info("Config: {}".format(cfg.flag_values_dict()))
    if cfg.is_training:
        tf.logging.info(' Start training...')
        train()
        tf.logging.info('Training done')
    else:
        evaluation()


if __name__ == '__main__':
    tf.app.run()
