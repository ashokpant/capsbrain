"""
-- User: Ashok Kumar Pant (asokpant@gmail.com)
-- Date: 3/13/18
-- Time: 1:29 PM
"""
import logging
import os
import time
import sys
import daiquiri
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import dcaps
import mcaps
import utils
import config
from config import cfg
from utils import get_create_inputs

slim = tf.contrib.slim

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)


def get_network(name, images=None, labels=None, batch_size=cfg.batch_size, is_training=True, *args):
    if name == 'dcaps':
        return dcaps.network.CapsNet(images= images, labels=labels, batch_size=batch_size, is_training=is_training)
    elif name == 'mcaps':
        return mcaps.network.CapsNet(*args)
    else:
        raise ValueError("Invalid network type {}, available networks = {}".format(cfg.network, ['dcaps', 'mcaps']))


def train():
    logger.info("Training on {} dataset.".format(cfg.dataset))

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

    model = get_network(cfg.network, images, labels, cfg.batch_size)

    with model.graph.as_default():
        if not os.path.exists(cfg.summary_dir):
            os.makedirs(cfg.summary_dir)

        if not os.path.exists(cfg.ckpt_dir):
            os.makedirs(cfg.ckpt_dir)

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

        for epoch in range(cfg.epoch):
            logger.info("Training for epoch {}/{}:".format(epoch, cfg.epoch))
            for step in tqdm(range(num_train_batch), total=num_train_batch, ncols=70, leave=False, unit='b'):
                if sv.should_stop():
                    logger.intfo('Session stopped!')
                    break
                tic = time.time()
                g_step = epoch * num_train_batch + step

                try:
                    if g_step % cfg.train_sum_freq == 0:
                        _, loss_value, train_acc, summary_str = sess.run(
                            [model.train_op, model.loss, model.accuracy, model.summary_op])

                        assert not np.isnan(loss_value), 'Something wrong! loss is nan...'
                        sv.summary_writer.add_summary(summary_str, g_step)
                        logger.info(
                            '{} iteration finises in {:.4f} second,  loss={:.4f}, train_acc={:.2f}'.format(step, (
                                    time.time() - tic),
                                                                                                           loss_value,
                                                                                                           train_acc))
                    else:
                        sess.run(model.train_op)
                except KeyboardInterrupt:
                    sess.close()
                    sys.exit()
                except tf.errors.InvalidArgumentError as e:
                    logger.warning('{} iteration contains NaN gradients. Discarding...'.format(step))
                    continue

                if (g_step + 1) % cfg.save_freq == 0:
                    ckpt_file = os.path.join(cfg.ckpt_dir, 'model_{:.4f}.ckpt'.format(loss_value))
                    sv.saver.save(sess, ckpt_file, global_step=g_step)

            """Update any variables every epoch (eg. m for mcpas)"""
            model.update_any()

            ckpt_file = os.path.join(cfg.ckpt_dir, 'model_{:.4f}.ckpt'.format(loss_value))
            sv.saver.save(sess, ckpt_file, global_step=g_step)
    sess.close()


def evaluation(scope='test'):
    logger.info("Evaluating on {} dataset.".format(scope))

    """Get data."""
    if scope == 'train':
        data = get_create_inputs(cfg.dataset, True, cfg.epoch, size=(cfg.input_size, cfg.input_size))
        num_batch = int(cfg.train_size / cfg.batch_size)
    else:
        data = get_create_inputs(cfg.dataset, False, cfg.epoch, size=(cfg.input_size, cfg.input_size))
        num_batch = int(cfg.test_size / cfg.batch_size)

    model = get_network(cfg.network, images=None, labels=None, batch_size=cfg.batch_size, is_training=False)

    with model.graph.as_default():
        if scope == 'train':
            summary_op = model.summary_op(scope="test", name_prefix='eval/train/')
        else:
            summary_op = model.summary_op(scope="test", name_prefix='eval/test/')

        summary_writer = tf.summary.FileWriterCache.get(cfg.summary_dir)

    sv = tf.train.Supervisor(
        graph=model.graph,
        is_chief=True,
        summary_writer=summary_writer,
        saver=model.saver)

    with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sv.saver.restore(sess, tf.train.latest_checkpoint(cfg.ckpt_dir))
        logger.info("Model is restored successfully: {}".format(cfg.ckpt_dir))
        threads = tf.train.start_queue_runners(sess=sess, coord=sv.coord)

        avg_acc = 0
        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            batch_images, batch_labels = sess.run(data)
            acc, summary_str = sess.run([model.accuracy, summary_op],
                                        feed_dict={model.x: batch_images, model.labels: batch_labels, model.cum_acc: avg_acc/(step+1.0)})
            summary_writer.add_summary(summary_str, step)
            avg_acc += acc
            logger.info("Batch {}, batch_accuracy: {:.2f}, total_acc: {:.2f}".format(step, acc, avg_acc / (step + 1.0)))

        avg_acc = avg_acc / num_batch
        logger.info("Total accuracy: {:2f}".format(avg_acc))
        sv.coord.join(threads)


def predict():
    logger.info("Prediction with {} model.".format(cfg.dataset))
    model = get_network(cfg.network, images=None, labels=None, batch_size=1, is_training=False)

    sv = tf.train.Supervisor(
        graph=model.graph,
        is_chief=True,
        summary_writer=None,
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
            split = filename.split(" ")
            if len(split)> 1:
                name = split[0]
                label = split[1]
            else:
                name = split[0]
                label = None

            tic = time.time()
            image = utils.imread(name)
            if image is None:
                continue

            img = utils.imresize(image, (cfg.input_size, cfg.input_size))

            if img.shape[2] ==3 and cfg.input_channel ==1:
                img = utils.bgr2gray(img)
                img = np.expand_dims(img, 3)

            img = np.expand_dims(img, 0)
            predictions = sess.run(model.predictions, feed_dict={model.x: img})
            tac = time.time() - tic
            if label is not None:
                logger.info("Input:{}, Target: {}, Prediction: {},  Time: {:.3f} sec.".format(name, label, predictions[0], tac))
            else:
                logger.info("Input:{}, Prediction: {}, Time: {:.3f} sec".format(name, predictions[0], tac))

            if label is not None:
                text = "T: "+str(label)+", O: "+str(predictions[0])
            else:
                text = str(predictions[0])
            utils.show_image(image=image, text=text, pause=0)


def main(_):
    config.update_config(argv=sys.argv)
    logger.info("Config: {}".format(cfg.flag_values_dict()))

    tf.logging.set_verbosity(tf.logging.INFO)
    np.random.seed(cfg.seed)
    tf.set_random_seed(cfg.seed)

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
