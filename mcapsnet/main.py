"""
-- User: Ashok Kumar Pant (asokpant@gmail.com)
-- Date: 3/13/18
-- Time: 1:29 PM
"""
import logging
import os

import daiquiri
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import mcapsnet
from mcapsnet.config import cfg, get_dataset_size_train, get_dataset_size_test
from mcapsnet.utils import load_data, get_create_inputs

slim = tf.contrib.slim

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)


def save_to(path):
    if not os.path.exists(path):
        os.makedirs(path)
    if cfg.is_training:
        loss = path + '/loss.csv'
        train_acc = path + '/train_acc.csv'
        val_acc = path + '/val_acc.csv'

        if os.path.exists(val_acc):
            os.remove(val_acc)
        if os.path.exists(loss):
            os.remove(loss)
        if os.path.exists(train_acc):
            os.remove(train_acc)

        fd_train_acc = open(train_acc, 'w')
        fd_train_acc.write('step,train_acc\n')
        fd_loss = open(loss, 'w')
        fd_loss.write('step,loss\n')
        fd_val_acc = open(val_acc, 'w')
        fd_val_acc.write('step,val_acc\n')
        return fd_train_acc, fd_loss, fd_val_acc
    else:
        test_acc = path + '/test_acc.csv'
        if os.path.exists(test_acc):
            os.remove(test_acc)
        fd_test_acc = open(test_acc, 'w')
        fd_test_acc.write('test_acc\n')
        return fd_test_acc


def train():
    tf.logging.set_verbosity(tf.logging.INFO)

    np.random.seed(cfg.seed)
    tf.set_random_seed(cfg.seed)

    dataset_size = get_dataset_size_train(cfg.dataset)

    """Get batches per epoch."""
    train_data = get_create_inputs(cfg.dataset, True, 10)
    iterations_per_epoch = int(dataset_size / cfg.batch_size)  # 60,000/24 = 2500

    fd_train_acc, fd_loss, fd_val_acc = save_to(cfg.result_dir)
    logger.info("All of results will be saved to directory: " + cfg.result_dir)

    """Val data"""
    val_data = get_create_inputs(cfg.dataset, False, 10)
    num_val_batch = int(get_dataset_size_test(cfg.dataset) / cfg.batch_size)

    """Set summary writer"""
    if not os.path.exists(cfg.summary_dir):
        os.makedirs(cfg.summary_dir)

    # images: Tensor (?, 28, 28, 1)
    # labels: Tensor (?)
    images = train_data[0]
    labels = train_data[1]

    # Tensor(?, 10)
    one_hot_labels = slim.one_hot_encoding(labels, cfg.num_class)

    print("images: ", images)
    print("labels: ", labels)
    print("one_hot_labels: ", one_hot_labels)

    # poses: Tensor(?, 10, 4, 4) activations: (?, 10)
    poses, activations = mcapsnet.network.capsules_net(images, num_classes=cfg.num_class, iterations=3,
                                                       batch_size=cfg.batch_size, name='capsules_em')

    global_step = tf.train.get_or_create_global_step()
    loss = mcapsnet.network.spread_loss(
        one_hot_labels, activations, iterations_per_epoch, global_step, name='spread_loss'
    )
    tf.summary.scalar('losses/spread_loss', loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_tensor = slim.learning.create_train_op(
        loss, optimizer, global_step=global_step, clip_gradient_norm=4.0
    )

    slim.learning.train(

        train_tensor,
        logdir=cfg.ckpt_dir,
        log_every_n_steps=10,
        save_summaries_secs=60,
        summary_writer=tf.summary.FileWriter(logdir=cfg.summary_dir),
        saver=tf.train.Saver(max_to_keep=2),
        save_interval_secs=600,
    )


def evaluation(model, supervisor, num_label):
    teX, teY, num_te_batch = load_data(cfg.dataset, cfg.batch_size, is_training=False)
    fd_test_acc = save_to()
    with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        supervisor.saver.restore(sess, tf.train.latest_checkpoint(cfg.logdir))
        tf.logging.info('Model restored!')

        test_acc = 0
        for i in tqdm(range(num_te_batch), total=num_te_batch, ncols=70, leave=False, unit='b'):
            start = i * cfg.batch_size
            end = start + cfg.batch_size
            acc = sess.run(model.accuracy, {model.X: teX[start:end], model.labels: teY[start:end]})
            test_acc += acc
        test_acc = test_acc / (cfg.batch_size * num_te_batch)
        fd_test_acc.write(str(test_acc))
        fd_test_acc.close()
        print('Test accuracy has been saved to ' + cfg.results + '/test_acc.csv')


def main(_):
    # tf.logging.info(' Loading Graph...')
    # model = MCapsNet()
    # tf.logging.info(' Graph loaded')

    # sv = tf.train.Supervisor(graph=model.graph, logdir=cfg.logdir, save_model_secs=0)
    print(cfg)
    if cfg.is_training:
        tf.logging.info(' Start training...')
        train()
        tf.logging.info('Training done')
    else:
        evaluation(None, None, num_label=cfg.num_class)


if __name__ == "__main__":
    tf.app.run()
