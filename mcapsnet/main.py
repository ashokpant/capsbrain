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

import mcapsnet
from mcapsnet import config
from mcapsnet.config import cfg, get_dataset_size_train, get_dataset_size_test
from mcapsnet.network import CapsNet
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

    accuracy = mcapsnet.network.accuracy(activations, labels, cfg.batch_size, "train_acc")
    tf.summary.scalar('acc/train_acc', accuracy)

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


def train1():
    tf.logging.set_verbosity(tf.logging.INFO)

    np.random.seed(cfg.seed)
    tf.set_random_seed(cfg.seed)

    dataset_size = get_dataset_size_train(cfg.dataset)

    """Get batches per epoch."""
    train_data = get_create_inputs(cfg.dataset, True, 10)
    num_train_batch = int(dataset_size / cfg.batch_size)  # 60,000/24 = 2500

    fd_train_acc, fd_loss, fd_val_acc = save_to(cfg.result_dir)
    logger.info("All of results will be saved to directory: " + cfg.result_dir)

    """Val data"""
    val_data = get_create_inputs(cfg.dataset, False, 10)
    num_val_batch = int(get_dataset_size_test(cfg.dataset) / cfg.batch_size)

    summary_list = []

    """Set summary writer"""
    if not os.path.exists(cfg.summary_dir):
        os.makedirs(cfg.summary_dir)
    graph = tf.get_default_graph()
    with graph.as_default():
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
            one_hot_labels, activations, num_train_batch, global_step, name='spread_loss'
        )
        summary_list.append(tf.summary.scalar('losses/spread_loss', loss))

        accuracy = mcapsnet.network.accuracy(activations, labels, cfg.batch_size, "train_acc")
        summary_list.append(tf.summary.scalar('acc/train_acc', accuracy))

        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        train_op = slim.learning.create_train_op(
            loss, optimizer, global_step=global_step, clip_gradient_norm=4.0
        )
        # train_op = optimizer.minimize(loss, global_step=global_step)

        summary_op = tf.summary.merge(summary_list)
        fd_train_acc, fd_loss, fd_val_acc = save_to(cfg.result_dir)
        logger.info("All of results will be saved to directory: " + cfg.result_dir)

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

    """Set Session settings."""
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    """Start coord & queue."""
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # """Main loop"""
    # m = cfg.m_min
    # # for step in range(cfg.epoch * num_batches_per_epoch + 1):
    # for step in tqdm(range(cfg.epoch * num_train_batch + 1), total=cfg.epoch * num_train_batch + 1,
    #                  ncols=70, leave=False, unit='b'):
    #     if coord.should_stop():
    #         print('supervisor stopped!')
    #         break
    #
    #     tic = time.time()
    #     """"TF queue would pop batch until no file"""
    #     try:
    #         if step % cfg.train_sum_freq == 0:
    #             loss_value, train_acc, summary_str = sess.run(
    #                 [train_op, accuracy, summary_op], feed_dict={global_step: step})
    #             assert not np.isnan(loss_value), 'Something wrong! Loss is NAN'
    #             summary_writer.add_summary(summary_str, step)
    #
    #             fd_loss.write(str(step) + ',' + str(loss_value) + "\n")
    #             fd_loss.flush()
    #             fd_train_acc.write(str(step) + ',' + str(train_acc / cfg.batch_size) + "\n")
    #             fd_train_acc.flush()
    #         else:
    #             loss_value, summary_str = sess.run(
    #                 [train_op, summary_op], feed_dict={global_step: step})
    #             logger.info('%d iteration finises in ' % step + '%f second' %
    #                         (time.time() - tic) + ' loss=%f' % loss_value)
    #
    #         # if cfg.val_sum_freq != 0 and step % cfg.val_sum_freq == 0:
    #         #     val_acc = 0
    #         #     print(val_data)
    #         #     print(val_data[0],val_data[1])
    #         #     x, y = sess.run([val_data[0], val_data[1]])
    #         #     print(x.shape, y.shape)
    #         #     for i in range(num_val_batch):
    #         #         print(i)
    #         #
    #         #         acc = sess.run(model.accuracy, {model.X: x, model.labels: y})
    #         #         val_acc += acc
    #         #     val_acc = val_acc / (cfg.batch_size * num_val_batch)
    #         #     fd_val_acc.write(str(step) + ',' + str(val_acc) + '\n')
    #         #     fd_val_acc.flush()
    #     except KeyboardInterrupt:
    #         coord.should_stop()
    #         sess.close()
    #         sys.exit()
    #     except tf.errors.InvalidArgumentError as e:
    #         logger.warning('{} iteration contains NaN gradients. Discard. {}'.format(step, e))
    #         continue
    #     else:
    #         """Write to summary."""
    #         if step % 5 == 0:
    #             summary_writer.add_summary(summary_str, step)
    #
    #         """Epoch wise linear annealing."""
    #         if (step % num_train_batch) == 0:
    #             if step > 0:
    #                 m += (cfg.m_max - cfg.m_min) / (cfg.epoch * cfg.m_schedule)
    #                 if m > cfg.m_max:
    #                     m = cfg.m_max
    #
    #             """Save model periodically"""
    #             ckpt_file = os.path.join(cfg.ckpt_dir, 'model_{:.4f}.ckpt'.format(loss_value))
    #             saver.save(sess, ckpt_file, global_step=step)
    #
    # fd_val_acc.close()
    # fd_train_acc.close()
    # fd_loss.close()

    # Main loop
    for epoch in range(cfg.epoch):
        logger.info("Training for epoch {}/{}:".format(epoch, cfg.epoch))
        if coord.should_stop():
            logger.intfo('Session stopped!')
            break
        for step in tqdm(range(num_train_batch), total=num_train_batch, ncols=70, leave=False, unit='b'):
            tic = time.time()

            g_step = epoch * num_train_batch + step

            if g_step % cfg.train_sum_freq == 0:
                print(global_step, g_step)
                loss_value, train_acc, summary_str = sess.run(
                    [train_op, loss, accuracy, summary_op], feed_dict={global_step: np.int64(g_step)})
                assert not np.isnan(loss_value), 'Something wrong! loss is nan...'
                summary_writer.add_summary(summary_str, g_step)
                logger.info('%d iteration finises in ' % step + '%f second' %
                            (time.time() - tic) + ' loss=%f' % loss_value + "train_acc=%f" + train_acc)

                fd_loss.write(str(g_step) + ',' + str(loss_value) + "\n")
                fd_loss.flush()
                fd_train_acc.write(str(g_step) + ',' + str(train_acc / cfg.batch_size) + "\n")
                fd_train_acc.flush()
            else:
                loss_value, summary_str = sess.run(
                    [train_op, summary_op], feed_dict={global_step: g_step})
                logger.info('%d iteration finises in ' % step + '%f second' %
                            (time.time() - tic) + ' loss=%f' % loss_value)

            # if cfg.val_sum_freq != 0 and step % cfg.val_sum_freq == 0:
            #     val_acc = 0
            #     print(val_data)
            #     print(val_data[0],val_data[1])
            #     x, y = sess.run([val_data[0], val_data[1]])
            #     print(x.shape, y.shape)
            #     for i in range(num_val_batch):
            #         print(i)
            #
            #         acc = sess.run(model.accuracy, {model.X: x, model.labels: y})
            #         val_acc += acc
            #     val_acc = val_acc / (cfg.batch_size * num_val_batch)
            #     fd_val_acc.write(str(step) + ',' + str(val_acc) + '\n')
            #     fd_val_acc.flush()

            if (g_step + 1) % cfg.save_freq_steps == 0:
                """Save model periodically"""
                ckpt_file = os.path.join(cfg.ckpt_dir, 'model_{:.4f}.ckpt'.format(loss_value))
                saver.save(sess, ckpt_file, global_step=g_step)

        if (epoch + 1) % cfg.save_freq == 0:
            """Save model periodically"""
            ckpt_file = os.path.join(cfg.ckpt_dir, 'model_{:.4f}.ckpt'.format(loss_value))
            saver.save(sess, ckpt_file, global_step=g_step)

    fd_val_acc.close()
    fd_train_acc.close()
    fd_loss.close()
    sess.close()


def train2():
    tf.logging.set_verbosity(tf.logging.INFO)

    np.random.seed(cfg.seed)
    tf.set_random_seed(cfg.seed)

    dataset_size = get_dataset_size_train(cfg.dataset)

    """Get batches per epoch."""
    train_data = get_create_inputs(cfg.dataset, True, 10)
    num_train_batch = int(dataset_size / cfg.batch_size)  # 60,000/24 = 2500

    fd_train_acc, fd_loss, fd_val_acc = save_to(cfg.result_dir)
    logger.info("All of results will be saved to directory: " + cfg.result_dir)

    """Val data"""
    val_data = get_create_inputs(cfg.dataset, False, 10)
    num_val_batch = int(get_dataset_size_test(cfg.dataset) / cfg.batch_size)

    summary_list = []

    """Set summary writer"""
    if not os.path.exists(cfg.summary_dir):
        os.makedirs(cfg.summary_dir)
    # images: Tensor (?, 28, 28, 1)
    # labels: Tensor (?)
    images = train_data[0]
    labels = train_data[1]

    model = CapsNet(images=images, labels=labels, num_train_batch=num_train_batch)

    with model.graph.as_default():
        fd_train_acc, fd_loss, fd_val_acc = save_to(cfg.result_dir)
        logger.info("All of results will be saved to directory: " + cfg.result_dir)

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

        """Set Session settings."""
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        """Start coord & queue."""
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # """Main loop"""
        # m = cfg.m_min
        # # for step in range(cfg.epoch * num_batches_per_epoch + 1):
        # for step in tqdm(range(cfg.epoch * num_train_batch + 1), total=cfg.epoch * num_train_batch + 1,
        #                  ncols=70, leave=False, unit='b'):
        #     if coord.should_stop():
        #         print('supervisor stopped!')
        #         break
        #
        #     tic = time.time()
        #     """"TF queue would pop batch until no file"""
        #     try:
        #         if step % cfg.train_sum_freq == 0:
        #             loss_value, train_acc, summary_str = sess.run(
        #                 [train_op, accuracy, summary_op], feed_dict={global_step: step})
        #             assert not np.isnan(loss_value), 'Something wrong! Loss is NAN'
        #             summary_writer.add_summary(summary_str, step)
        #
        #             fd_loss.write(str(step) + ',' + str(loss_value) + "\n")
        #             fd_loss.flush()
        #             fd_train_acc.write(str(step) + ',' + str(train_acc / cfg.batch_size) + "\n")
        #             fd_train_acc.flush()
        #         else:
        #             loss_value, summary_str = sess.run(
        #                 [train_op, summary_op], feed_dict={global_step: step})
        #             logger.info('%d iteration finises in ' % step + '%f second' %
        #                         (time.time() - tic) + ' loss=%f' % loss_value)
        #
        #         # if cfg.val_sum_freq != 0 and step % cfg.val_sum_freq == 0:
        #         #     val_acc = 0
        #         #     print(val_data)
        #         #     print(val_data[0],val_data[1])
        #         #     x, y = sess.run([val_data[0], val_data[1]])
        #         #     print(x.shape, y.shape)
        #         #     for i in range(num_val_batch):
        #         #         print(i)
        #         #
        #         #         acc = sess.run(model.accuracy, {model.X: x, model.labels: y})
        #         #         val_acc += acc
        #         #     val_acc = val_acc / (cfg.batch_size * num_val_batch)
        #         #     fd_val_acc.write(str(step) + ',' + str(val_acc) + '\n')
        #         #     fd_val_acc.flush()
        #     except KeyboardInterrupt:
        #         coord.should_stop()
        #         sess.close()
        #         sys.exit()
        #     except tf.errors.InvalidArgumentError as e:
        #         logger.warning('{} iteration contains NaN gradients. Discard. {}'.format(step, e))
        #         continue
        #     else:
        #         """Write to summary."""
        #         if step % 5 == 0:
        #             summary_writer.add_summary(summary_str, step)
        #
        #         """Epoch wise linear annealing."""
        #         if (step % num_train_batch) == 0:
        #             if step > 0:
        #                 m += (cfg.m_max - cfg.m_min) / (cfg.epoch * cfg.m_schedule)
        #                 if m > cfg.m_max:
        #                     m = cfg.m_max
        #
        #             """Save model periodically"""
        #             ckpt_file = os.path.join(cfg.ckpt_dir, 'model_{:.4f}.ckpt'.format(loss_value))
        #             saver.save(sess, ckpt_file, global_step=step)
        #
        # fd_val_acc.close()
        # fd_train_acc.close()
        # fd_loss.close()

        # Main loop
        for epoch in range(cfg.epoch):
            logger.info("Training for epoch {}/{}:".format(epoch, cfg.epoch))
            if coord.should_stop():
                logger.intfo('Session stopped!')
                break
            for step in tqdm(range(num_train_batch), total=num_train_batch, ncols=70, leave=False, unit='b'):
                tic = time.time()

                g_step = epoch * num_train_batch + step

                if g_step % cfg.train_sum_freq == 0:
                    loss_value, train_acc, summary_str = sess.run([model.train_op, model.accuracy, model.summary_op])
                    assert not np.isnan(loss_value), 'Something wrong! loss is nan...'
                    summary_writer.add_summary(summary_str, g_step)
                    logger.info('{} iteration finises in {} second,  loss={}, train_acc={}'.format(step, time.time() - tic, loss_value, train_acc))

                    fd_loss.write(str(g_step) + ',' + str(loss_value) + "\n")
                    fd_loss.flush()
                    fd_train_acc.write(str(g_step) + ',' + str(train_acc / cfg.batch_size) + "\n")
                    fd_train_acc.flush()
                else:
                    loss_value, summary_str = sess.run([model.train_op, model.summary_op])
                    logger.info(
                        '{} iteration finises in {} second,  loss={}'.format(step, time.time() - tic,
                                                                                           loss_value))

                # if cfg.val_sum_freq != 0 and step % cfg.val_sum_freq == 0:
                #     val_acc = 0
                #     print(val_data)
                #     print(val_data[0],val_data[1])
                #     x, y = sess.run([val_data[0], val_data[1]])
                #     print(x.shape, y.shape)
                #     for i in range(num_val_batch):
                #         print(i)
                #
                #         acc = sess.run(model.accuracy, {model.X: x, model.labels: y})
                #         val_acc += acc
                #     val_acc = val_acc / (cfg.batch_size * num_val_batch)
                #     fd_val_acc.write(str(step) + ',' + str(val_acc) + '\n')
                #     fd_val_acc.flush()

                if (g_step + 1) % cfg.save_freq_steps == 0:
                    """Save model periodically"""
                    ckpt_file = os.path.join(cfg.ckpt_dir, 'model_{:.4f}.ckpt'.format(loss_value))
                    saver.save(sess, ckpt_file, global_step=g_step)

            if (epoch + 1) % cfg.save_freq == 0:
                """Save model periodically"""
                ckpt_file = os.path.join(cfg.ckpt_dir, 'model_{:.4f}.ckpt'.format(loss_value))
                saver.save(sess, ckpt_file, global_step=g_step)

    fd_val_acc.close()
    fd_train_acc.close()
    fd_loss.close()
    sess.close()


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
    config.update_cfg(cfg.dataset)
    logger.info("Config: {}".format(cfg.flag_values_dict()))
    if cfg.is_training:
        tf.logging.info(' Start training...')
        train2()
        tf.logging.info('Training done')
    else:
        evaluation(None, None, None)


if __name__ == '__main__':
    tf.app.run()
