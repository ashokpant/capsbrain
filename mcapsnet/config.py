import os

import tensorflow as tf

flags = tf.app.flags

############################
#    hyper parameters      #
############################

flags.DEFINE_float('ac_lambda0', 0.01, '\lambda in the activation function a_c, iteration 0')
flags.DEFINE_float('ac_lambda_step', 0.01,
                   'It is described that \lambda increases at each iteration with a fixed schedule, however specific '
                   'super parameters is absent.')

flags.DEFINE_boolean('weight_reg', False, 'train with regularization of weights')
flags.DEFINE_string('norm', 'norm2', 'norm type')
# For separate margin loss
flags.DEFINE_float('m_max', 0.9, 'the parameter of m plus')
flags.DEFINE_float('m_min', 0.1, 'the parameter of m minus')
flags.DEFINE_float('m_schedule', 0.2, 'the m will get to 0.9 at current epoch')
flags.DEFINE_float('lambda_val', 0.5, 'down weight of the loss for absent digit classes')
flags.DEFINE_float('epsilon', 1e-9, 'epsilon')

# for training
flags.DEFINE_integer('batch_size', 16, 'batch size')
flags.DEFINE_integer('epoch', 5, 'epoch')
flags.DEFINE_integer('iter_routing', 3, 'number of iterations in routing algorithm')
flags.DEFINE_boolean('mask_with_y', True, 'use the true label to mask out target capsule or not')

flags.DEFINE_float('stddev', 0.01, 'stddev for W initializer')
flags.DEFINE_float('regularization_scale', 0.392,
                   'regularization coefficient for reconstruction loss, default to 0.0005*784=0.392')

################################
#    structure parameters      #
################################
flags.DEFINE_integer('A', 32, 'number of channels in output from ReLU Conv1')
flags.DEFINE_integer('B', 8, 'number of capsules in output from PrimaryCaps')
flags.DEFINE_integer('C', 16, 'number of channels in output from ConvCaps1')
flags.DEFINE_integer('D', 16, 'number of channels in output from ConvCaps2')

############################
#   environment setting    #
############################
flags.DEFINE_string('dataset', 'att_faces', 'The name of dataset [mnist, fashion_mnist, smallNORB, cifar10, cifar100, att_faces')
flags.DEFINE_string('dataset_dir', None, 'Dataset dir')
flags.DEFINE_string('num_class', None, 'Number of classes')
flags.DEFINE_string('input_size', None, 'Input image size')
flags.DEFINE_string('input_channel', None, 'Input image channels')
flags.DEFINE_string('train_size', None, 'Train samples')
flags.DEFINE_string('test_size', None, 'Test samples')
flags.DEFINE_string('mode', 'train', 'Operation mode[train, eval, predict]')
flags.DEFINE_string('input_file', 'data/image.jpg', 'Input image to predict')
flags.DEFINE_integer('num_threads', 8, 'number of threads of enqueueing examples')
flags.DEFINE_string('log_dir', 'outputs', 'logs directory')
flags.DEFINE_string('result_dir', None, 'result directory')
flags.DEFINE_string('ckpt_dir', None, 'ckpt directory')
flags.DEFINE_string('summary_dir', None, 'summary directory')
flags.DEFINE_integer('train_sum_freq', 5, 'the frequency of saving train summary(step)')
flags.DEFINE_integer('val_sum_freq', 20, 'the frequency of saving valuation summary(step)')
flags.DEFINE_integer('log_freq', 1, 'the frequency of logging(steps)')
flags.DEFINE_integer('save_freq', 10, 'the frequency of saving model(steps)')
flags.DEFINE_integer('max_steps', 100000, 'the max steps')
flags.DEFINE_string('results', 'results', 'path for saving results')
flags.DEFINE_integer('seed', 1234, "Initial random seed")
############################
#   distributed setting    #
############################
flags.DEFINE_integer('num_gpu', 2, 'number of gpus for distributed training')
flags.DEFINE_integer('batch_size_per_gpu', 128, 'batch size on 1 gpu')
flags.DEFINE_integer('thread_per_gpu', 4, 'Number of preprocessing threads per tower.')

cfg = tf.app.flags.FLAGS


# tf.logging.set_verbosity(tf.logging.INFO)

def update_cfg(dataset):
    if dataset == "mnist":
        cfg.num_class = 10
        cfg.input_size = 28
        cfg.input_channel = 1
        cfg.train_size = 60000
        cfg.test_size = 10000
        cfg.batch_size = 16
    elif dataset == "fashion_mnist":
        cfg.num_class = 10
        cfg.input_size = 28
        cfg.input_channel = 1
        cfg.train_size = 60000
        cfg.test_size = 10000
        cfg.batch_size = 16
    elif dataset == "cifar10":
        cfg.num_class = 10
        cfg.input_size = 32
        cfg.input_channel = 3
        cfg.train_size = 50000
        cfg.test_size = 10000
    elif dataset == "cifar100":
        cfg.num_class = 100
        cfg.input_size = 32
        cfg.input_channel = 3
        cfg.train_size=50000
        cfg.test_size=10000
    elif dataset == "smallNORB":
        cfg.num_class = 5
        cfg.input_size = 32
        cfg.input_channel = 1
        cfg.train_size = 23400 * 2
        cfg.test_size = 23400 * 2
    elif dataset == "att_faces":
        cfg.num_class = 40
        cfg.input_size = 48
        cfg.input_channel = 3
        cfg.train_size = 320
        cfg.test_size = 80
        cfg.batch_size = 16
    elif dataset == "casia":
        cfg.num_class = 10575
        cfg.input_size = 32
        cfg.input_channel = 3
        cfg.train_size = 789530
        cfg.test_size = 197382
        cfg.batch_size = 16
        cfg.num_threads = 16
    else:
        raise KeyError(dataset)

    cfg.result_dir = os.path.join(cfg.log_dir, dataset, 'results')
    cfg.ckpt_dir = os.path.join(cfg.log_dir, dataset, 'model')
    cfg.summary_dir = os.path.join(cfg.log_dir, dataset, 'train_log')
