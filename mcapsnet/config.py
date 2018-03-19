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
flags.DEFINE_string('dataset', 'cifar10', 'The name of dataset [mnist, fashion-mnist')
flags.DEFINE_string('num_class', None, 'Number of classes')
flags.DEFINE_string('input_size', None, 'Input image size')
flags.DEFINE_string('input_channel', None, 'Input image channels')
flags.DEFINE_boolean('is_training', True, 'train or predict phase')
flags.DEFINE_integer('num_threads', 8, 'number of threads of enqueueing examples')
flags.DEFINE_string('log_dir', 'outputs', 'logs directory')
flags.DEFINE_string('result_dir', None, 'result directory')
flags.DEFINE_string('ckpt_dir', None, 'ckpt directory')
flags.DEFINE_string('summary_dir', None, 'summary directory')
flags.DEFINE_integer('train_sum_freq', 10, 'the frequency of saving train summary(step)')
flags.DEFINE_integer('val_sum_freq', 20, 'the frequency of saving valuation summary(step)')
flags.DEFINE_integer('save_freq', 3, 'the frequency of saving model(epoch)')
flags.DEFINE_integer('save_freq_steps', 100, 'the frequency of saving model(steps)')
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

if cfg.dataset is "mnist":
    cfg.num_class = 10
    cfg.input_size = 28
    cfg.input_channel = 1
elif cfg.dataset is "fashion_mnist":
    cfg.num_class = 10
    cfg.input_size = 28
    cfg.input_channel = 1
elif cfg.dataset is "cifar10":
    cfg.num_class = 10
    cfg.input_size = 32
    cfg.input_channel = 3
elif cfg.dataset is "cifar100":
    cfg.num_class = 10
    cfg.input_size = 32
    cfg.input_channel = 3
elif cfg.dataset is "smallNORB":
    cfg.num_class = 10
    cfg.input_size = 32
    cfg.input_channel = 1
else:
    pass

cfg.result_dir = os.path.join(cfg.log_dir, cfg.dataset, 'results')
cfg.ckpt_dir = os.path.join(cfg.log_dir, cfg.dataset, 'model')
cfg.summary_dir = os.path.join(cfg.log_dir, cfg.dataset, 'train_log')


def get_coord_add(dataset: str):
    import numpy as np
    # TODO: get coord add for cifar10/100 datasets (32x32x3)
    options = {'mnist': ([[[8., 8.], [12., 8.], [16., 8.]],
                          [[8., 12.], [12., 12.], [16., 12.]],
                          [[8., 16.], [12., 16.], [16., 16.]]], 28.),
               'fashion_mnist': ([[[8., 8.], [12., 8.], [16., 8.]],
                                  [[8., 12.], [12., 12.], [16., 12.]],
                                  [[8., 16.], [12., 16.], [16., 16.]]], 28.),
               'smallNORB': ([[[8., 8.], [12., 8.], [16., 8.], [24., 8.]],
                              [[8., 12.], [12., 12.], [16., 12.], [24., 12.]],
                              [[8., 16.], [12., 16.], [16., 16.], [24., 16.]],
                              [[8., 24.], [12., 24.], [16., 24.], [24., 24.]]], 32.),
               'cifar10': ([[[[8., 8.], [12., 8.], [16., 8.], [24., 8.]],
                             [[8., 12.], [12., 12.], [16., 12.], [24., 12.]],
                             [[8., 16.], [12., 16.], [16., 16.], [24., 16.]],
                             [[8., 24.], [12., 24.], [16., 24.], [24., 24.]]],
                            [[[8., 8.], [12., 8.], [16., 8.], [24., 8.]],
                             [[8., 12.], [12., 12.], [16., 12.], [24., 12.]],
                             [[8., 16.], [12., 16.], [16., 16.], [24., 16.]],
                             [[8., 24.], [12., 24.], [16., 24.], [24., 24.]]],
                            [[[8., 8.], [12., 8.], [16., 8.], [24., 8.]],
                             [[8., 12.], [12., 12.], [16., 12.], [24., 12.]],
                             [[8., 16.], [12., 16.], [16., 16.], [24., 16.]],
                             [[8., 24.], [12., 24.], [16., 24.], [24., 24.]]]
                            ], 32.),
               'cifar100': ([[[[8., 8.], [12., 8.], [16., 8.], [24., 8.]],
                              [[8., 12.], [12., 12.], [16., 12.], [24., 12.]],
                              [[8., 16.], [12., 16.], [16., 16.], [24., 16.]],
                              [[8., 24.], [12., 24.], [16., 24.], [24., 24.]]],
                             [[[8., 8.], [12., 8.], [16., 8.], [24., 8.]],
                              [[8., 12.], [12., 12.], [16., 12.], [24., 12.]],
                              [[8., 16.], [12., 16.], [16., 16.], [24., 16.]],
                              [[8., 24.], [12., 24.], [16., 24.], [24., 24.]]],
                             [[[8., 8.], [12., 8.], [16., 8.], [24., 8.]],
                              [[8., 12.], [12., 12.], [16., 12.], [24., 12.]],
                              [[8., 16.], [12., 16.], [16., 16.], [24., 16.]],
                              [[8., 24.], [12., 24.], [16., 24.], [24., 24.]]]
                             ], 32.)
               }
    coord_add, scale = options[dataset]

    coord_add = np.array(coord_add, dtype=np.float32) / scale

    return coord_add


def get_dataset_size_train(dataset_name: str):
    options = {'mnist': 55000, 'smallNORB': 23400 * 2,
               'fashion_mnist': 55000, 'cifar10': 50000, 'cifar100': 50000}
    return options[dataset_name]


def get_dataset_size_test(dataset_name: str):
    options = {'mnist': 10000, 'smallNORB': 23400 * 2,
               'fashion_mnist': 10000, 'cifar10': 10000, 'cifar100': 10000}
    return options[dataset_name]


def get_num_classes(dataset_name: str):
    options = {'mnist': 10, 'smallNORB': 5, 'fashion_mnist': 10, 'cifar10': 10, 'cifar100': 100}
    return options[dataset_name]
