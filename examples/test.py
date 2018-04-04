"""
-- User: Ashok Kumar Pant (asokpant@gmail.com)
-- Date: 3/14/18
-- Time: 10:49 AM
"""

import numpy as np


def peicewise_const():
    import tensorflow as tf
    graph = tf.get_default_graph()
    with graph.as_default():
        step = tf.placeholder(dtype=tf.int32, shape=())
        iter_per_epoch = tf.placeholder(dtype=tf.int32, shape=())
        margin = tf.train.piecewise_constant(
            tf.cast(step, dtype=tf.int32),
            boundaries=[
                (iter_per_epoch * x) for x in range(1, 8)
            ],
            values=[x / 10.0 for x in range(2, 10)]
        )

    with tf.Session() as sess:
        for s in range(1, 1000):
            m = sess.run(margin, feed_dict={step: s, iter_per_epoch: 10})
            print(m)


def update_m():
    def update_m(m, gstep, epoches):
        m_min = 0.2
        m_max = 0.9
        if gstep <= 0:
            m = m_min
        else:
            m += (m_max - m_min) / (epoches * 0.2)
            if m > m_max:
                m = m_max
        return m

    m = 0
    epoch = 5
    for s in range(1, epoch):
        m = update_m(m, s, epoch)
        print(m)


def peicewise_const1():
    m_min = 0.2
    m_max = 0.9
    m = m_min
    m_schedule = 0.2
    epoch = 5
    for s in range(1, epoch):
        print(m)
        m += (m_max - m_min) / (epoch * m_schedule)
        if m > m_max:
            m = m_max


def get_coords_to_add(dataset_name):
    options = {'mnist': ([[[8., 8.], [12., 8.], [16., 8.]],
                          [[8., 12.], [12., 12.], [16., 12.]],
                          [[8., 16.], [12., 16.], [16., 16.]]], 28.),
               'fashion_mnist': ([[[8., 8.], [12., 8.], [16., 8.]],
                                  [[8., 12.], [12., 12.], [16., 12.]],
                                  [[8., 16.], [12., 16.], [16., 16.]]], 28.),
               'smallNORB': ([[[8., 8.], [12., 8.], [16., 8.], [24., 8.]],
                              [[8., 12.], [12., 12.], [16., 12.], [24., 12.]],
                              [[8., 16.], [12., 16.], [16., 16.], [24., 16.]],
                              [[8., 24.], [12., 24.], [16., 24.], [24., 24.]]], 32.)
               }
    coord_add, scale = options[dataset_name]

    coord_add = np.array(coord_add, dtype=np.float32) / scale
    return coord_add


def coords_add_test():
    print(get_coords_to_add("mnist"))
    print(get_coords_to_add("fashion_mnist"))
    print(get_coords_to_add("smallNORB"))


if __name__ == '__main__':
    update_m()
