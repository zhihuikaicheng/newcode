from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pdb

from nets import resnet_v1

FLAGS = tf.app.flags.FLAGS
slim = tf.contrib.slim

def LeakyRelu(x, leak=0.1):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * tf.abs(x)

class model():
    def __init__(self, num_classes, size, scope, is_training, output_stride, global_pool=True, spatial_squeeze=True, reuse=None):
        self.num_classes = num_classes
        self.size = size
        self.scope = scope
        self.is_training = is_training
        self.output_stride = output_stride
        self.spatial_squeeze = spatial_squeeze
        self.reuse = reuse
        self.global_pool = global_pool
        with tf.variable_scope(scope):
            self.init_input()
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                self.init_network()
            self.init_loss()

    def init_input(self):
        self.image = tf.placeholder(tf.float32, [None, FLAGS.origin_height, FLAGS.origin_width, FLAGS.origin_channel])
        self.label = tf.placeholder(tf.float32, [None, self.num_classes])

    def init_network(self):
        image = tf.image.resize_images(self.image, self.size, 0)
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        # bone network
        net, end_points = resnet_v1.resnet_v1_50(
            image,
            is_training=self.is_training,
            global_pool=self.global_pool,
            output_stride=self.output_stride,
            spatial_squeeze=self.spatial_squeeze,
            num_classes=self.num_classes,
            reuse=self.reuse,
            scope='resnet_v1_50'
            )

        self.feature = end_points['global_pool']
        
        # embedding
        # with tf.variable_scope('embedding'):
        #     net = end_points['global_pool']
        #     net = slim.flatten(net)
        #     net = slim.fully_connected(net, 512, activation_fn=None)
        #     net = slim.batch_norm(net, activation_fn=None)
        #     net = LeakyRelu(net, leak=0.1)
        #     net = slim.dropout(net, 0.5)
        #     net = slim.fully_connected(net, self.num_classes, activation_fn=None, scope='logits')

        # pred = slim.softmax(net)
        # end_points['logits'] = net
        # end_points['prediction'] = pred
        self.end_points = end_points

    def init_loss(self):
        cross_entropy = -tf.reduce_sum(self.label*tf.log(self.end_points['predictions'] + FLAGS.opt_epsilon), axis=1)
        cross_entropy = tf.reduce_mean(cross_entropy)

        regular_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        regularizers = tf.add_n(regular_vars)

        self.loss = cross_entropy + FLAGS.weight_decay * regularizers

        # calc acc
        corr_pred = tf.equal(tf.argmax(self.label,1), tf.argmax(self.end_points['predictions'],1))
        self.acc = tf.reduce_sum(tf.cast(corr_pred, tf.int32))
