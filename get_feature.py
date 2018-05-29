from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pdb
import os
import time
import re
import numpy as np
import sys
from scipy import io

from deployment import model_deploy

from nets import my_model as model

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_model_summary_secs', 600,
    'The frequency with which the model is saved and summaries are saved, in seconds.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1e-8, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'probe_dataset_dir', None, 'The directory where the probe dataset files are stored.')

tf.app.flags.DEFINE_string(
    'gallery_dataset_dir', None, 'The directory where the gallery dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

tf.app.flags.DEFINE_integer('origin_height', 128, 'origin height of image')

tf.app.flags.DEFINE_integer('origin_width', 64, 'origin width of image')

tf.app.flags.DEFINE_integer('origin_channel', 3, 'origin channel of image')

tf.app.flags.DEFINE_integer('num_classes', 751, 'num of classes')

tf.app.flags.DEFINE_integer('scale_height', 384, 'size of scale in single model')

tf.app.flags.DEFINE_integer('scale_width', 128, 'size of scale in single model')

tf.app.flags.DEFINE_string('GPU_use', 0, 'number of GPU to use')

tf.app.flags.DEFINE_bool('only_pcb', True, 'only train pcb')

tf.app.flags.DEFINE_bool('only_classifier', False, 'only train classifier')

tf.app.flags.DEFINE_integer('max_step_to_train_pcb', 100000, 'The max step you wish pcb to train')

tf.app.flags.DEFINE_integer('max_step_to_train_classifier', 40000, 'The max step you wish refined part classifier to train')

tf.app.flags.DEFINE_integer('ckpt_num', None, 'The number of ckpt model.')
#####################
# Dir Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_dir', None,
    'dir to save checkpoint')

tf.app.flags.DEFINE_string(
    'pretrain_path', None,
    'path to load pretrain model')

tf.app.flags.DEFINE_string(
  'log_dir', None, 'dir of summar')

FLAGS = tf.app.flags.FLAGS

# jh-future:it needs to be add to tf.app.flags
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.GPU_use

def init_batch(dataset_dir):
    tfrecord_list = os.listdir(dataset_dir)
    tfrecord_list = [os.path.join(dataset_dir, name) for name in tfrecord_list if name.endswith('tfrecords')]
    file_queue = tf.train.string_input_producer(tfrecord_list, num_epochs=1)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)

    features = tf.parse_single_example(serialized_example,features={
        'label': tf.FixedLenFeature([], tf.int64),
        'img' : tf.FixedLenFeature([], tf.string),
        'img_height': tf.FixedLenFeature([], tf.int64),
        'img_width': tf.FixedLenFeature([], tf.int64),
        'cam': tf.FixedLenFeature([], tf.int64)
        })

    img = tf.decode_raw(features['img'], tf.uint8)
    img_height = tf.cast(features['img_height'], tf.int32)
    img_width = tf.cast(features['img_width'], tf.int32)
    img = tf.reshape(img, tf.stack([FLAGS.origin_height, FLAGS.origin_width, FLAGS.origin_channel]))
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)

    label = features['label']
    cam = features['cam']

    images, labels, cams = tf.train.batch([img, label, cam],
        batch_size = FLAGS.batch_size,
        capacity = 3000,
        num_threads = 4,
        allow_smaller_final_batch=True
    )

    return images, labels, cams

def init_network():
    network = model.model(
        FLAGS.num_classes-FLAGS.labels_offset,
        [FLAGS.scale_height, FLAGS.scale_width],
        is_training=False,
        scope='my_model',
        global_pool=True,
        output_stride=16,
        spatial_squeeze=False,
        reuse=None
        )
    return network

def get_feature(images, labels, cams, network, sess, test_set):
    # init vars
    img_features = []
    img_labels = []
    img_cams = []

    # get feature
    for i in range(9999999):
        # batch
        try:
            batch = sess.run([images, labels, cams])
        except Exception as e:
            break

        # feed
        feed = {
            network.image:batch[0]
        }
        # calc_obj
        calc_obj = [network.feature]
        
        # run
        calc_ans = sess.run(calc_obj, feed_dict=feed)

        img_features.append(np.squeeze(calc_ans[0], axis=None))
        img_labels.append(batch[1])
        img_cams.append(batch[2])

    img_features = np.concatenate(img_features, axis=0)
    img_labels = np.concatenate(img_labels, axis=0)
    img_cams = np.concatenate(img_cams, axis=0)

    print (img_features.shape)

    # save feature
    file_path = str(FLAGS.ckpt_num)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    if test_set == 'probe':
        io.savemat(file_path + '/test_probe_features.mat', {'test_probe_features': img_features})
        io.savemat(file_path + '/test_probe_labels.mat', {'test_probe_labels': img_labels})
        io.savemat(file_path + '/queryCAM.mat', {'queryCAM': img_cams})
    elif test_set == 'gallery':
        io.savemat(file_path + '/test_gallery_features.mat', {'test_gallery_features': img_features})
        io.savemat(file_path + '/test_gallery_labels.mat', {'test_gallery_labels': img_labels})
        io.savemat(file_path + '/testCAM.mat', {'testCAM': img_cams})

def load_model(saver, sess):
    # return num of last-batch
    # if no checkpoint, return -1
    if os.path.exists(FLAGS.checkpoint_dir):
        filenames = os.listdir(FLAGS.checkpoint_dir)
        filenames = [name for name in filenames if name.endswith('index')]
        if len(filenames) > 0:
            # pattern = r'model\.ckpt\-(\d+)\.index'
            # nums = [int(re.search(pattern, name).groups()[0]) for name in filenames]
            max_num = FLAGS.ckpt_num
          
            saver.restore(sess, os.path.join(FLAGS.checkpoint_dir, 'model.ckpt-{}'.format(max_num)))
            print("[zkc]use checkpoint-{} weights".format(max_num))
            return max_num

    if os.path.exists(FLAGS.pretrain_path):
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        d = {}
        for var in variables:
            name = var.name.replace('my_model/', '').replace(':0', '')
            if name.startswith('resnet_v1_50/logits') or name.startswith('embedding'):
                continue
            d[name] = var
        saver = tf.train.Saver(d)
        saver.restore(sess, FLAGS.pretrain_path)
        print("[zkc]use pretrain init weights")
        return -1

    print("[JH]use random init weights")
    return -1

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():

        probe_images, probe_labels, probe_cams = init_batch(FLAGS.probe_dataset_dir)
        gallery_images, gallery_labels, gallery_cams = init_batch(FLAGS.gallery_dataset_dir)

        network = init_network()

        # sess
        sess_config = tf.ConfigProto()
        sess_config.allow_soft_placement=True #allow cpu calc when gpu can't
        sess_config.gpu_options.allow_growth = True
        sess = tf.Session(config=sess_config)

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        # multi-thread-read
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        # saver
        saver = tf.train.Saver()

        # load model
        last_step = load_model(saver, sess)
        
        get_feature(probe_images, probe_labels, probe_cams, network, sess, test_set='probe')
        get_feature(gallery_images, gallery_labels, gallery_cams, network, sess, test_set='gallery')

        # end
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    tf.app.run()
