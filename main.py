import numpy as np
import tensorflow as tf

from tensorflow import flags

from dcgan import DCGAN
from library.utils import create_path
from library.utils import generate_log_model_dirs

FLAGS = flags.FLAGS
flags.DEFINE_float('d_learning_rate', 0.0002,
                    'Learning rate for discriminator')
flags.DEFINE_float('g_learning_rate', 0.0002,
                   'Learning rate for generator')
flags.DEFINE_integer('batch_size', 128,
                     'Number of date poitns in each training batch')
flags.DEFINE_float('keep_prob', 0.5,
                   'For generator dropout layer')
flags.DEFINE_integer('num_epochs', 25,
                     'Number of epochs for training')
flags.DEFINE_boolean('shuffle', True,
                     'Whether to shuffle the training set for each epoch')
flags.DEFINE_integer('eval_frequency', 20,
                     'Number of steps between validation set '
                     'evaluations or model file updates')
flags.DEFINE_string('root_logdir', './tf_logs/',
                    'Root directory for storing tensorboard logs')
flags.DEFINE_string('root_model_dir', './tf_models/',
                    'Root directory for storing tensorflow models')
flags.DEFINE_integer('random_state', 666,
                     'Random state or seed')
flags.DEFINE_float('beta1', 0.5,
                   'beta1 for AdamOptimizer')
flags.DEFINE_string('data_nm', 'mnist',
                    'Select from mnist and celeba')
flags.DEFINE_integer('noise_len', 100,
                     'Length of noise vector')
flags.DEFINE_integer('sample_freq', 100,
                     'Number of steps between sample pic generations')
flags.DEFINE_list('input_img_size', [28, 28, 1],
                  'Size of input image')


def main(argv=None):
    log_dir, model_dir = generate_log_model_dirs(
        FLAGS.root_logdir, FLAGS.root_model_dir)
    create_path(log_dir)
    create_path(model_dir)

    tf.reset_default_graph()
    with tf.Session() as sess:
        dcgan_nn = DCGAN(sess, log_dir, model_dir)
        dcgan_nn.build_graph(FLAGS)
        dcgan_nn.train(FLAGS)

if __name__ == '__main__':
    tf.app.run()
