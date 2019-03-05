import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from library.utils import Dataset
from library.utils import create_path
from library.utils import BatchManager
from library.data_loader import load_data
from library.utils import train_val_split
from library.data_loader import load_celeba_files

def discriminator(img_inputs, batch_size, is_train, reuse=False):

    with tf.variable_scope('discriminator', reuse=reuse):
        with tf.variable_scope('D_layer1'):
            D_cnn_1 = tf.layers.conv2d(
                img_inputs, 128, [5, 5], [2, 2], 'same',
                kernel_initializer=tf.initializers.truncated_normal(
                    stddev=0.02), name='D_cnn_1')
            D_cnn_1_lrelu = tf.nn.leaky_relu(D_cnn_1, name='D_cnn_1_lrelu')
        with tf.variable_scope('D_layer2'):
            D_cnn_2 = tf.layers.conv2d(
                D_cnn_1_lrelu, 256, [5, 5], [2, 2], 'same',
                kernel_initializer=tf.initializers.truncated_normal(
                    stddev=0.02), name='D_cnn_2')
            D_cnn_2_bn = tf.layers.batch_normalization(
                D_cnn_2, training=is_train, name='D_cnn_2_bn')
            D_cnn_2_lrelu = tf.nn.leaky_relu(D_cnn_2_bn, name='D_cnn_2_lrelu')
        with tf.variable_scope('D_layer3'):
            D_cnn_3 = tf.layers.conv2d(
                D_cnn_2_lrelu, 512, [5, 5], [2, 2], 'same',
                kernel_initializer=tf.initializers.truncated_normal(
                    stddev=0.02), name='D_cnn_3')
            D_cnn_3_bn = tf.layers.batch_normalization(
                D_cnn_3, training=is_train, name='D_cnn_3_bn')
            D_cnn_3_lrelu = tf.nn.leaky_relu(D_cnn_3_bn, name='D_cnn_3_lrelu')
        with tf.variable_scope('D_layer4'):
            D_cnn_4 = tf.layers.conv2d(
                D_cnn_3_lrelu, 1024, [5, 5], [2, 2], 'same',
                kernel_initializer=tf.initializers.truncated_normal(
                    stddev=0.02), name='D_cnn_4')
            D_cnn_4_bn = tf.layers.batch_normalization(
                D_cnn_4, training=is_train, name='D_cnn_4_bn')
            D_cnn_4_lrelu = tf.nn.leaky_relu(D_cnn_4_bn, name='D_cnn_4_lrelu')
        with tf.variable_scope('D_output'):
            D_cnn_4_reshaped = tf.reshape(D_cnn_4_lrelu, [batch_size, -1])
            D_logit_out = tf.layers.dense(
                D_cnn_4_reshaped, 1,
                kernel_initializer=tf.initializers.truncated_normal(stddev=0.02))

    return D_logit_out

def generator(noise_input, num_channels, is_train, keep_prob, random_state):

    with tf.variable_scope('generator'):
        with tf.variable_scope('projection'):
            G_proj = tf.layers.dense(
                noise_input, 4*4*1024,
                kernel_initializer=tf.initializers.random_normal(
                    stddev=0.02), name='G_proj')
            G_proj_reshaped = tf.reshape(G_proj, [-1, 4, 4, 1024], 'G_proj_reshaped')
            G_proj_bn = tf.layers.batch_normalization(
                G_proj_reshaped, training=is_train, name='G_proj_bn')
            G_proj_relu = tf.nn.relu(G_proj_bn, name='G_proj_relu')
        with tf.variable_scope('G_layer1'):
            G_cnn_1 = tf.layers.conv2d_transpose(
                G_proj_relu, 512, [5, 5], [2, 2], 'same',
                kernel_initializer=tf.initializers.random_normal(
                    stddev=0.02), name='G_cnn_1')
            G_cnn_1_bn = tf.layers.batch_normalization(
                G_cnn_1, training=is_train, name='G_cnn_1_bn')
            G_cnn_1_relu = tf.nn.relu(G_cnn_1_bn, name='G_cnn_1_relu')
            G_cnn_1_dropout = tf.layers.dropout(
                G_cnn_1_relu, rate=keep_prob, seed=random_state,
                training=is_train, name='G_cnn_1_dropout')
        with tf.variable_scope('G_layer2'):
            G_cnn_2 = tf.layers.conv2d_transpose(
                G_cnn_1_dropout, 256, [5, 5], [2, 2], 'same',
                kernel_initializer=tf.initializers.random_normal(
                    stddev=0.02), name='G_cnn_2')
            G_cnn_2_bn = tf.layers.batch_normalization(
                G_cnn_2, training=is_train, name='G_cnn_2_bn')
            G_cnn_2_relu = tf.nn.relu(G_cnn_2_bn, name='G_cnn_2_relu')
            G_cnn_2_dropout = tf.layers.dropout(
                G_cnn_2_relu, rate=keep_prob, seed=random_state,
                training=is_train, name='G_cnn_2_dropout')
        with tf.variable_scope('G_layer3'):
            G_cnn_3 = tf.layers.conv2d_transpose(
                G_cnn_2_dropout, 128, [5, 5], [2, 2], 'same',
                kernel_initializer=tf.initializers.random_normal(
                    stddev=0.02), name='G_cnn_3')
            G_cnn_3_bn = tf.layers.batch_normalization(
                G_cnn_3, training=is_train, name='G_cnn_3_bn')
            G_cnn_3_relu = tf.nn.relu(G_cnn_3_bn, name='G_cnn_3_relu')
            G_cnn_3_dropout = tf.layers.dropout(
                G_cnn_3_relu, rate=keep_prob, seed=random_state,
                training=is_train, name='G_cnn_3_dropout')
        with tf.variable_scope('G_output'):
            G_cnn_out = tf.layers.conv2d_transpose(
                G_cnn_3_dropout, num_channels, [5, 5], [2, 2], 'same',
                kernel_initializer=tf.initializers.random_normal(
                    stddev=0.02), name='G_cnn_4')
            G_cnn_tanh_out = tf.nn.tanh(G_cnn_out)

    return G_cnn_tanh_out

class DCGAN:

    def __init__(self, sess, log_dir, model_dir):
        self.sess = sess
        self.log_dir = log_dir
        self.model_dir = model_dir

    def build_graph(self, config):
        self.input_img_size = [int(i) for i in config.input_img_size]
        self.img_inputs = tf.placeholder(
            tf.float32, [config.batch_size] + self.input_img_size, 'img_inputs')
        self.noise_inputs = tf.placeholder(
            tf.float32, [config.batch_size, 100], 'noise_inputs')
        self.is_train = tf.placeholder_with_default(False, [], name='is_train')

        self.g_out = generator(
            self.noise_inputs, self.input_img_size[2], self.is_train,
            config.keep_prob, config.random_state)
        print('g_out: {}'.format(self.g_out))

        img_inputs_resized = tf.image.resize_images(self.img_inputs, [64, 64])
        img_inputs_normalized = (img_inputs_resized - 127.5) / 127.5
        d_out_real = discriminator(
            img_inputs_normalized, config.batch_size, self.is_train, reuse=False)
        print('d_out_real: {}'.format(d_out_real))
        d_out_fake = discriminator(
            self.g_out, config.batch_size, self.is_train, reuse=True)
        print('d_out_fake: {}'.format(d_out_fake))

        with tf.variable_scope('loss'):
            self.loss_d_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.random_uniform(
                    [config.batch_size, 1], 0.7, 1.2, seed=config.random_state),
                logits=d_out_real))
            self.loss_d_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.random_uniform([
                    config.batch_size, 1], 0.0, 0.3, seed=config.random_state),
                logits=d_out_fake))
            self.loss_d = 0.5 * (self.loss_d_real + self.loss_d_fake)
            self.loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(d_out_fake), logits=d_out_fake))

        all_vars = tf.trainable_variables()
        d_vars = [var for var in all_vars if 'discriminator' in var.name]
        g_vars = [var for var in all_vars if 'generator' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.variable_scope('optimization'):
            with tf.control_dependencies(update_ops):
                d_optimizer = tf.train.AdamOptimizer(
                    config.d_learning_rate, beta1=config.beta1)
                self.d_train = d_optimizer.minimize(
                    self.loss_d, var_list=d_vars, name='discriminator_optimizer')
                g_optimizer = tf.train.AdamOptimizer(
                    config.g_learning_rate, beta1=config.beta1)
                self.g_train = g_optimizer.minimize(
                    self.loss_g, var_list=g_vars, name='generator_optimizer')

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.file_writer = tf.summary.FileWriter(self.log_dir, graph=self.sess.graph)

    def generate_samples(self, samples, epoch, step, config):
        samples_path = self.model_dir+'samples/'
        create_path(samples_path)
        plt.figure(figsize=[12, 12])
        plt.subplots_adjust(wspace=0.02, hspace=0.02)
        for i in range(64):
            plt.subplot(8, 8, i+1)
            if config.data_nm == 'mnist':
                plt.imshow(samples[i][:, :, 0])
            elif config.data_nm == 'celeba':
                plt.imshow(samples[i])
            plt.axis('off')
        plt.savefig(samples_path+'sample_epoch{}_{}.png'.format(epoch, step))

    def train(self, config):
        step_counter = 1

        self.sess.run(self.init)

        print('Loading data...')
        images = load_data(config.data_nm, load_train=True)
        train_set = Dataset(images)
        batch_manager = BatchManager(
            train_set, config.num_epochs, config.shuffle, random_state=config.random_state)

        print('Training model...')
        np.random.seed(config.random_state)
        while True:
            batch_x = batch_manager.next_batch(config.batch_size)
            if config.data_nm == 'celeba':
                batch_files = batch_manager.next_batch(config.batch_size)
                batch_x = load_celeba_files(batch_files)

            noise_input_batch = np.random.uniform(
                -1, 1, size=[config.batch_size, config.noise_len])
            if batch_x is None:
                break

            if step_counter % config.eval_frequency == 1:
                loss_g_train = self.sess.run(self.loss_g, feed_dict={
                    self.noise_inputs:noise_input_batch})

                loss_d_train, loss_d_real_train, loss_d_fake_train = self.sess.run(
                    [self.loss_d, self.loss_d_real, self.loss_d_fake],
                    feed_dict={
                        self.img_inputs:batch_x,
                        self.noise_inputs:noise_input_batch})

                print('epoch: {} | g_loss: {} | d_loss: {} | d_real_loss: {}\
                      | d_fake_loss: {}'.format(
                          batch_manager.current_epoch, loss_g_train, loss_d_train,
                          loss_d_real_train, loss_d_fake_train))

                self.saver.save(
                    self.sess, self.model_dir+'gan.ckpt')

            if step_counter % config.sample_freq == 0:
                samples = self.sess.run(self.g_out, feed_dict={
                    self.noise_inputs:noise_input_batch})
                self.generate_samples(
                    samples, batch_manager.current_epoch, step_counter, config)

            self.sess.run(self.d_train, feed_dict={
                self.img_inputs:batch_x,
                self.noise_inputs:noise_input_batch,
                self.is_train:True})

            self.sess.run(self.g_train, feed_dict={
                self.img_inputs:batch_x,
                self.noise_inputs:noise_input_batch,
                self.is_train:True})

            step_counter += 1
        print('Finished!')
