import os
import errno
import numpy as np
import tensorflow as tf
from datetime import datetime

def create_path(path):
    """Create path if not exist"""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def generate_log_model_dirs(root_logdir, root_model_dir):
    datetime_now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    logdir = root_logdir + 'run-{}/'.format(datetime_now)
    model_dir = root_model_dir + 'model-{}/'.format(datetime_now)
    return logdir, model_dir

def train_val_split(train_mtx, label_arr, train_proportion=0.8, random_state=666):
    np.random.seed(random_state)
    num_train_rows = np.round(train_mtx.shape[0] * train_proportion).astype(int)
    rows_selected = np.random.choice(train_mtx.shape[0],
                                     num_train_rows, replace=False)
    rows_not_selected = list(
        set(range(train_mtx.shape[0])) - set(rows_selected))

    return (train_mtx[rows_selected], train_mtx[rows_not_selected],
            label_arr[rows_selected], label_arr[rows_not_selected])


class Dataset():
    def __init__(self, X, y=None):
        self.X = X.copy()
        if y is not None:
            self.y = y.copy()


class BatchManager():

    def __init__(self, train_set, num_epochs, shuffle=True,
                 random_state=666):
        """
        train_set, val_set: RNNDataset instances
        """
        self.train_set = train_set
        self.num_epochs = num_epochs
        self.shuffle = shuffle
        self.random_state = random_state
        self.current_epoch = 0
        self.rows_in_batch = []

    def next_batch(self, batch_size):
        """
        Output next batch, return None if ran over num_epochs
        """
        num_rows = self.train_set.X.shape[0]

        while len(self.rows_in_batch) < batch_size:
            self.current_epoch += 1
            row_nums = list(range(num_rows))
            if self.shuffle:
                np.random.seed(self.random_state)
                np.random.shuffle(row_nums)
            self.rows_in_batch += row_nums

        selected_X = self.train_set.X[self.rows_in_batch[:batch_size]]
        self.rows_in_batch = self.rows_in_batch[batch_size:]

        if self.current_epoch > self.num_epochs:
            return None
        return selected_X

def gan_predict(model_path, noise_input):

    tf.reset_default_graph()

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_path+'gan.ckpt.meta')
        saver.restore(sess, model_path+'gan.ckpt')

        graph = tf.get_default_graph()
        with graph.as_default():
            g_input = graph.get_tensor_by_name('noise_inputs:0')
            g_out = graph.get_tensor_by_name('generator/G_output/Tanh:0')

            preds = sess.run(g_out, feed_dict={
                g_input:noise_input
            })

    preds = np.round(preds * 127.5 + 127.5).astype(int)

    return preds