import numpy as np
from mnist import MNIST

def load_mnist(path='./mnist_data/', load_train=True):
    """
    load_train: load train or test data
    """
    mndata = MNIST(path)
    if load_train:
        images, labels = mndata.load_training()
    else:
        images, labels = mndata.load_testing()
    images = np.array(images)
    labels = np.array(labels.tolist())
    images = images.reshape([-1, 28, 28, 1])
    return images, labels

def load_data(data_nm, load_train=True):
    if data_nm == 'mnist':
        return load_mnist(load_train=load_train)
