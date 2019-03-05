import os
import numpy as np
import matplotlib.pyplot as plt

from mnist import MNIST

def load_mnist(path='./mnist_data/', load_train=True):
    """
    load_train: load train or test data
    """
    mndata = MNIST(path)
    if load_train:
        images, _ = mndata.load_training()
    else:
        images, _ = mndata.load_testing()
    images = np.array(images)
    images = images.reshape([-1, 28, 28, 1])
    return images

def load_celeba(path='./celeba-dataset/img_align_celeba/'):
    imgs = np.array(os.listdir(path))[:, np.newaxis]
    return imgs

def load_celeba_files(file_names, path='./celeba-dataset/img_align_celeba/'):
    file_paths = np.array([path + fnm[0] for fnm in file_names])
    imgs = []
    for img_path in file_paths:
        imgs.append(plt.imread(img_path))
    imgs = np.array(imgs)
    return imgs

def load_data(data_nm, load_train=True):
    if data_nm == 'mnist':
        return load_mnist(load_train=load_train)
    elif data_nm == 'celeba':
        return load_celeba()
