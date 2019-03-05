from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from library.utils import gan_predict

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='''Path to the trained model'''
    )
    parser.add_argument(
        '--data_nm',
        type=str,
        required=True,
        help='''mnist or celeba'''
    )
    args = parser.parse_args()

    noise_input = np.random.uniform(-1, 1, size=[128, 100])
    samples = gan_predict(args.model_path, noise_input)

    plt.figure(figsize=[12, 12])
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    for i in range(64):
        plt.subplot(8, 8, i+1)
        if args.data_nm == 'mnist':
            plt.imshow(samples[i][:, :, 0])
        elif args.data_nm == 'celeba':
            plt.imshow(samples[i])
        plt.axis('off')
    plt.show()
