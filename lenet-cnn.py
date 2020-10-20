import numpy as np
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing import image_dataset_from_directory


# main lenet class
class Lenet:
    def __init__(self):
        # initialize parameters here
        self.batch_size = 64
        self.optimizer = 'adam'
        self.epochs = 10

    def model(self):
        pass


# execute everything from here
if __name__ == '__main__':
    # download mnist dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print('Shapes ---- x_train: {}, y_train: {}'.format(x_train.shape, y_train.shape))
    print('Shapes ---- x_test: {}, y_test: {}'.format(x_test.shape, y_test.shape))

    #

    print('now everything is running well')
