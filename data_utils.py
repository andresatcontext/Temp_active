import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
import numpy as np

class CIFAR10Data(object):
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


    def get_stretch_data(self, subtract_mean=True):
        """
        reshape X each image to row vector, and transform Y to one_hot label.
        :param subtract_mean:Indicate whether subtract mean image.
        :return: x_train, one_hot_y_train, x_test, one_hot_y_test
        """
        num_classes = len(self.classes)
        # x_train = np.reshape(self.x_train, (self.x_train.shape[0], -1)).astype('float64')
        x_train = np.reshape(self.x_train, (self.x_train.shape[0], -1)).astype('float16')
        y_train = keras.utils.to_categorical(self.y_train, num_classes)

        # x_test = np.reshape(self.x_test, (self.x_test.shape[0], -1)).astype('float64')
        x_test = np.reshape(self.x_test, (self.x_test.shape[0], -1)).astype('float16')
        y_test = keras.utils.to_categorical(self.y_test, num_classes)

        if subtract_mean:
            mean_image = np.mean(x_train, axis=0).astype('uint8')
            x_train -= mean_image
            x_test -= mean_image

        return x_train, y_train, x_test, y_test

    def get_data(self, normalize_data=False, subtract_mean= False):
        """
        The data is not reshaped, keep 3 channel.
        :param normalize_data:Indicate whether normalize the images.
        :return: x_train, one_hot_y_train, x_test, one_hot_y_test
        """
        def normalize(x_data,mean_channel=[0.4914, 0.4822, 0.4465],std_channel=[0.2023, 0.1994, 0.2010]):
            x_data = x_data/255
            for i, (mean_c, std_c) in enumerate(zip(mean_channel, std_channel)):
                x_data[:,:,:,i] = (x_data[:,:,:,i] - mean_channel[i]) / std_channel[i]
            return x_data

        
        num_classes = len(self.classes)
        x_train = self.x_train
        x_test = self.x_test


        x_train = x_train.astype('float16')
        y_train = keras.utils.to_categorical(self.y_train, num_classes)

        x_test = x_test.astype('float16')
        y_test = keras.utils.to_categorical(self.y_test, num_classes)
                 
        if normalize_data:
            x_train = normalize(x_train)
            x_test = normalize(x_test)
            
        if subtract_mean:
            mean_image = np.mean(x_train, axis=0).astype('uint8')
            x_train -= mean_image
            x_test -= mean_image

                 
        return x_train, y_train, x_test, y_test
    
