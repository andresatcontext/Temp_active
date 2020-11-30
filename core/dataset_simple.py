import os
import random
import numpy as np
import tensorflow as tf



class dataset(object):
    def __init__(self):
        self.data_dict = {}

    def __getitem__(self, key):
        return self.data_dict[key]
    def get_keys(self):
        return self.data_dict.keys()
    def get_datadict(self):
        return self.data_dict

class detectionDataset(dataset):    
    def __init__(self, x_train, y_train, config):
        super(detectionDataset, self).__init__()
        
        self.batchsize   = config.batch_size
        self.num_samples = len(x_train)

        tf_data = tf.data.Dataset.from_generator(batch_generator, output_types=np.int32, output_shapes=[self.batchsize])
        tf_data = tf_data.repeat()
        
        