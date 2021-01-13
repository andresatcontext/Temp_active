import tensorflow as tf
import glob as glob
import numpy as np
import os
import json

from PIL import Image
from AutoML import DataNet, AutoMLDataset, AutoML
from multiprocessing.pool import ThreadPool




class ClassificationDataset_AL:
    def __init__(self,
                 batchsize,
                 filenames,
                 labels,
                 list_classes,
                 data_augmentation=False,
                 outsize=None,
                 original_size=256,
                 pad=False,
                 subset="train",
                 random_crop = True,
                 random_crop_pad = 25,
                 random_flip = True,
                 random_hue = False,
                 random_brightness=False,
                 random_saturation=False):
        
        """Create the datagenerator for classification using tf.data.Dataset

        # Arguments
            batchsize: 
            dataset: The dataset from AutoML of the path of a directory
                (AutoML.AutoML_dataset.AutoMLDataset or path)
            data_augmentation: whether to use data augmentation or not
                (True or False, default: False).
            outsize: the width and height of the input image to the classifier
                (int, default:None)
            original_size: the size of the image get from datanet to transform
                (int, default:256)
            pad: Resizes an image to a target width and height by keeping the 
                aspect ratio the same without distortion.
                (True or False, default: False).
            sampling:
                (
                "id_sampling" :
                "test" :
                "log_proportional" :
                "equi_class" :
                )
            subset: Select if the dataloader is for training or testing
                ("train", "test")
            random_crop_margin: For data augmentation. It extracts crops from the 
                input image tensor and resizes them using bilinear sampling or 
                nearest neighbor sampling (possibly with aspect ratio change) 
                to a common output size specified by crop_size.
                (float, between 0 and 1, that correspond to the porcentage of the 
                offset to start the crop. Default:0.1)
            random_greyscale: transform the images to grey scale with a probability of 
                5%
                (True or False, default: False).
            random_hue: Adjust the hue of RGB images by a random factor of 0.1 with a 
                probability of 5%
                (True or False, default: False).
            rot90: rotate the images 90 degrees with a probability of 5%
                (True or False, default: False).
            no_image_check: When loading from path check if the file is an image.
                (True or False, default: False).

        # Returns


        # Raises

        """
        self.batchsize = batchsize
        self.side_size = original_size
        self.subset = subset
        
        print("Subset set to : "+subset)
        self.x_dim = self.side_size
        self.y_dim = self.side_size
        self.datanet = DataNet()
        self.AutoML  = AutoML()
        self.nb_parallel = 8
        self.pad = pad
        

        self.from_datanet= True
        self.source="AutoML"
        self.nb_elements = len(filenames)
        self.files = tf.constant(filenames, name="files")
        self.labels = tf.constant(labels, name="labels")
        self.list_classes = list_classes
        self.class_names=tf.constant(list(set(labels)), name="class_names")

        outsize_ = outsize

        self.real_names_classes = []
        # read the human labels for each class
        for cl in self.list_classes:
            res = self.AutoML.get_nodes(node=int(cl))
            self.real_names_classes.append(res[0]['name'])
            
        print(self.real_names_classes )
        
        # create the dataloader
        with tf.device("/cpu:0"):
            with tf.name_scope("Classification_Dataset"):


                # define if the image will be loaded as the original size or at the side size (normally 256)
                if self.pad :
                    self.datanet_size = None
                else:
                    self.datanet_size = self.side_size
                    
                # fuction to read the images from datanet
                def get_image_from_datanet(filename):
                    try:
                        image = np.array(self.datanet.get_image(filename, source=self.source, size=self.datanet_size))
                        return image
                    except Exception as e:
                        print(e)
                
                # function for the dataloader to read the images 
                def parse_function(filename, label):
                    image = tf.py_func(get_image_from_datanet, [filename], tf.uint8)
                    
                    image = tf.cast(image, dtype=tf.float32)
                    
                    image = tf.image.per_image_standardization(image)

                    #This will convert to float values in [0, 1]
                    #image = tf.image.convert_image_dtype(image, tf.float32)
                    
                    if self.pad:
                        image = tf.image.resize_image_with_pad(image, self.side_size, self.side_size)

                    #image = tf.image.resize_images(image, [self.side_size, self.side_size])
                    image = tf.reshape(image, [self.side_size, self.side_size, 3])
                        
                    if outsize_ != None:
                        image = tf.image.resize_images(image, [outsize_, outsize_])

                    label = tf.expand_dims(label, -1)

                    return image, label, filename
                
                def train_data_augmentation(image, label, filename):
                    
                    ################################
                    # random flip (normally 1/2 of the images)
                    ################################
                    if random_flip:
                        image = tf.image.random_flip_left_right(image)
                        
                    ################################
                    # Random crop to every image
                    ################################
                    if random_crop:
                        image = tf.image.resize_image_with_crop_or_pad(image, self.side_size + random_crop_pad, self.side_size + random_crop_pad) 
                        # Random crop back to the original size
                        image = tf.image.random_crop(image, size=[self.side_size, self.side_size, 3])
                        
                    ################################
                    # random_brightness
                    ################################
                    if random_brightness:
                        image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)

                    ################################
                    # random_saturation
                    ################################
                    if random_saturation:
                        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

                    ################################
                    # random hue to 5 % of the images
                    ################################
                    if random_hue:
                        image = tf.image.random_hue(image, 0.1)

                    #Make sure the image is still in [0, 1]
                    #image = tf.clip_by_value(image, 0.0, 1.0)

                    return image, label, filename
        
                dataset = tf.data.Dataset.from_tensor_slices((self.files, self.labels))
                if subset=="train":
                    dataset = dataset.shuffle(self.nb_elements,reshuffle_each_iteration=True)
                    dataset = dataset.repeat()
                dataset = dataset.map(parse_function, num_parallel_calls=self.nb_parallel)
                if data_augmentation:
                    dataset = dataset.map(train_data_augmentation, num_parallel_calls=self.nb_parallel)
                dataset = dataset.batch(batchsize)
                dataset = dataset.prefetch(10)
                
                if subset=="train":
                    self.iterator  = dataset.make_one_shot_iterator()
                else:
                    self.iterator  = dataset.make_initializable_iterator()
                
                self.next_element  = self.iterator.get_next()
                
                self.images_tensor = self.next_element[0]
                
                self.labels_tensor = {}
                self.labels_tensor['c_pred']   = self.next_element[1]
                self.labels_tensor['l_pred_w'] = self.next_element[1]
                self.labels_tensor['l_pred_s'] = self.next_element[1]

                self.files_tensor = self.next_element[2]
