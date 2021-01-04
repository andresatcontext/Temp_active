import tensorflow as tf
import glob as glob
import numpy as np
import os
import json

from PIL import Image
from AutoML import DataNet, AutoMLDataset
from multiprocessing.pool import ThreadPool



class AL_temp_Dataset:
    def __init__(self,
                 batchsize,
                 filenames,
                 labels,
                 data_augmentation=False,
                 outsize=None,
                 original_size=256,
                 pad=False,
                 sampling="log_proportional",
                 subset="train",
                 random_crop_margin=0.1,
                 random_greyscale=False,
                 random_hue = False,
                 rot90=False,
                 random_brightness=False,
                 random_saturation=False,
                 no_image_check=False):
        """Create the datagenerator for classification using tf.data.Dataset

        # Arguments
            batchsize: 
            dataset: The dataset from AutoML of the path of a directory
                (AutoML.AutoML_dataset.AutoMLDataset or path)
            data_augmentation: whether to use data augmentation or not
                (True or False, default: True).
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
        # print("Image check : " + str(not no_image_check))
        with tf.device("/cpu:0"):
            with tf.name_scope("Classification_Dataset"):
                self.batchsize = batchsize
                self.filenames = filenames
                self.labels = labels
                self.side_size = original_size
                self.sampling = sampling
                self.subset = subset
                print("Subset set to : "+subset)

                self.x_dim = self.side_size
                self.y_dim = self.side_size
                
                self.from_datanet= True
                self.source="AutoML"
                self.nb_elements = len(filenames)
                self.files = tf.constant(filenames, name="files")
                self.labels = tf.constant(labels, name="labels")
                self.list_classes = list(set(labels))
                self.class_names=tf.constant(list(set(labels)), name="class_names")
                
                if subset == "test":
                    def batch_generator():
                        i = 0
                        while i < self.nb_elements:
                            batch = []
                            for e in range(batchsize):
                                batch.append(i%self.nb_elements)
                                i = i+1
                            yield batch
                else:
                    file_choices = np.arange(0, self.nb_elements, dtype=np.int32)
                    def batch_generator():
                        yield np.random.choice(file_choices, size=batchsize, replace=False)
                
                tf_data = tf.data.Dataset.from_generator(batch_generator, output_types=np.int32, output_shapes=[batchsize])
                tf_data = tf_data.repeat()
                tf_data = tf_data.prefetch(1000)


                def read_op(indexs):
                    outsize_ = outsize
                    files = tf.gather(self.files, indexs)
                    if self.from_datanet:
                        datanet = DataNet()
                        nb_parallel = 8
                        if pad:
                            datanet_size = None
                        else:
                            datanet_size = original_size
                        def get_image_from_datanet(image):
                            try:
                                image = np.array(datanet.get_image(image, source=self.source, size=datanet_size))
                                return image
                            except Exception as e:
                                print(e)
                        reader = lambda x : tf.py_func(get_image_from_datanet, [x], tf.uint8)
                    else:
                        nb_parallel = 4
                        reader = lambda x : tf.image.decode_image(tf.read_file(x), channels=3)
                    if pad:
                        reader_pad = lambda x : tf.image.resize_image_with_pad(reader(x), self.side_size, self.side_size)
                        images = tf.map_fn(reader_pad, files, dtype = tf.float32, parallel_iterations=nb_parallel)
                    else:
                        images = tf.map_fn(reader, files, dtype = tf.uint8, parallel_iterations=nb_parallel)
                    images = tf.reshape(images, [self.batchsize, self.side_size, self.side_size, 3])
                    if False:
                        images = tf.image.convert_image_dtype(images, tf.float32)
                    
                    images = tf.cast(images, dtype=tf.float32)
                    if outsize_ != None:
                        images = tf.image.resize_images(images, [outsize_, outsize_])
                    else:
                        outsize_=256

                    images = tf.image.per_image_standardization(tf.cast(images, dtype=tf.dtypes.float32))
                    
                    if data_augmentation:
                        ################################
                        # Random crop to every image
                        ################################
                        marge = random_crop_margin
                        base = 1.0-random_crop_margin
                        offset = [[0,0,base,base]]
                        boxes = tf.random.uniform([batchsize,4], minval = 0.0, maxval=marge) + np.array(offset)
                        images = tf.image.crop_and_resize(images, boxes = boxes,
                                                          box_ind=np.array(range(batchsize),dtype=np.int32),
                                                          crop_size=[outsize_,outsize_])
                        ################################
                        # random_brightness
                        ################################
                        if random_brightness:
                            images = tf.image.random_brightness(images, max_delta=32.0 / 255.0)
                            
                        ################################
                        # random_saturation
                        ################################
                        if random_saturation:
                            images = tf.image.random_saturation(images, lower=0.5, upper=1.5)
                            
                        ################################
                        # random hue to 5 % of the images
                        ################################
                        if random_hue:
                            images_hue = tf.map_fn(lambda x : tf.image.random_hue(x, 0.1), images)
                            images = tf.where(tf.greater(tf.random.uniform([batchsize], minval=0.0, maxval=1.0), 0.95),
                                              x=images_hue, y=images)
                            
                        ################################
                        # random 90` rotation to 5 % of the images
                        ################################
                        if rot90:
                            for k in [1, 2, 3]:
                                images_rot = tf.map_fn(lambda x : tf.image.rot90(x,k=k), images)
                                images = tf.where(tf.greater(tf.random.uniform([batchsize], minval=0.0, maxval=1.0), 0.95),
                                                  x=images_rot, y=images)
                                
                        ################################
                        # random greyscale to 5 % of the images
                        ################################
                        if random_greyscale:
                            images_grey = tf.reduce_mean(images, axis=-1)
                            images_grey = tf.stack([images_grey, images_grey, images_grey], axis=-1)
                            images = tf.where(tf.greater(tf.random.uniform([batchsize], minval=0.0, maxval=1.0), 0.95), x = images_grey, y = images)
                        
                        ################################
                        # random flip (normally 1/2 of the images)
                        ################################
                        images = tf.image.random_flip_left_right(images)
                            
                    images = tf.clip_by_value(images, 0.0, 1.0)
                    return images
                tf_data = tf_data.map(lambda x : (read_op(x), tf.expand_dims(tf.gather(self.labels, x), -1), tf.gather(self.files, x)), num_parallel_calls=6)
                tf_data = tf_data.apply(tf.data.experimental.ignore_errors())
                tf_data = tf_data.prefetch(100)

                self.data_tensors = tf_data.make_one_shot_iterator().get_next()
                tf.summary.image("input_images", self.data_tensors[0],max_outputs=16)
                self.images_tensor = self.data_tensors[0]
                
                self.labels_tensor = {}
                self.labels_tensor['c_pred'] = self.data_tensors[1]
                self.labels_tensor['l_pred_w']   = self.data_tensors[1]
                self.labels_tensor['l_pred_s']   = self.data_tensors[1]

                self.files_tensor = self.data_tensors[2]
