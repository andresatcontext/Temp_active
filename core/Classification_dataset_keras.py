import tensorflow as tf
import glob as glob
import numpy as np
import os
import json

from PIL import Image
from AutoML import DataNet, AutoMLDataset
from multiprocessing.pool import ThreadPool

from tensorflow.keras.utils import Sequence

class Generator_cifar_train(Sequence):
    'Generates data for Keras'
    def __init__(self, dataset, config, shuffle=True):
        
        self.batchsize = batchsize
        self.path = dataset
        self.side_size = original_size
        self.sampling = sampling
        self.subset = subset
        print("Subset set to : "+subset)
        self.load_metadata(no_image_check)

        self.x_dim = self.side_size
        self.y_dim = self.side_size
        
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
    
    def load_metadata(self, no_image_check):
        
        # check if the path is in AutoML
        if isinstance(self.path, AutoMLDataset):
            self.dataset_header = self.path.get_dataset_header()
            self.from_datanet = True
            if "source" in self.dataset_header:
                self.source = self.dataset_header["source"]
            else:
                self.source="AutoML"
        else:
            self.from_datanet = False
            self.source = None
            id_directories = sorted(glob.glob(os.path.join(self.path, self.subset, "*/")))
            dataset_header = {}
            dataset_header[self.subset+"_images"] = {}
            
            for i in range(len(id_directories)):
                id_dir = id_directories[i]
                splt_dir = id_dir.split("/")
                class_name = splt_dir[-2].replace("\n", "")

                id_files = glob.glob(id_dir+"*.jpg") + glob.glob(id_dir+"*.png")
                if not no_image_check:
                    valid_id_files = []
                    for img in id_files:
                        try:
                            im = Image.open(img)
                            valid_id_files.append(img)
                        except Exception as e:
                            pass
                    id_files = valid_id_files
                dataset_header[self.subset+"_images"][class_name] = id_files        
            self.dataset_header = dataset_header
                    
        
        list_classes = list(self.dataset_header[self.subset+"_images"].keys())
                                    
        self.nb_classes = len(list_classes)        
        self.classes={}
        files = []
        labels = []
        nb_img_per_class = np.zeros(self.nb_classes, dtype = np.int32)
        classes_cardinality = {}
        
        for class_name in list(self.dataset_header[self.subset+"_images"].keys()):
            class_index = list_classes.index(class_name)
            classes_cardinality[class_name] = len(self.dataset_header[self.subset+"_images"][class_name])
            nb_img_per_class[class_index] = classes_cardinality[class_name]
            start = len(files)
            id_files = self.dataset_header[self.subset+"_images"][class_name]
            files+=id_files
            labels+=[class_index] * len(id_files)
            end = len(files)
            self.classes[class_index]=range(start, end)

        nb_img_per_class = np.array(nb_img_per_class)
        #probas = np.array(np.minimum(nb_img_per_class, 50), dtype=np.float32)
        if self.sampling == "log_proportional":
            probas = np.square(np.maximum(np.log(nb_img_per_class)+1.0,0.001))
        elif self.sampling == "equi_class":
            probas = 1.0 / nb_img_per_class
        elif self.sampling == "id_sampling":
            # probas = np.array(nb_img_per_class, dtype=np.float32)
            probas = np.square(np.maximum(np.log(nb_img_per_class) + 1.0, 0.001))
        else:
            probas = np.square(np.maximum(np.log(nb_img_per_class) + 1.0, 0.001))
        probas = probas / np.sum(probas)
        files_proba = []
        for p, class_card in zip(probas, nb_img_per_class):
            files_proba += [ p for i in range(class_card)]
        self.files_proba = np.array(files_proba)
        self.files_proba = files_proba / np.sum(files_proba)
        self.class_probas = probas

        assert len(files) > 0

        self.nb_elements = len(files)
        self.files = tf.constant(files, name="files")
        self.labels = tf.constant(labels, name="labels")
        self.class_names=tf.constant(list_classes, name="class_names")
        self.list_classes = list_classes

    

class ClassificationDataset:
    def __init__(self,
                 batchsize,
                 dataset,
                 data_augmentation=True,
                 outsize=None,
                 original_size=256,
                 pad=False,
                 sampling="log_proportional",
                 subset="train",
                 random_crop_margin=0.1,
                 random_greyscale=False,
                 random_hue = False,
                 rot90=False,
                 no_image_check=False):
        # print("Image check : " + str(not no_image_check))
        with tf.device("/cpu:0"):
            with tf.name_scope("Classification_Dataset"):
                self.batchsize = batchsize
                self.path = dataset
                self.side_size = original_size
                self.sampling = sampling
                self.subset = subset
                print("Subset set to : "+subset)
                self.load_metadata(no_image_check)

                self.x_dim = self.side_size
                self.y_dim = self.side_size

                if sampling == "id_sampling":
                    class_choice = np.arange(0, self.nb_classes, dtype=np.int32)

                    def batch_generator():
                        r_classes = np.random.choice(class_choice, size=batchsize, replace=False, p=self.class_probas)
                        batch = []
                        i = 0
                        nb_elements = 0
                        while len(batch) < self.batchsize:
                            nb_samples = min(batchsize - nb_elements,len(self.classes[r_classes[i]]),np.random.randint(low=1,high=5))
                            batch.append(np.random.choice(self.classes[r_classes[i]], size=nb_samples, replace=False))
                            nb_elements += nb_samples
                            i+=1
                        yield np.concatenate(batch, axis=0)
                elif sampling == "test":
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
                        yield np.random.choice(file_choices, size=batchsize, replace=False, p=self.files_proba)

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
                    images = tf.cast(images, dtype=tf.float32)
                    if outsize_ != None:
                        images = tf.image.resize_images(images, [outsize_, outsize_])
                    else:
                        outsize_=256

                    images = tf.image.per_image_standardization(tf.cast(images, dtype=tf.dtypes.float32))
                    if data_augmentation:
                        marge = random_crop_margin
                        base = 1.0-random_crop_margin
                        offset = [[0,0,base,base]]
                        boxes = tf.random.uniform([batchsize,4], minval = 0.0, maxval=marge) + np.array(offset)
                        images = tf.image.crop_and_resize(images, boxes = boxes,
                                                          box_ind=np.array(range(batchsize),dtype=np.int32),
                                                          crop_size=[outsize_,outsize_])
                        if random_hue:
                            images_hue = tf.map_fn(lambda x : tf.image.random_hue(x, 0.1), images)
                            images = tf.where(tf.greater(tf.random.uniform([batchsize], minval=0.0, maxval=1.0), 0.95),
                                              x=images_hue, y=images)
                        if rot90:
                            for k in [1, 2, 3]:
                                images_rot = tf.map_fn(lambda x : tf.image.rot90(x,k=k), images)
                                images = tf.where(tf.greater(tf.random.uniform([batchsize], minval=0.0, maxval=1.0), 0.95),
                                                  x=images_rot, y=images)
                        images = tf.image.random_flip_left_right(images)
                        if random_greyscale:
                            images_grey = tf.reduce_mean(images, axis=-1)
                            images_grey = tf.stack([images_grey, images_grey, images_grey], axis=-1)
                            images = tf.where(tf.greater(tf.random.uniform([batchsize], minval=0.0, maxval=1.0), 0.95), x = images_grey, y = images)
                    return images
                tf_data = tf_data.map(lambda x : (read_op(x), tf.gather(self.labels, x), tf.gather(self.files, x)), num_parallel_calls=6)
                tf_data = tf_data.apply(tf.contrib.data.ignore_errors())
                tf_data = tf_data.prefetch(100)

                data_tensors = tf_data.make_one_shot_iterator().get_next()
                tf.summary.image("input_images", data_tensors[0],max_outputs=16)
                self.images_tensor = data_tensors[0]
                self.labels_tensor = data_tensors[1]
                self.files_tensor = data_tensors[2]

    def load_metadata(self, no_image_check):
        
        # checck if the path is in AutoML
        
        if isinstance(self.path, AutoMLDataset):
            self.dataset_header = self.path.get_dataset_header()
            self.from_datanet = True
            if "source" in self.dataset_header:
                self.source = self.dataset_header["source"]
            else:
                self.source="AutoML"
        else:
            self.from_datanet = False
            self.source = None
            id_directories = sorted(glob.glob(os.path.join(self.path, self.subset, "*/")))
            dataset_header = {}
            dataset_header[self.subset+"_images"] = {}
            
            for i in range(len(id_directories)):
                id_dir = id_directories[i]
                splt_dir = id_dir.split("/")
                class_name = splt_dir[-2].replace("\n", "")

                id_files = glob.glob(id_dir+"*.jpg") + glob.glob(id_dir+"*.png")
                if not no_image_check:
                    valid_id_files = []
                    for img in id_files:
                        try:
                            im = Image.open(img)
                            valid_id_files.append(img)
                        except Exception as e:
                            pass
                    id_files = valid_id_files
                dataset_header[self.subset+"_images"][class_name] = id_files        
            self.dataset_header = dataset_header
                    
        
        list_classes = list(self.dataset_header[self.subset+"_images"].keys())
                                    
        self.nb_classes = len(list_classes)        
        self.classes={}
        files = []
        labels = []
        nb_img_per_class = np.zeros(self.nb_classes, dtype = np.int32)
        classes_cardinality = {}
        
        for class_name in list(self.dataset_header[self.subset+"_images"].keys()):
            class_index = list_classes.index(class_name)
            classes_cardinality[class_name] = len(self.dataset_header[self.subset+"_images"][class_name])
            nb_img_per_class[class_index] = classes_cardinality[class_name]
            start = len(files)
            id_files = self.dataset_header[self.subset+"_images"][class_name]
            files+=id_files
            labels+=[class_index] * len(id_files)
            end = len(files)
            self.classes[class_index]=range(start, end)

        nb_img_per_class = np.array(nb_img_per_class)
        #probas = np.array(np.minimum(nb_img_per_class, 50), dtype=np.float32)
        if self.sampling == "log_proportional":
            probas = np.square(np.maximum(np.log(nb_img_per_class)+1.0,0.001))
        elif self.sampling == "equi_class":
            probas = 1.0 / nb_img_per_class
        elif self.sampling == "id_sampling":
            # probas = np.array(nb_img_per_class, dtype=np.float32)
            probas = np.square(np.maximum(np.log(nb_img_per_class) + 1.0, 0.001))
        else:
            probas = np.square(np.maximum(np.log(nb_img_per_class) + 1.0, 0.001))
        probas = probas / np.sum(probas)
        files_proba = []
        for p, class_card in zip(probas, nb_img_per_class):
            files_proba += [ p for i in range(class_card)]
        self.files_proba = np.array(files_proba)
        self.files_proba = files_proba / np.sum(files_proba)
        self.class_probas = probas

        assert len(files) > 0

        self.nb_elements = len(files)
        self.files = tf.constant(files, name="files")
        self.labels = tf.constant(labels, name="labels")
        self.class_names=tf.constant(list_classes, name="class_names")
        self.list_classes = list_classes

    def load_metadata_active_learning(self, no_image_check):
        
        # checck if the path is in AutoML
        
        if isinstance(self.path, AutoMLDataset):
            self.dataset_header = self.path.get_dataset_header()
            self.from_datanet = True
            if "source" in self.dataset_header:
                self.source = self.dataset_header["source"]
            else:
                self.source="AutoML"
        else:
            self.from_datanet = False
            self.source = None
            id_directories = sorted(glob.glob(os.path.join(self.path, self.subset, "*/")))
            dataset_header = {}
            dataset_header[self.subset+"_images"] = {}
            
            for i in range(len(id_directories)):
                id_dir = id_directories[i]
                splt_dir = id_dir.split("/")
                class_name = splt_dir[-2].replace("\n", "")

                id_files = glob.glob(id_dir+"*.jpg") + glob.glob(id_dir+"*.png")
                if not no_image_check:
                    valid_id_files = []
                    for img in id_files:
                        try:
                            im = Image.open(img)
                            valid_id_files.append(img)
                        except Exception as e:
                            pass
                    id_files = valid_id_files
                dataset_header[self.subset+"_images"][class_name] = id_files        
            self.dataset_header = dataset_header
                    
        
        list_classes = list(self.dataset_header[self.subset+"_images"].keys())
                                    
        self.nb_classes = len(list_classes)        
        self.classes={}
        files = []
        labels = []
        nb_img_per_class = np.zeros(self.nb_classes, dtype = np.int32)
        classes_cardinality = {}
        
        for class_name in list(self.dataset_header[self.subset+"_images"].keys()):
            class_index = list_classes.index(class_name)
            classes_cardinality[class_name] = len(self.dataset_header[self.subset+"_images"][class_name])
            nb_img_per_class[class_index] = classes_cardinality[class_name]
            start = len(files)
            id_files = self.dataset_header[self.subset+"_images"][class_name]
            files+=id_files
            labels+=[class_index] * len(id_files)
            end = len(files)
            self.classes[class_index]=range(start, end)

        nb_img_per_class = np.array(nb_img_per_class)
        #probas = np.array(np.minimum(nb_img_per_class, 50), dtype=np.float32)
        if self.sampling == "log_proportional":
            probas = np.square(np.maximum(np.log(nb_img_per_class)+1.0,0.001))
        elif self.sampling == "equi_class":
            probas = 1.0 / nb_img_per_class
        elif self.sampling == "id_sampling":
            # probas = np.array(nb_img_per_class, dtype=np.float32)
            probas = np.square(np.maximum(np.log(nb_img_per_class) + 1.0, 0.001))
        else:
            probas = np.square(np.maximum(np.log(nb_img_per_class) + 1.0, 0.001))
        probas = probas / np.sum(probas)
        files_proba = []
        for p, class_card in zip(probas, nb_img_per_class):
            files_proba += [ p for i in range(class_card)]
        self.files_proba = np.array(files_proba)
        self.files_proba = files_proba / np.sum(files_proba)
        self.class_probas = probas

        assert len(files) > 0

        self.nb_elements = len(files)
        self.files = tf.constant(files, name="files")
        self.labels = tf.constant(labels, name="labels")
        self.class_names=tf.constant(list_classes, name="class_names")
        self.list_classes = list_classes