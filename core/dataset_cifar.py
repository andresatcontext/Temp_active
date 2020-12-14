from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence

class Generator_cifar_train(Sequence):
    """Wrapper of 2 ImageDataGenerator"""

    def __init__(self, x_train, y_train, config):
        
        
        # Keras generator
        self.generator = ImageDataGenerator(
                        width_shift_range=config["DATASET"]["width_shift_range"],
                        height_shift_range=config["DATASET"]["height_shift_range"],
                        horizontal_flip=config["DATASET"]["horizontal_flip"])
        

        # Real time multiple input data augmentation
        self.genX1 = self.generator.flow(x_train, y_train,  batch_size=config["TRAIN"]["batch_size"])


    def __len__(self):
        """It is mandatory to implement it on Keras Sequence"""
        return self.genX1.__len__()

    def __getitem__(self, index):
        """Getting items from the 2 generators and packing them"""
        X_batch, Y_batch = self.genX1.__getitem__(index)
        #X_batch = [X_batch]
        Y_batch = [Y_batch,Y_batch,Y_batch]
        return X_batch, Y_batch
    
    
class Generator_cifar_test(Sequence):
    """Wrapper generator"""

    def __init__(self, x_test, y_test, config):
        # Keras generator
        self.generator = ImageDataGenerator()
        
        # Real time multiple input data augmentation
        self.genX1 = self.generator.flow(x_test, y_test, batch_size=config["TRAIN"]["batch_size"])

    def __len__(self):
        """It is mandatory to implement it on Keras Sequence"""
        return self.genX1.__len__()

    def __getitem__(self, index):
        """Getting items from the 2 generators and packing them"""
        X_batch, Y_batch = self.genX1.__getitem__(index)
        #X_batch = [X_batch]
        Y_batch = [Y_batch,Y_batch,Y_batch]
        return X_batch, Y_batch