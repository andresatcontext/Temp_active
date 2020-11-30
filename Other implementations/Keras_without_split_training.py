import tensorflow as tf
import numpy as np
from tensorflow.keras import backend, layers, models, utils, losses, regularizers
from tensorflow.keras import datasets, Sequential, preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import random
from data_utils import CIFAR10Data

# Inside my model training code
import wandb
from wandb.keras import WandbCallback


gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

from models.resnet18_paper import ResNet18


# Generate model
def generate_model(config):

    def get_embedding_nets():
        return Sequential([layers.GlobalAveragePooling2D(),layers.Dense(config.embedding_size),layers.Activation("relu")])

    
    def get_classifcation_net_vw():
        return Sequential([layers.AveragePooling2D(pool_size=(4, 4), padding='valid'),
                           layers.Flatten(),
                           layers.Dense(config.classes_data,kernel_regularizer=regularizers.l2(config.wdecay)), 
                           layers.Activation("softmax")])
    
    def get_classifcation_net():
        return Sequential([layers.GlobalAveragePooling2D(),
                           layers.Dense(config.classes_data,kernel_regularizer=regularizers.l2(config.wdecay)), 
                           layers.Activation("softmax")])

    def get_classifcation_net_v2():
        return Sequential([layers.GlobalAveragePooling2D(),layers.Dense(config.classes_data)])
    
    # generate the rest of the model
    inputs = tf.keras.Input(shape=config.input_shape)
    # add backbone
    with tf.variable_scope("backbone"):
        #ResNet18(classes, input_shape, weight_decay=1e-4)
        backbone = ResNet18(config.classes_data,config.input_shape,config.wdecay)
        print(backbone.summary())
        #backbone = ResNet18(input_shape=config.input_shape,include_top=False)
        x = backbone(inputs)
    # make normal classification
    with tf.variable_scope("classification"):
        #classification = get_classifcation_net()(x[0])
        classification = x[0]
        #classification = tf.identity(classification,'out_classification')


    with tf.variable_scope("loss_learning"):
        # generate embeddings for each other output
        embeddings_list =[]
        for out in x[1:]:
            embeddings_list.append(get_embedding_nets()(out))
        embedding = layers.Concatenate()(embeddings_list)
        #embedding = tf.identity(embedding,'out_embedding')
        out_loss = layers.Dense(1)(embedding)
        out_loss = layers.Concatenate()([classification,out_loss])
        #out_loss = tf.identity(out_loss,'out_loss_learning')

    classifier = models.Model(inputs, [classification,embedding,out_loss])
    
    return classifier

class Loss_leaning_loss(losses.Loss):
    def __init__(self, margin=1.0, reduction='mean', name="Learning_loss"):
        super().__init__(name=name)
        self.margin=1.0
        self.reduction= 'mean'


    def call(self, y_true, y_pred):  

        c_pred = y_pred[:,:-1]
        # loss prediction
        l_pred = y_pred[:,-1]
        l_pred_r = l_pred[::-1]
        #assert tf.shape(y_pred) == tf.shape(l_pred_r)

        l_pred = (l_pred - l_pred_r)[:y_pred.shape[0]//2]

        # y_true is just the classification as the l_true is calculated here
        scc = losses.CategoricalCrossentropy(reduction='none')
        class_loss = scc(y_true,c_pred)

        l_true = (class_loss - class_loss[::-1])[:class_loss.shape[0]//2]
        l_true = tf.stop_gradient(l_true)

        one = (2 * tf.math.sign(  tf.clip_by_value( l_true, 0, 1))) - 1

        if self.reduction == 'mean':
            loss = tf.reduce_sum(tf.clip_by_value(self.margin - one * l_pred, 0,10000))
            loss = tf.math.divide(loss , tf.cast(tf.shape(l_pred)[0], loss.dtype) ) # Note that the size of l_pred is already halved
        elif self.reduction == 'none':
            loss = tf.clip_by_value(self.margin - one * l_pred, 0,10000)
        else:
            NotImplementedError()

        return loss

def scheduler(epoch):
    lr= wandb.config.lr
    for i in wandb.config.MILESTONES:
        if epoch>i:
            lr*=0.1
    return lr

def mod_data_gen(generator):
    while True:
        X,Y = generator.next()
        yield X, [Y, Y]


if __name__ == "__main__":
    
    # just to select faster if run active learning or train the whole network
    active_learning = True
    
    wandb.init(project="Active Learning CIFAR 10")
    
    # Load data
    cifar10_data = CIFAR10Data()
    x_train, y_train, x_test, y_test = cifar10_data.get_data(subtract_mean=True)
    
    
    # total images for training
    wandb.config.NUM_TRAIN = len(x_train) # N
    # shape input
    wandb.config.input_shape = x_train.shape[1:]
    
    # parametres data augmentation
    wandb.config.width_shift_range = 4
    wandb.config.height_shift_range = 4
    wandb.config.horizontal_flip = True
    wandb.config.featurewise_center = False
    wandb.config.featurewise_std_normalization = False
    
    # number of classes
    wandb.config.classes_data = 10# len(np.unique(y_train))

    # common config
    # length embedding for z
    wandb.config.embedding_size = 128
    
    # 
    
    if active_learning:
        print(100*'#')
        print("Running to compute a model using active learning")
        print(100*'#')
        
        
        # how many images infer to check its importance to training
        wandb.config.SUBSET    = 10000 # M
        # from the subset select the best ADDENDUM images to train the network
        wandb.config.ADDENDUM  = 1000 # K

        # How many times train test the algorithm with different starting points (different data to train 0)
        wandb.config.TRIALS = 1
        # for how many cycles for annotation time (CYCLES*ADDENDUM = total labeled images)
        wandb.config.CYCLES = 10
        
        wandb.config.batch_size = 256

        # epoch number
        wandb.config.EPOCH = 200

        # change learning rate after the numbers in the list
        wandb.config.MILESTONES = [160]
        
        # TODO this has not been implemented
        wandb.config.EPOCHL = 120 
        # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model

        # Parameters losses
        # "learning loss" loss
        wandb.config.MARGIN = 1.0
        wandb.config.reduction ='mean' # 'none'
        # crossentropy loss
        wandb.config.lr = 1e-1
        wandb.config.momentum = 0.9
        wandb.config.wdecay = 5e-4

        # weights when adding the losses
        wandb.config.w_classif_loss = 1.0
        wandb.config.w_loss_loss = 1.0

    else:
        
        # how many images infer to check its importance to training
        wandb.config.SUBSET    = wandb.config.NUM_TRAIN # M
        # from the subset select the best ADDENDUM images to train the network
        wandb.config.ADDENDUM  = wandb.config.NUM_TRAIN # K

        # How many times train test the algorithm with different starting points (different data to train 0)
        wandb.config.TRIALS = 1
        # for how many cycles for annotation time (CYCLES*ADDENDUM = total labeled images)
        wandb.config.CYCLES = 1
        
        wandb.config.batch_size = 128

        # epoch number
        wandb.config.EPOCH = 50

        # change learning rate after the numbers in the list
        wandb.config.MILESTONES = [25, 35]
        
        # TODO this has not been implemented
        wandb.config.EPOCHL = 40 
        # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model

        # Parameters losses
        # "learning loss" loss
        wandb.config.MARGIN = 1.0
        wandb.config.reduction ='mean' # 'none'
        # crossentropy loss
        wandb.config.lr = 1e-1
        wandb.config.momentum = 0.9
        wandb.config.wdecay = 5e-4

        # weights when adding the losses
        wandb.config.w_classif_loss = 1.0
        wandb.config.w_loss_loss = 0

    

    # generate dataloader for train
    train_datagen = ImageDataGenerator(
            featurewise_center= wandb.config.featurewise_center,
            featurewise_std_normalization= wandb.config.featurewise_std_normalization,
            width_shift_range=wandb.config.width_shift_range,
            height_shift_range=wandb.config.height_shift_range,
            horizontal_flip=wandb.config.horizontal_flip)

    # generate dataloader for test
    test_datagen = ImageDataGenerator()
    
    # generate the classifier
    Classification_with_AL = generate_model(wandb.config)
    
    
    # losses
    loss_dict={Classification_with_AL.output_names[0]:tf.keras.losses.CategoricalCrossentropy(),
               Classification_with_AL.output_names[2]:Loss_leaning_loss(wandb.config.MARGIN,wandb.config.reduction)}
    
    # weigths for each loss
    weigths_dict={ Classification_with_AL.output_names[0]:wandb.config.w_classif_loss,
                   Classification_with_AL.output_names[1]:0, 
                   Classification_with_AL.output_names[2]:wandb.config.w_loss_loss}

    # metrics to compute
    metrics={ Classification_with_AL.output_names[0]:tf.keras.metrics.CategoricalAccuracy()}

    # Optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=wandb.config.lr,
                                        momentum=wandb.config.momentum, 
                                        nesterov=False)

    # callback to define a LearningRateScheduler
    callbacks = []
    # Change learning rate
    callbacks.append( tf.keras.callbacks.LearningRateScheduler(scheduler) )
    # get te values to wandb
    callbacks.append( WandbCallback() )

    # get the data to test 
    test_gen = test_datagen.flow(x_test,
                                 y_test,
                                 batch_size=wandb.config.batch_size,
                                 shuffle=False)


    # Initialize a labeled dataset by randomly sampling K=ADDENDUM=1,000 data points from the entire dataset.
    indices = list(range(wandb.config.NUM_TRAIN))
    random.shuffle(indices)
    labeled_set = indices[:wandb.config.ADDENDUM]
    unlabeled_set = indices[wandb.config.ADDENDUM:]

    # 
    Classification_with_AL.compile( optimizer=optimizer,
                                    loss=loss_dict,
                                    loss_weights=weigths_dict,
                                    metrics=metrics)
    
    # Active learning cycles
    for cycle in range(wandb.config.CYCLES):
        
        train_gen = train_datagen.flow(x_train[labeled_set],
                                   y_train[labeled_set],
                                   batch_size=wandb.config.batch_size)
        
        print(100*'#')
        print('Cycle {}/{} || Label set size {}'.format(cycle+1, wandb.config.CYCLES, len(labeled_set)))
        print("Trainig using: ",len(labeled_set),  " labeeled images")
        print(100*'#')

        # 
        history = Classification_with_AL.fit_generator(mod_data_gen(train_gen), 
                                             epochs=wandb.config.EPOCH, 
                                             steps_per_epoch= len(train_gen), 
                                             validation_data=mod_data_gen(test_gen),
                                             validation_steps=len(test_gen),
                                             validation_freq=1,
                                             callbacks=callbacks)
        
        acc =  max(history.history['val_ResNet18_categorical_accuracy'])
        wandb.log({'accuracy': acc , 'Number of labeled data': len(labeled_set)})
        
        print(100*'#')
        print('Cycle {}/{} || Label set size {} || Accuracy {}'.format(cycle+1, wandb.config.CYCLES, len(labeled_set), acc))
        print("Trainig using: ",len(labeled_set),  " labeeled images")
        print(100*'#')
        
        ##
        #  Update the labeled dataset via loss prediction-based uncertainty measurement
        #  Randomly sample 10000 unlabeled data points
        random.shuffle(unlabeled_set)
        subset = unlabeled_set[:wandb.config.SUBSET]

        classif_pred, embedding_pred, class_loss_pred = Classification_with_AL.predict(x_train[subset])

        # get the uncertanty (concatenated with class)
        uncertainty = class_loss_pred[:,-1]
        # Index in ascending order
        arg = np.argsort(uncertainty)

        # Update the labeled dataset and the unlabeled dataset, respectively
        labeled_set += list(np.array(subset)[arg][-wandb.config.ADDENDUM:])
        unlabeled_set = list(np.array(subset)[arg][:-wandb.config.ADDENDUM]) + unlabeled_set[wandb.config.SUBSET:]


        
        
