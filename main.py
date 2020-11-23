import tensorflow as tf
import numpy as np
from tensorflow.keras import backend, layers, models, utils
from tensorflow.keras import datasets, Sequential, preprocessing
from PIL import Image
# Inside my model training code
import config from wandb
from wandb.keras import WandbCallback

from models.resnet import ResNet18


# Generate model
def generate_model(config):

    def get_embedding_nets():
        return Sequential([layers.GlobalAveragePooling2D(),layers.Dense(config.embedding_size),layers.Activation("relu")])

    def get_classifcation_net():
        return Sequential([layers.GlobalAveragePooling2D(),layers.Dense(config.classes_data),layers.Activation("softmax")])

    def get_classifcation_net():
        return Sequential([layers.GlobalAveragePooling2D(),layers.Dense(config.classes_data),layers.Activation("softmax")])
    
    # generate the rest of the model
    inputs = tf.keras.Input(shape=config.input_shape)
    # add backbone
    with tf.variable_scope("backbone"):
        backbone = ResNet18(input_shape=config.input_shape, weights='imagenet',include_top=False)
        x = backbone(inputs)
    # make normal classification
    with tf.variable_scope("classification"):
        classification = get_classifcation_net()(x[0])
        classification = tf.identity(classification,'out_classification')


    with tf.variable_scope("loss_learning"):
        # generate embeddings for each other output
        embeddings_list =[]
        for out in x[1:]:
            embeddings_list.append(get_embedding_nets()(out))
        embedding = layers.Concatenate()(embeddings_list)
        embedding = tf.identity(embedding,'out_embedding')
        out_loss = layers.Dense(1)(embedding)
        out_loss = layers.Concatenate()([classification,out_loss])
        out_loss = tf.identity(out_loss,'out_loss_learning')

    classifier = models.Model(inputs, [classification,embedding,out_loss])
    
    return classifier


class Loss_leaning_loss(tf.keras.losses.Loss):
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
        scc = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')
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
    for i in wandb.config.milestones:
        if epoch>i:
            lr*=0.1
    return lr

def mod_data_gen(generator):
    while True:
        X,Y = generator.next()
        yield X, [Y, Y]


if __name__ == "__main__":
    
    # Load data
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

    wandb.config.input_shape = x_train.shape[1:]
    wandb.config.classes_data = len(np.unique(y_train))

    # generate dataloader for train
    train_datagen = ImageDataGenerator(
            width_shift_range=[-2,2],
            height_shift_range=[-2,2],
            horizontal_flip=True)

    # generate dataloader for test
    test_datagen = ImageDataGenerator()
       
    # generate the classifier
    Classification_with_AL = generate_model(wandb.config)
    
    # losses
    loss_dict={Classification_with_AL.output_names[0]:tf.keras.losses.SparseCategoricalCrossentropy(),
               Classification_with_AL.output_names[2]:Loss_leaning_loss()}
    
    # weigths for each loss
    weigths_dict={ Classification_with_AL.output_names[0]:wandb.config.w_classif_loss,
                   Classification_with_AL.output_names[1]:0, 
                   Classification_with_AL.output_names[2]:wandb.config.w_loss_loss}

    # metrics to compute
    metrics={ Classification_with_AL.output_names[0]:tf.keras.metrics.SparseCategoricalAccuracy()}

    # Optimizer
    optimizer = tf.keras.optimizers.SGD( learning_rate=wandb.config.lr, momentum=wandb.config.momentum)

    # callback to define a LearningRateScheduler
    callbacks = []
    # Change learning rate
    callbacks.append( tf.keras.callbacks.LearningRateScheduler(scheduler) )
    # get te values to wandb
    callbacks.append( WandbCallback()   )
    # save the best model
    #callbacks.append( tf.keras.callbacks.ModelCheckpoint( filepath='model.{epoch:02d}-{val_loss:.2f}.h5'), save_freq='5', **kwargs  )
    

    # get the data to test 
    test_gen = test_datagen.flow(x_test,y_test,batch_size=wandb.config.batch_size,shuffle=False)
    
    
    # 
    train_gen = train_datagen.flow(x_train,y_train,batch_size=wandb.config.batch_size)

    Classification_with_AL.compile(  optimizer=optimizer,
                                    loss=loss_dict,
                                    loss_weights=weigths_dict,
                                    metrics=metrics)
    
    Classification_with_AL.fit_generator(mod_data_gen(train_gen), 
                                         epochs=wandb.config.epoch, 
                                         steps_per_epoch= len(train_gen), 
                                         validation_data=mod_data_gen(test_gen),
                                         validation_steps=len(test_gen),
                                         validation_freq=10,
                                         callbacks=callbacks)

