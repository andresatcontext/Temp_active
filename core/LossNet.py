import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, losses, Model, Sequential, backend
# this should change for newer versions of tensorflow
import tensorflow.python.keras.metrics as metrics
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import Callback


def Lossnet(inputs_lossnet, embedding_size):
    """LossNet network"""
    def get_embedding_nets(embedding_size):
        return Sequential([layers.GlobalAveragePooling2D(),layers.Dense(embedding_size),layers.Activation("relu")])
    
    
    c_pred = inputs_lossnet[0]
    features_w = inputs_lossnet[1:]
    
    # split the foward passing of the features to be able to split the training
    # stop the gradient back to the backbone (expresed as s for split and w for whole)
    features_s= []         
    for i, out in enumerate(features_w):
        features_s.append(layers.Lambda(lambda x: backend.stop_gradient(x))(out))


    embeddings_fn_list=[]
    # generate the embeddings layers
    for feat in features_w:
        embeddings_fn_list.append(get_embedding_nets(embedding_size))
    # define dense function
    dense_fn    = layers.Dense(1,name="L_pred")
    concat_same = layers.Concatenate(name="Embedding")

    # 
    embeddings_list_whole =[]
    embeddings_list_split =[]
    for i, out in enumerate(features_w):
        embeddings_list_split.append(embeddings_fn_list[i](features_s[i]))
        embeddings_list_whole.append(embeddings_fn_list[i](features_w[i]))

    embedding_whole    =  concat_same(embeddings_list_whole)
    embedding_split    =  concat_same(embeddings_list_split)

    #l_pred_w = tf.squeeze(dense_fn(embedding_whole))
    #l_pred_s = tf.squeeze(dense_fn(embedding_split))
    
    l_pred_w = dense_fn(embedding_whole)
    l_pred_s = dense_fn(embedding_split)
    
    # concatenate the prediction of the classes with the predicted loss in order to compute the loss
    concat_w = layers.Concatenate(axis=-1,name='l_pred_w')([c_pred,l_pred_w])
    concat_s = layers.Concatenate(axis=-1,name='l_pred_s')([c_pred,l_pred_s])

    
    return [concat_w, concat_s, embedding_whole, embedding_split]

def Loss_Lossnet(c_true, y_pred):
    
    margin = 1.0
    # classification predition
    c_pred = y_pred[:, :-1]
    #print('c_pred',c_pred.shape)
    # loss prediction
    l_pred = y_pred[:, -1]
    #print('l_pred',l_pred.shape)
    # get the true loss by computing the loss between c_pred and c_true
    l_true = tf.keras.losses.sparse_categorical_crossentropy(c_true, c_pred)
    #l_true = tf.nn.softmax_cross_entropy_with_logits_v2(labels=c_true, logits=c_pred)
    #print('l_true',l_true.shape)
    # compute the classification loss non reducted
    get_batch_size = tf.shape(l_pred)[0]
    #print('get_batch_size',get_batch_size.shape)
    # 
    #l_pred = tf.squeeze(l_pred)
    #print('l_pred',l_pred.shape)
    l_pred2 = (l_pred - l_pred[::-1])[:get_batch_size//2]
    #print('l_pred',l_pred.shape)
    #
    l_true2 = (l_true - l_true[::-1] )[:get_batch_size//2]
    #print('l_true',l_true.shape)
    # value used in the lossnet loss
    one = (2 * tf.math.sign(  tf.clip_by_value( l_true2, 0, 1))) - 1
    #print('one',one.shape)
    
    temp = margin - one * l_pred2

    l_loss = tf.reduce_sum(tf.clip_by_value(temp, 0,10000))
    #print('l_loss',l_loss.shape)
    l_loss = tf.math.divide(l_loss , tf.cast(tf.shape(l_pred)[0], l_loss.dtype)) # Note that the size of l_pred is already halved
    #print('l_loss',l_loss.shape)
    return l_loss


def MAE_Lossnet(c_true, y_pred):
    # classification predition
    c_pred = y_pred[:, :-1]
    # loss prediction
    l_pred = y_pred[:, -1]
    # get the true loss by computing the loss between c_pred and c_true
    l_true = tf.keras.losses.sparse_categorical_crossentropy(c_true, c_pred)
    #l_true = tf.nn.softmax_cross_entropy_with_logits_v2(labels=c_true, logits=c_pred)
    # get 
    absolute_errors = tf.math.abs(l_true - l_pred)
    return tf.math.reduce_mean(absolute_errors)


class Change_loss_weights(Callback):
    def __init__(self, weight_w, weight_s, split_epoch, weight_lossnet_loss):
        self.weight_w = weight_w
        self.weight_s = weight_s
        self.split_epoch = split_epoch
        self.weight_lossnet_loss = weight_lossnet_loss
    # customize your behavior
    def on_epoch_end(self, epoch, logs={}):
        print(logs)
        if epoch == self.split_epoch-1:
            print("Change to split learning")
            print('Previus weigths',self.weight_w,self.weight_s)
                
        if epoch<self.split_epoch-1:
            self.weight_w = self.weight_lossnet_loss
            self.weight_s = 0
        else:
            self.weight_w = 0
            self.weight_s = self.weight_lossnet_loss
            
        if epoch == self.split_epoch-1:
            print('Updated weigths',self.weight_w,self.weight_s)
        
def add_weight_decay(model, weight_decay):
    if (weight_decay is None) or (weight_decay == 0.0):
        return

    # recursion inside the model
    def add_decay_loss(m, factor):
        if isinstance(m, Model):
            for layer in m.layers:
                add_decay_loss(layer, factor)
        else:
            for param in m.trainable_weights:
                with backend.name_scope('weight_regularizer'):
                    regularizer = lambda: tf.keras.regularizers.l2(factor)(param)
                    m.add_loss(regularizer)

    # weight decay and l2 regularization differs by a factor of 2
    add_decay_loss(model, weight_decay/2.0)
    return