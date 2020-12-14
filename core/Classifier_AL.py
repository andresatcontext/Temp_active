import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, losses, metrics, Model, Sequential, backend
from tensorflow.keras.layers import Layer

def Lossnet_keras(loss_net_inputs, embedding_size):
    
    def get_embedding_nets(embedding_size):
        return Sequential([layers.GlobalAveragePooling2D(),layers.Dense(embedding_size),layers.Activation("relu")])

    embeddings_list =[]
    for out in loss_net_inputs:
        embeddings_list.append(get_embedding_nets(128)(out))
    embedding = layers.Concatenate()(embeddings_list)
    #embedding = tf.identity(embedding,'out_embedding')
    out_loss = layers.Dense(1)(embedding)

    return Model(inputs=loss_net_inputs, outputs=[out_loss, embedding],name='LossNet')


def Loss_fn(c_loss_nr, l_pred ,margin=1.0, reduction= "mean"):
    # compute the classification loss non reducted
    get_batch_size = tf.shape(l_pred)[0]

 
    l_true = (c_loss_nr - c_loss_nr[::-1] )[:get_batch_size//2]
    # get the value without the gradient
    l_true = tf.stop_gradient(l_true)

    # value used in the lossnet loss
    one = (2 * tf.math.sign(  tf.clip_by_value( l_true, 0, 1))) - 1

    l_pred = (l_pred - l_pred[::-1])[:get_batch_size//2]
    if reduction == 'mean':
        l_loss = tf.reduce_sum(tf.clip_by_value(margin - one * l_pred, 0,10000))
        l_loss = tf.math.divide(l_loss , tf.cast(tf.shape(l_pred)[0], l_loss.dtype) ) # Note that the size of l_pred is already halved
    elif reduction == 'none':
        l_loss = tf.clip_by_value(margin - one * l_pred, 0,10000)
    else:
        NotImplementedError()

    return l_loss


class LossNet_layer(Layer):
    """Layer generates the Lossnet"""

    def __init__(self, embedding_size=128, number_feats=4):
        super(LossNet_layer, self).__init__()
        
        self.embedding_size = embedding_size
        
        def get_embedding_nets(embedding_size,i):
            return Sequential([layers.GlobalAveragePooling2D(name='Feat_GAP_'+str(i)),
                               layers.Dense(embedding_size,name='Dense_GAP_'+str(i)),
                               layers.Activation("relu",name='Activatin_GAP_'+str(i))])
        
        # generate the embeddings layers
        for i in range(number_feats):
            setattr(self, 'feat_'+str(i), get_embedding_nets(embedding_size,i))
            setattr(self, 'Stop_grad_'+str(i), layers.Lambda(lambda x: backend.stop_gradient(x)))


        self.dense_fn = layers.Dense(1)
        self.concat_same = layers.Concatenate()


    def call(self, features_w):
        
        features_s= []                
        for i, out in enumerate(features_w):
            features_s.append(tf.stop_gradient(tf.identity( out, name="Copy_features_"+str(i)),name="stop_grad_"+str(i)))

        embeddings_list_whole =[]
        embeddings_list_split =[]
        for i, feat_w in enumerate(features_w):
            feat_s = getattr(self,'Stop_grad_'+str(i))(feat_w)
            embeddings_list_split.append(getattr(self,'feat_'+str(i))(feat_s))
            embeddings_list_whole.append(getattr(self,'feat_'+str(i))(feat_w))

        embedding_whole    =  self.concat_same(embeddings_list_whole)
        embedding_split    =  self.concat_same(embeddings_list_split)

        l_pred_w = tf.squeeze(self.dense_fn(embedding_whole))
        l_pred_s = tf.squeeze(self.dense_fn(embedding_split))

        return l_pred_w, l_pred_s, embedding_whole, embedding_split

        
def Loss_fn(c_loss_nr, l_pred, margin=1.0, reduction= "mean"):
    
    # compute the classification loss non reducted
    get_batch_size = tf.shape(l_pred)[0]

    l_true = (c_loss_nr - c_loss_nr[::-1] )[:get_batch_size//2]
    # get the value without the gradient
    l_true = tf.stop_gradient(l_true)

    # value used in the lossnet loss
    one = (2 * tf.math.sign(  tf.clip_by_value( l_true, 0, 1))) - 1

    l_pred = (l_pred - l_pred[::-1])[:get_batch_size//2]
    if reduction == 'mean':
        l_loss = tf.reduce_sum(tf.clip_by_value(margin - one * l_pred, 0,10000))
        l_loss = tf.math.divide(l_loss , tf.cast(tf.shape(l_pred)[0], l_loss.dtype) ) # Note that the size of l_pred is already halved
    elif reduction == 'none':
        l_loss = tf.clip_by_value(margin - one * l_pred, 0,10000)
    else:
        NotImplementedError()

    return l_loss



class Learning_loss_loss(object):
    def __init__(self, margin=1.0, reduction= "mean"):
        self.margin    = margin
        self.reduction = reduction

    def __call__(self, c_true, l_pred):
        # compute the classification loss non reducted
        get_batch_size = tf.shape(l_pred)[0]

        # 
        l_pred = tf.squeeze(l_pred)
        l_pred = (l_pred - l_pred[::-1])[:get_batch_size//2]
        
        #
        l_true = (c_true - c_true[::-1] )[:get_batch_size//2]

        # value used in the lossnet loss
        one = (2 * tf.math.sign(  tf.clip_by_value( l_true, 0, 1))) - 1

        if   self.reduction == 'mean':
            l_loss = tf.reduce_sum(tf.clip_by_value(self.margin - one * l_pred, 0,10000))
            l_loss = tf.math.divide(l_loss , tf.cast(tf.shape(l_pred)[0], l_loss.dtype)) # Note that the size of l_pred is already halved
        elif self.reduction == 'none':
            l_loss = tf.clip_by_value(self.margin - one * l_pred, 0,10000)
        else:
            NotImplementedError()

        return l_loss

class Learning_loss_concat(object):
    def __init__(self, margin=1.0, reduction= "mean", name="learning_loss"):
        self.margin    = margin
        self.reduction = reduction
        self.name = name
        
    def __call__(self, y_true, y_pred):
        margin=1.0
        l_true = y_pred[:, 0]
        l_pred = y_pred[:, 1]
        
        # compute the classification loss non reducted
        get_batch_size = tf.shape(l_pred)[0]

        # 
        l_pred = tf.squeeze(l_pred)
        l_pred = (l_pred - l_pred[::-1])[:get_batch_size//2]
        
        #
        l_true = (l_true - l_true[::-1] )[:get_batch_size//2]

        # value used in the lossnet loss
        one = (2 * tf.math.sign(  tf.clip_by_value( l_true, 0, 1))) - 1

        if   self.reduction == 'mean':
            l_loss = tf.reduce_sum(tf.clip_by_value(self.margin - one * l_pred, 0,10000))
            l_loss = tf.math.divide(l_loss , tf.cast(tf.shape(l_pred)[0], l_loss.dtype)) # Note that the size of l_pred is already halved
        elif self.reduction == 'none':
            l_loss = tf.clip_by_value(self.margin - one * l_pred, 0,10000)
        else:
            NotImplementedError()

        return l_loss

class Loss_Lossnet(object):
    """Loss of lossnet"""
    def __init__(self,  margin=1.0, reduction= "mean" ):
        # parameters model
        self.margin       = margin
        self.reduction    = reduction
        
    def compute_loss(self,c_true,c_pred,l_pred_w,l_pred_s):
        
        l_pred_w = tf.squeeze(l_pred_w)
        l_pred_s = tf.squeeze(l_pred_s)
        # compute the classification loss non reducted
        get_batch_size = tf.shape(c_true)[0]

        # Classification loss
        with tf.compat.v1.variable_scope("Classification_loss"):
            c_loss_non_reducted = tf.nn.softmax_cross_entropy_with_logits_v2(labels=c_true, logits=c_pred)
            c_loss = tf.reduce_mean(c_loss_non_reducted)

        with tf.compat.v1.variable_scope("Reference_Loss_LossNet"):
            l_true = (c_loss_non_reducted - c_loss_non_reducted[::-1] )[:get_batch_size//2]
            # get the value without the gradient
            l_true = tf.stop_gradient(l_true)


        # value used in the lossnet loss
        one = (2 * tf.math.sign(  tf.clip_by_value( l_true, 0, 1))) - 1

        with tf.compat.v1.variable_scope("Learning_loss_loss_whole"):

            l_pred_w = (l_pred_w - l_pred_w[::-1])[:get_batch_size//2]
            if self.reduction == 'mean':
                l_loss_w = tf.reduce_sum(tf.clip_by_value(self.margin - one * l_pred_w, 0,10000))
                l_loss_w = tf.math.divide(l_loss_w , tf.cast(tf.shape(l_pred_w)[0], l_loss_w.dtype) ) # Note that the size of l_pred is already halved
            elif self.reduction == 'none':
                l_loss_w = tf.clip_by_value(self.margin - one * l_pred_w, 0,10000)
            else:
                NotImplementedError()

        with tf.compat.v1.variable_scope("Learning_loss_loss_split"):
            l_pred_s = (l_pred_s - l_pred_s[::-1])[:get_batch_size//2]
            if self.reduction == 'mean':
                l_loss_s = tf.reduce_sum(tf.clip_by_value(self.margin - one * l_pred_s, 0,10000))
                l_loss_s = tf.math.divide(l_loss_s , tf.cast(tf.shape(l_pred_s)[0], l_loss_s.dtype) ) # Note that the size of l_pred is already halved
            elif self.reduction == 'none':
                l_loss_s = tf.clip_by_value(self.margin - one * l_pred_s, 0,10000)
            else:
                NotImplementedError()
                
        return c_loss_non_reducted, c_loss, l_loss_w, l_loss_s
                
        


class Classifier_AL(object):
    """Implement tensoflow yolov3 here"""
    def __init__(self, backbone, config, trainable=True, reduction='mean'):
        
        self.backbone = backbone
        
        self.trainable = trainable

        
        # parameters model
        self.num_class      = len(config["CLASSES"])
        self.embedding_size = config["embedding_size"]
        
        # learning loss parameters
        self.margin       = config["MARGIN"]
        self.reduction    = reduction

    def build_nework(self,input_data):
        def get_embedding_nets(embedding_size):
            return Sequential([layers.GlobalAveragePooling2D(),layers.Dense(embedding_size),layers.Activation("relu")])

        # add backbone
        with tf.compat.v1.variable_scope("Backbone"):
            #ResNet18(classes, input_shape, weight_decay=1e-4)
            x = self.backbone(input_data,self.num_class,self.trainable)
            c_pred = x[0]

        with tf.compat.v1.variable_scope("LossNet"):
            
            embeddings_fn_list=[]
            # generate the embeddings layers
            for out in x[1:]:
                embeddings_fn_list.append(get_embedding_nets(self.embedding_size))
            # define dense function
            dense_fn = layers.Dense(1)
            concat_same = layers.Concatenate()
            
            # copy the features from the backbone to divide from stop grad and not stop grad
            new_x = []
            for i, out in enumerate(x[1:]):
                new_x.append(tf.stop_gradient(tf.identity( out, name="Copy_features_"+str(i)),name="stop_grad_"+str(i)))
            
            # 
            embeddings_list_whole =[]
            embeddings_list_split =[]
            for i, out in enumerate(x[1:]):
                embeddings_list_split.append(embeddings_fn_list[i](new_x[i]))
                embeddings_list_whole.append(embeddings_fn_list[i](out))
            
            embedding_whole    =  concat_same(embeddings_list_whole)
            embedding_split    =  concat_same(embeddings_list_split)
            
            l_pred_w = tf.squeeze(dense_fn(embedding_whole))
            l_pred_s = tf.squeeze(dense_fn(embedding_split))
            
            self.emb_w = embedding_whole
            self.emb_s = embedding_split 

            self.c_pred = c_pred
            self.l_pred_w = l_pred_w
            self.l_pred_s = l_pred_s

        return self.c_pred, self.l_pred_w, self.l_pred_s
    
    def compute_loss(self,c_true):

        # compute the classification loss non reducted
        get_batch_size = tf.shape(c_true)[0]

        # Classification loss
        with tf.compat.v1.variable_scope("Classification_loss"):
            c_loss_non_reducted = tf.nn.softmax_cross_entropy_with_logits_v2(labels=c_true, logits=self.c_pred)
            c_loss = tf.reduce_mean(c_loss_non_reducted)

        with tf.compat.v1.variable_scope("Reference_Loss_LossNet"):
            l_true = (c_loss_non_reducted - c_loss_non_reducted[::-1] )[:get_batch_size//2]
            # get the value without the gradient
            l_true = tf.stop_gradient(l_true)


        # value used in the lossnet loss
        one = (2 * tf.math.sign(  tf.clip_by_value( l_true, 0, 1))) - 1

        with tf.compat.v1.variable_scope("Learning_loss_loss_whole"):

            l_pred_w = (self.l_pred_w - self.l_pred_w[::-1])[:get_batch_size//2]
            if self.reduction == 'mean':
                l_loss_w = tf.reduce_sum(tf.clip_by_value(self.margin - one * l_pred_w, 0,10000))
                l_loss_w = tf.math.divide(l_loss_w , tf.cast(tf.shape(l_pred_w)[0], l_loss_w.dtype) ) # Note that the size of l_pred is already halved
            elif self.reduction == 'none':
                l_loss_w = tf.clip_by_value(self.margin - one * self.l_pred_w, 0,10000)
            else:
                NotImplementedError()

        with tf.compat.v1.variable_scope("Learning_loss_loss_split"):
            l_pred_s = (self.l_pred_s - self.l_pred_s[::-1])[:get_batch_size//2]
            if self.reduction == 'mean':
                l_loss_s = tf.reduce_sum(tf.clip_by_value(self.margin - one * l_pred_s, 0,10000))
                l_loss_s = tf.math.divide(l_loss_s , tf.cast(tf.shape(self.l_pred_s)[0], l_loss_s.dtype) ) # Note that the size of l_pred is already halved
            elif self.reduction == 'none':
                l_loss_s = tf.clip_by_value(self.margin - one * self.l_pred_s, 0,10000)
            else:
                NotImplementedError()

        self.c_loss =  c_loss
        self.l_loss_w = l_loss_w
        self.l_loss_s = l_loss_s
        self.l_true  = l_true
        self.c_loss_non_reducted = c_loss_non_reducted

        return self.c_loss, self.l_loss_w, self.l_loss_s
    
  
def Loss_fn(c_loss_nr, l_pred, name= 'Learning_Loss', margin=1.0, reduction= "mean"):
    
    # compute the classification loss non reducted
    get_batch_size = tf.shape(l_pred)[0]

    l_true = (c_loss_nr - c_loss_nr[::-1] )[:get_batch_size//2]
    # get the value without the gradient
    l_true = tf.stop_gradient(l_true)

    # value used in the lossnet loss
    one = (2 * tf.math.sign(  tf.clip_by_value( l_true, 0, 1))) - 1
    
    l_pred = tf.squeeze(l_pred)
    l_pred = (l_pred - l_pred[::-1])[:get_batch_size//2]
    if reduction == 'mean':
        l_loss = tf.reduce_sum(tf.clip_by_value(margin - one * l_pred, 0,10000))
        l_loss = tf.math.divide(l_loss , tf.cast(tf.shape(l_pred)[0], l_loss.dtype) , name = name) # Note that the size of l_pred is already halved
    elif reduction == 'none':
        l_loss = tf.clip_by_value(margin - one * l_pred, 0,10000)
    else:
        NotImplementedError()

    return l_loss

class loss_nr_layer(layers.Layer):
    """Layer that creates an activity sparsity regularization loss."""
    def __init__(self):
        super(loss_nr_layer, self).__init__()
        self.concat_losses_w = layers.Concatenate(axis=-1)
        self.concat_losses_s = layers.Concatenate(axis=-1)
    def call(self, inputs):
        c_true, c_pred, l_pred_w, l_pred_s = inputs
        c_loss_nr = tf.expand_dims(tf.nn.softmax_cross_entropy_with_logits_v2(labels=c_true, logits=c_pred, name='Non_reducted_c_loss'),axis=-1)
        concat_w = self.concat_losses_w([c_loss_nr,l_pred_w])
        concat_s = self.concat_losses_s([c_loss_nr,l_pred_s])
        return c_loss_nr, concat_w, concat_s

def Categorical_Accuracy(c_true,c_pred):
    correct_prediction = tf.equal( tf.argmax(c_true, 1), tf.argmax(c_pred, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def MeanAbsoluteError(c_true,c_pred):
    return tf.reduce_mean(tf.math.abs(tf.math.subtract(l_true, l_pred)))
    
class LossNet_layer_loss(Layer):
    """Layer that creates an activity sparsity regularization loss."""

    def __init__(self, margin=1.0, reduction= "mean"):
        super(LossNet_layer_loss, self).__init__()
        self.margin    = margin
        self.reduction = reduction
        self.stop_grad = layers.Lambda(lambda x: backend.stop_gradient(x))
        
        # for tensorflow 1.15
        #self.MAE_Whole = metrics.MeanAbsoluteError(name='MAE_Whole')
        #self.MAE_Split = metrics.MeanAbsoluteError(name='MAE_Split')
        #self.Class_acc = metrics.CategoricalAccuracy(name='Classification_Accuracy')

    def call(self, inputs):
        c_true, c_pred, l_pred_w, l_pred_s = inputs
        
        # loss for each element
        c_loss_nr = tf.nn.softmax_cross_entropy_with_logits_v2(labels=c_true, logits=c_pred)
        # Classification loss
        self.add_loss(tf.reduce_mean(c_loss_nr,name='Classification_Loss'))
        #self.add_metric(Categorical_Accuracy(c_true,c_pred))
        
        # Loss whole system training
        l_loss_w = Loss_fn(c_loss_nr, l_pred_w, name='Whole_Learning_Loss', margin=1.0, reduction= "mean")
        #self.add_metric(tf.reduce_mean(tf.math.abs(tf.math.subtract(c_loss_nr, l_pred_w))))
        self.add_loss(l_loss_w)
        
        # loss split system training
        l_loss_s = Loss_fn(c_loss_nr, l_pred_s, name='Split_Learning_Loss', margin=1.0, reduction= "mean")
        #self.add_metric(MeanAbsoluteError(c_loss_nr,l_loss_s), name = "MeanAbsoluteError_s")
        self.add_loss(l_loss_s)
        
        
        return inputs
        
        