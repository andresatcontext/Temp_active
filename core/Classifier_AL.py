import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, losses, metrics
from tensorflow.keras import Sequential


def Lossnet(features, embedding_size):
    """LossNet network"""
    def get_embedding_nets(embedding_size):
        return Sequential([layers.GlobalAveragePooling2D(),layers.Dense(embedding_size),layers.Activation("relu")])

    embeddings_fn_list=[]
    # generate the embeddings layers
    for out in features[1:]:
        embeddings_fn_list.append(get_embedding_nets(embedding_size))
    # define dense function
    dense_fn = layers.Dense(1)
    concat_same = layers.Concatenate()

    # copy the features from the backbone to divide from stop grad and not stop grad
    new_x = []
    for i, out in enumerate(features[1:]):
        new_x.append(tf.stop_gradient(tf.identity( out, name="Copy_features_"+str(i)),name="stop_grad_"+str(i)))

    # 
    embeddings_list_whole =[]
    embeddings_list_split =[]
    for i, out in enumerate(features[1:]):
        embeddings_list_split.append(embeddings_fn_list[i](new_x[i]))
        embeddings_list_whole.append(embeddings_fn_list[i](out))

    embedding_whole    =  concat_same(embeddings_list_whole)
    embedding_split    =  concat_same(embeddings_list_split)

    l_pred_w = tf.squeeze(dense_fn(embedding_whole))
    l_pred_s = tf.squeeze(dense_fn(embedding_split))

    return l_pred_w, l_pred_s, embedding_whole, embedding_split

class Loss_Lossnet(object):
    """Loss of lossnet"""
    def __init__(self,  margin=1.0, reduction= "mean" ):
        # parameters model
        self.margin       = margin
        self.reduction    = reduction
        
    def compute_loss(self,c_true,c_pred,l_pred_w,l_pred_s):
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
    
