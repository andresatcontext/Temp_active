import tensorflow as tf
import tensorflow.contrib as tf_contrib

# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

weight_init = tf_contrib.layers.variance_scaling_initializer()
weight_regularizer = tf_contrib.layers.l2_regularizer(0.0001)


##################################################################################
# Layer
##################################################################################

def conv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, scope='conv_0'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias, padding=padding)

        return x

def fully_conneted(x, units, use_bias=True, scope='fully_0'):
    with tf.variable_scope(scope):
        x = flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

        return x

def resblock(x_init, channels, is_training=True, use_bias=True, downsample=False, scope='resblock') :
    with tf.variable_scope(scope) :

        x = batch_norm(x_init, is_training, scope='batch_norm_0')
        x = relu(x)


        if downsample :
            x = conv(x, channels, kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
            x_init = conv(x_init, channels, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')

        else :
            x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_0')

        x = batch_norm(x, is_training, scope='batch_norm_1')
        x = relu(x)
        x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_1')

        return x + x_init


##################################################################################
# Sampling
##################################################################################
def flatten(x) :
    return tf.layers.flatten(x)

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    return gap

def avg_pooling(x) :
    return tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='SAME')

##################################################################################
# Activation function
##################################################################################
def relu(x):
    return tf.nn.relu(x)

##################################################################################
# Normalization function
##################################################################################
def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)


def resnet18( img_input, classes, is_training=True):
    ch = 64

    x = conv(img_input, channels=ch, kernel=3, stride=1, scope='conv')
    x = batch_norm(x, is_training, scope='batch_norm')
    x = relu(x)

    x = resblock(x, channels=ch, is_training=is_training, downsample=False, scope='resblock0_0')
    x = resblock(x, channels=ch, is_training=is_training, downsample=False, scope='resblock0_1')        
    feat_0 = x

    ########################################################################################################

    x = resblock(x, channels=ch*2, is_training=is_training, downsample=True, scope='resblock1_0')
    x = resblock(x, channels=ch*2, is_training=is_training, downsample=False, scope='resblock1_1')

    feat_1 = x

    ########################################################################################################

    x = resblock(x, channels=ch*4, is_training=is_training, downsample=True, scope='resblock2_0')
    x = resblock(x, channels=ch*4, is_training=is_training, downsample=False, scope='resblock2_1')

    feat_2 = x

    ########################################################################################################

    x = resblock(x, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0')
    x = resblock(x, channels=ch*8, is_training=is_training, downsample=False, scope='resblock_3_1')

    ########################################################################################################

    feat_3 = x

    x = global_avg_pooling(x)
    x = fully_conneted(x, units=classes, scope='logit')

    return [x,feat_0,feat_1,feat_2,feat_3]
