from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def conv2d_bn(x, filters, kernel_size, strides=(1, 1)):
    layer = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same',
                   use_bias=False
                   )(x)
    layer = BatchNormalization()(layer)
    return layer


def conv2d_bn_relu(x, filters, kernel_size, strides=(1, 1)):
    layer = conv2d_bn(x, filters, kernel_size, strides)
    layer = Activation('relu')(layer)
    return layer


def ResidualBlock(x, filters, kernel_size, downsample=True):
    if downsample:
        # residual_x = conv2d_bn_relu(x, filters, kernel_size=1, strides=2)
        residual_x = conv2d_bn(x, filters, kernel_size=1, strides=2)
        stride = 2
    else:
        residual_x = x
        stride = 1
    residual = conv2d_bn_relu(x,
                              filters=filters,
                              kernel_size=kernel_size,
                              strides=stride,
                              )
    residual = conv2d_bn(residual,
                         filters=filters,
                         kernel_size=kernel_size,
                         strides=1,
                         )
    out = layers.add([residual_x, residual])
    out = Activation('relu')(out)
    return out


def ResNet18_cifar(input_tensor, classes=10, include_top = True):

    x = conv2d_bn_relu(input_tensor, filters=64, kernel_size=(3, 3), strides=(1, 1))
    feats = []
    # # conv 2
    x = ResidualBlock(x, filters=64, kernel_size=(3, 3), downsample=False)
    x = ResidualBlock(x, filters=64, kernel_size=(3, 3), downsample=False)
    feats.append(x)
    # # conv 3
    x = ResidualBlock(x, filters=128, kernel_size=(3, 3), downsample=True)
    x = ResidualBlock(x, filters=128, kernel_size=(3, 3), downsample=False)
    feats.append(x)
    # # conv 4
    x = ResidualBlock(x, filters=256, kernel_size=(3, 3), downsample=True)
    x = ResidualBlock(x, filters=256, kernel_size=(3, 3), downsample=False)
    feats.append(x)
    # # conv 5
    x = ResidualBlock(x, filters=512, kernel_size=(3, 3), downsample=True)
    x = ResidualBlock(x, filters=512, kernel_size=(3, 3), downsample=False)
    feats.append(x)
    if include_top:
        x = AveragePooling2D(pool_size=(4, 4), padding='valid')(x)
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='c_pred')(x)
        
    return [x]+feats
