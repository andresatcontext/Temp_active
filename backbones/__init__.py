"""
This code is taken from 
https://github.com/keras-team/keras-applications

"""

def get_submodules_from_kwargs(kwargs=False):
    from tensorflow.keras import backend, layers, models, utils
    
    return backend, layers, models, utils


def correct_pad(backend, inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.

    # Returns
        A tuple.
    """
    img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))

# TODO modify this backbones to return the features
#from . import vgg16
#from . import vgg19
#from . import inception_v3
#from . import inception_resnet_v2
#from . import xception
#from . import mobilenet
#from . import mobilenet_v2
#from . import mobilenet_v3
#from . import densenet
#from . import nasnet

# cifar resnet
from .resnet18_cifar import ResNet18_cifar

# resnet18
from .small_resnet import ResNet18
from .small_resnet import ResNet34
from .small_resnet import SEResNet18
from .small_resnet import SEResNet34

# resnet
from .resnet_common import ResNet50
from .resnet_common import ResNet101
from .resnet_common import ResNet152
# resnet v2
from .resnet_common import ResNet50V2
from .resnet_common import ResNet101V2
from .resnet_common import ResNet152V2

# resnet v2
from .resnet_common import ResNeXt50
from .resnet_common import ResNeXt101

# efficientnet
"""
# it needs tensorflow newer than 15.0
# check the error:
# https://github.com/tensorflow/tensorflow/issues/30946
# dropout not working with varible batch size
from .efficientnet import EfficientNetB0
from .efficientnet import EfficientNetB1
from .efficientnet import EfficientNetB2
from .efficientnet import EfficientNetB3
from .efficientnet import EfficientNetB4
from .efficientnet import EfficientNetB5
from .efficientnet import EfficientNetB6
from .efficientnet import EfficientNetB7
"""