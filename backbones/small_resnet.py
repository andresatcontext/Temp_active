import os
import collections

from . import get_submodules_from_kwargs

backend = None
layers = None
models = None
keras_utils = None

ModelParams = collections.namedtuple(
    'ModelParams',
    ['model_name', 'repetitions', 'residual_block', 'attention']
)



def _find_weights(model_name, dataset, include_top):
    w = list(filter(lambda x: x['model'] == model_name, WEIGHTS_COLLECTION))
    w = list(filter(lambda x: x['dataset'] == dataset, w))
    w = list(filter(lambda x: x['include_top'] == include_top, w))
    return w


def load_model_weights(model, model_name, dataset, classes, include_top, **kwargs):
    _, _, _, keras_utils = get_submodules_from_kwargs(kwargs)

    weights = _find_weights(model_name, dataset, include_top)

    if weights:
        weights = weights[0]

        if include_top and weights['classes'] != classes:
            raise ValueError('If using `weights` and `include_top`'
                             ' as true, `classes` should be {}'.format(weights['classes']))
            
        weights_path = keras_utils.get_file(
            weights['name'],
            weights['url'],
            cache_subdir='models',
            md5_hash=weights['md5']
        )

        model.load_weights(weights_path)

    else:
        raise ValueError('There is no weights for such configuration: ' +
                         'model = {}, dataset = {}, '.format(model.name, dataset) +
                         'classes = {}, include_top = {}.'.format(classes, include_top))


WEIGHTS_COLLECTION = [

    # ResNet18
    {
        'model': 'resnet18',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet18_imagenet_1000.h5',
        'name': 'resnet18_imagenet_1000.h5',
        'md5': '64da73012bb70e16c901316c201d9803',
    },

    {
        'model': 'resnet18',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet18_imagenet_1000_no_top.h5',
        'name': 'resnet18_imagenet_1000_no_top.h5',
        'md5': '318e3ac0cd98d51e917526c9f62f0b50',
    },

    # ResNet34
    {
        'model': 'resnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet34_imagenet_1000.h5',
        'name': 'resnet34_imagenet_1000.h5',
        'md5': '2ac8277412f65e5d047f255bcbd10383',
    },

    {
        'model': 'resnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet34_imagenet_1000_no_top.h5',
        'name': 'resnet34_imagenet_1000_no_top.h5',
        'md5': '8caaa0ad39d927cb8ba5385bf945d582',
    },

    {
        'model': 'seresnet18',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnet18_imagenet_1000.h5',
        'name': 'seresnet18_imagenet_1000.h5',
        'md5': '9a925fd96d050dbf7cc4c54aabfcf749',
    },

    {
        'model': 'seresnet18',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnet18_imagenet_1000_no_top.h5',
        'name': 'seresnet18_imagenet_1000_no_top.h5',
        'md5': 'a46e5cd4114ac946ecdc58741e8d92ea',
    },

    {
        'model': 'seresnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnet34_imagenet_1000.h5',
        'name': 'seresnet34_imagenet_1000.h5',
        'md5': '863976b3bd439ff0cc05c91821218a6b',
    },

    {
        'model': 'seresnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnet34_imagenet_1000_no_top.h5',
        'name': 'seresnet34_imagenet_1000_no_top.h5',
        'md5': '3348fd049f1f9ad307c070ff2b6ec4cb',
    },

]


def ChannelSE(reduction=16, **kwargs):
    """
    Squeeze and Excitation block, reimplementation inspired by
        https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py
    Args:
        reduction: channels squeeze factor
    """
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
    channels_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def layer(input_tensor):
        # get number of channels/filters
        channels = backend.int_shape(input_tensor)[channels_axis]

        x = input_tensor

        # squeeze and excitation block in PyTorch style with
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Lambda(expand_dims, arguments={'channels_axis': channels_axis})(x)
        x = layers.Conv2D(channels // reduction, (1, 1), kernel_initializer='he_uniform')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(channels, (1, 1), kernel_initializer='he_uniform')(x)
        x = layers.Activation('sigmoid')(x)

        # apply attention
        x = layers.Multiply()([input_tensor, x])

        return x

    return layer


# -------------------------------------------------------------------------
#   Helpers functions
# -------------------------------------------------------------------------

def handle_block_names(stage, block):
    name_base = 'stage{}_unit{}_'.format(stage + 1, block + 1)
    conv_name = name_base + 'conv'
    bn_name = name_base + 'bn'
    relu_name = name_base + 'relu'
    sc_name = name_base + 'sc'
    return conv_name, bn_name, relu_name, sc_name


def get_conv_params(**params):
    default_conv_params = {
        'kernel_initializer': 'he_uniform',
        'use_bias': False,
        'padding': 'valid',
    }
    default_conv_params.update(params)
    return default_conv_params


def get_bn_params(**params):
    axis = 3 if backend.image_data_format() == 'channels_last' else 1
    default_bn_params = {
        'axis': axis,
        'momentum': 0.99,
        'epsilon': 2e-5,
        'center': True,
        'scale': True,
    }
    default_bn_params.update(params)
    return default_bn_params


# -------------------------------------------------------------------------
#   Residual blocks
# -------------------------------------------------------------------------

def residual_conv_block(filters, stage, block, strides=(1, 1), attention=None, cut='pre'):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        cut: one of 'pre', 'post'. used to decide where skip connection is taken
    # Returns
        Output tensor for the block.
    """

    def layer(input_tensor):

        # get params and names of layers
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = layers.BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
        x = layers.Activation('relu', name=relu_name + '1')(x)

        # defining shortcut connection
        if cut == 'pre':
            shortcut = input_tensor
        elif cut == 'post':
            shortcut = layers.Conv2D(filters, (1, 1), name=sc_name, strides=strides, **conv_params)(x)
        else:
            raise ValueError('Cut type not in ["pre", "post"]')

        # continue with convolution layers
        x = layers.ZeroPadding2D(padding=(1, 1))(x)
        x = layers.Conv2D(filters, (3, 3), strides=strides, name=conv_name + '1', **conv_params)(x)

        x = layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = layers.Activation('relu', name=relu_name + '2')(x)
        x = layers.ZeroPadding2D(padding=(1, 1))(x)
        x = layers.Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)(x)

        # use attention block if defined
        if attention is not None:
            x = attention(x)

        # add residual connection
        x = layers.Add()([x, shortcut])
        return x

    return layer


def residual_bottleneck_block(filters, stage, block, strides=None, attention=None, cut='pre'):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        cut: one of 'pre', 'post'. used to decide where skip connection is taken
    # Returns
        Output tensor for the block.
    """

    def layer(input_tensor):

        # get params and names of layers
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = layers.BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
        x = layers.Activation('relu', name=relu_name + '1')(x)

        # defining shortcut connection
        if cut == 'pre':
            shortcut = input_tensor
        elif cut == 'post':
            shortcut = layers.Conv2D(filters * 4, (1, 1), name=sc_name, strides=strides, **conv_params)(x)
        else:
            raise ValueError('Cut type not in ["pre", "post"]')

        # continue with convolution layers
        x = layers.Conv2D(filters, (1, 1), name=conv_name + '1', **conv_params)(x)

        x = layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = layers.Activation('relu', name=relu_name + '2')(x)
        x = layers.ZeroPadding2D(padding=(1, 1))(x)
        x = layers.Conv2D(filters, (3, 3), strides=strides, name=conv_name + '2', **conv_params)(x)

        x = layers.BatchNormalization(name=bn_name + '3', **bn_params)(x)
        x = layers.Activation('relu', name=relu_name + '3')(x)
        x = layers.Conv2D(filters * 4, (1, 1), name=conv_name + '3', **conv_params)(x)

        # use attention block if defined
        if attention is not None:
            x = attention(x)

        # add residual connection
        x = layers.Add()([x, shortcut])

        return x

    return layer


# -------------------------------------------------------------------------
#   Residual Model Builder
# -------------------------------------------------------------------------


def ResNet(model_params, input_shape=None, input_tensor=None, include_top=True,
           classes=1000, weights='imagenet', **kwargs):
    """Instantiates the ResNet, SEResNet architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Args:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    Returns:
        A Keras model instance.

    Raises:
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape, name='data')
    else:
        img_input = input_tensor

    # choose residual block type
    ResidualBlock = model_params.residual_block
    if model_params.attention:
        Attention = model_params.attention(**kwargs)
    else:
        Attention = None

    # get parameters for model layers
    no_scale_bn_params = get_bn_params(scale=False)
    bn_params = get_bn_params()
    conv_params = get_conv_params()
    init_filters = 64

    # resnet bottom
    x = layers.BatchNormalization(name='bn_data', **no_scale_bn_params)(img_input)
    x = layers.ZeroPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(init_filters, (7, 7), strides=(2, 2), name='conv0', **conv_params)(x)
    x = layers.BatchNormalization(name='bn0', **bn_params)(x)
    x = layers.Activation('relu', name='relu0')(x)
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pooling0')(x)

    features = []
    # resnet body
    for stage, rep in enumerate(model_params.repetitions):
        for block in range(rep):

            filters = init_filters * (2 ** stage)

            # first block of first stage without strides because we have maxpooling before
            if block == 0 and stage == 0:
                x = ResidualBlock(filters, stage, block, strides=(1, 1),
                                  cut='post', attention=Attention)(x)

            elif block == 0:
                x = ResidualBlock(filters, stage, block, strides=(2, 2),
                                  cut='post', attention=Attention)(x)

            else:
                x = ResidualBlock(filters, stage, block, strides=(1, 1),
                                  cut='pre', attention=Attention)(x)
        features.append(x)

    x = layers.BatchNormalization(name='bn1', **bn_params)(x)
    x = layers.Activation('relu', name='relu1')(x)

    # resnet top
    if include_top:
        x = layers.GlobalAveragePooling2D(name='pool1')(x)
        x = layers.Dense(classes, name='fc1')(x)
        x = layers.Activation('softmax', name='softmax')(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, [x]+features)

    if weights:
        if type(weights) == str and os.path.exists(weights):
            model.load_weights(weights)
        else:
            load_model_weights(model, model_params.model_name,
                               weights, classes, include_top, **kwargs)

    return model


# -------------------------------------------------------------------------
#   Residual Models
# -------------------------------------------------------------------------

MODELS_PARAMS = {
    'resnet18': ModelParams('resnet18', (2, 2, 2, 2), residual_conv_block, None),
    'resnet34': ModelParams('resnet34', (3, 4, 6, 3), residual_conv_block, None),
    'seresnet18': ModelParams('seresnet18', (2, 2, 2, 2), residual_conv_block, ChannelSE),
    'seresnet34': ModelParams('seresnet34', (3, 4, 6, 3), residual_conv_block, ChannelSE),
}


def ResNet18(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True, **kwargs):
    return ResNet(
        MODELS_PARAMS['resnet18'],
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights,
        **kwargs
    )


def ResNet34(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True, **kwargs):
    return ResNet(
        MODELS_PARAMS['resnet34'],
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights,
        **kwargs
    )


def SEResNet18(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True, **kwargs):
    return ResNet(
        MODELS_PARAMS['seresnet18'],
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights,
        **kwargs
    )


def SEResNet34(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True, **kwargs):
    return ResNet(
        MODELS_PARAMS['seresnet34'],
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights,
        **kwargs
    )


def preprocess_input(x, **kwargs):
    return x


setattr(ResNet18, '__doc__', ResNet.__doc__)
setattr(ResNet34, '__doc__', ResNet.__doc__)
setattr(SEResNet18, '__doc__', ResNet.__doc__)
setattr(SEResNet34, '__doc__', ResNet.__doc__)