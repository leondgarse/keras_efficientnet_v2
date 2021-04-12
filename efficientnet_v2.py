"""
Creates a EfficientNetV2 Model as defined in:
Mingxing Tan, Quoc V. Le. (2021).
EfficientNetV2: Smaller Models and Faster Training
arXiv preprint arXiv:2104.00298.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    GlobalAveragePooling2D,
    Input,
    PReLU,
    Reshape,
    Multiply,
)
import math

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
CONV_KERNEL_INITIALIZER = keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")
# CONV_KERNEL_INITIALIZER = 'glorot_uniform'


def _make_divisible(v, divisor=4, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv2d_no_bias(inputs, filters, kernel_size, strides=1, padding="VALID"):
    return Conv2D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER
    )(inputs)


def batchnorm_with_activation(inputs, activation="swish"):
    """Performs a batch normalization followed by an activation. """
    bn_axis = 1 if K.image_data_format() == "channels_first" else -1
    nn = BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
    )(inputs)
    if activation:
        nn = Activation(activation=activation)(nn)
        # nn = PReLU(shared_axes=[1, 2], alpha_initializer=tf.initializers.Constant(0.25))(nn)
    return nn


def se_module(inputs, se_ratio=4):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = inputs.shape[channel_axis]
    reduction = _make_divisible(filters // se_ratio, 8)
    se = GlobalAveragePooling2D()(inputs)
    se = Reshape((1, 1, filters))(se)
    se = Conv2D(reduction, kernel_size=1, use_bias=True, kernel_initializer=CONV_KERNEL_INITIALIZER)(se)
    # se = PReLU(shared_axes=[1, 2])(se)
    se = Activation("swish")(se)
    se = Conv2D(filters, kernel_size=1, use_bias=True, kernel_initializer=CONV_KERNEL_INITIALIZER)(se)
    se = Activation("sigmoid")(se)
    return Multiply()([inputs, se])


def MBConv(inputs, output_channel, stride, expand_ratio, shortcut, use_se):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    input_channel = inputs.shape[channel_axis]
    hidden_dim = input_channel * expand_ratio

    if use_se:
        # pw
        nn = conv2d_no_bias(inputs, hidden_dim, (1, 1), strides=(1, 1), padding="valid")
        nn = batchnorm_with_activation(nn)
        # dw
        nn = DepthwiseConv2D(
            (3, 3), padding="same", strides=stride, use_bias=False, depthwise_initializer=CONV_KERNEL_INITIALIZER
        )(nn)
        nn = batchnorm_with_activation(nn)
        # se
        nn = se_module(nn, se_ratio=4 * expand_ratio)
    else:
        # fused
        nn = conv2d_no_bias(inputs, hidden_dim, (3, 3), strides=stride, padding="same")
        nn = batchnorm_with_activation(nn)

    # pw-linear
    nn = conv2d_no_bias(nn, output_channel, (1, 1), strides=(1, 1), padding="valid")
    nn = batchnorm_with_activation(nn, activation=None)

    if shortcut:
        return Add()([inputs, nn])
    else:
        return nn


def EfficientNetV2(input_shape=(224, 224, 3), include_top=True, classes=1000, width_mult=1, depth_mul=1, strides=2, name="EfficientNetV2"):
    inputs = Input(shape=input_shape)
    out_channel = _make_divisible(24 * width_mult, 8)
    nn = conv2d_no_bias(inputs, out_channel, (3, 3), strides=strides, padding="same")
    nn = batchnorm_with_activation(nn)

    expand_ratios = [1, 4, 4, 4, 6, 6]
    output_channels = [24, 48, 64, 128, 160, 272]
    depths = [2, 4, 4, 6, 9, 15]
    strides = [1, 2, 2, 2, 1, 2]
    use_ses = [0, 0, 0, 1, 1, 1]

    pre_out = out_channel
    for expand_ratio, output_channel, depth, stride, se in zip(expand_ratios, output_channels, depths, strides, use_ses):
        out = _make_divisible(output_channel * width_mult, 8)
        depth = int(math.ceil(depth * depth_mul))
        for ii in range(depth):
            stride = stride if ii == 0 else 1
            shortcut = True if out == pre_out and stride == 1 else False
            nn = MBConv(nn, out, stride, expand_ratio, shortcut, se)
            pre_out = out

    out = _make_divisible(1792 * width_mult, 8)
    nn = conv2d_no_bias(nn, out, (1, 1), strides=(1, 1), padding="valid")
    nn = batchnorm_with_activation(nn)

    if include_top:
        nn = GlobalAveragePooling2D(name="avg_pool")(nn)
        nn = Dense(classes, activation="softmax", name="predictions")(nn)
    return Model(inputs=inputs, outputs=nn, name=name)
