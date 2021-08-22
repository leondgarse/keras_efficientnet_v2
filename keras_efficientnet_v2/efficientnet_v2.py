"""
Creates a EfficientNetV2 Model as defined in:
Mingxing Tan, Quoc V. Le. (2021).
EfficientNetV2: Smaller Models and Faster Training
arXiv preprint arXiv:2104.00298.
"""
import os
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
    Dropout,
    GlobalAveragePooling2D,
    Input,
    PReLU,
    Reshape,
    Multiply,
)

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 0.001
CONV_KERNEL_INITIALIZER = keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")
# CONV_KERNEL_INITIALIZER = 'glorot_uniform'

BLOCK_CONFIGS = {
    "b0": {  # width 1.0, depth 1.0
        "first_conv_filter": 32,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [16, 32, 48, 96, 112, 192],
        "depthes": [1, 2, 2, 3, 5, 8],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
    },
    "b1": {  # width 1.0, depth 1.1
        "first_conv_filter": 32,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [16, 32, 48, 96, 112, 192],
        "depthes": [2, 3, 3, 4, 6, 9],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
    },
    "b2": {  # width 1.1, depth 1.2
        "first_conv_filter": 32,
        "output_conv_filter": 1408,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [16, 32, 56, 104, 120, 208],
        "depthes": [2, 3, 3, 4, 6, 10],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
    },
    "b3": {  # width 1.2, depth 1.4
        "first_conv_filter": 40,
        "output_conv_filter": 1536,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [16, 40, 56, 112, 136, 232],
        "depthes": [2, 3, 3, 5, 7, 12],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
    },
    "t": {  # width 1.4 * 0.8, depth 1.8 * 0.9, from timm
        "first_conv_filter": 24,
        "output_conv_filter": 1024,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [24, 40, 48, 104, 128, 208],
        "depthes": [2, 4, 4, 6, 9, 14],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
    },
    "s": {  # width 1.4, depth 1.8
        "first_conv_filter": 24,
        "output_conv_filter": 1280,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [24, 48, 64, 128, 160, 256],
        "depthes": [2, 4, 4, 6, 9, 15],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
    },
    "m": {  # width 1.6, depth 2.2
        "first_conv_filter": 24,
        "output_conv_filter": 1280,
        "expands": [1, 4, 4, 4, 6, 6, 6],
        "out_channels": [24, 48, 80, 160, 176, 304, 512],
        "depthes": [3, 5, 5, 7, 14, 18, 5],
        "strides": [1, 2, 2, 2, 1, 2, 1],
        "use_ses": [0, 0, 0, 1, 1, 1, 1],
    },
    "l": {  # width 2.0, depth 3.1
        "first_conv_filter": 32,
        "output_conv_filter": 1280,
        "expands": [1, 4, 4, 4, 6, 6, 6],
        "out_channels": [32, 64, 96, 192, 224, 384, 640],
        "depthes": [4, 7, 7, 10, 19, 25, 7],
        "strides": [1, 2, 2, 2, 1, 2, 1],
        "use_ses": [0, 0, 0, 1, 1, 1, 1],
    },
    "xl": {
        "first_conv_filter": 32,
        "output_conv_filter": 1280,
        "expands": [1, 4, 4, 4, 6, 6, 6],
        "out_channels": [32, 64, 96, 192, 256, 512, 640],
        "depthes": [4, 8, 8, 16, 24, 32, 8],
        "strides": [1, 2, 2, 2, 1, 2, 1],
        "use_ses": [0, 0, 0, 1, 1, 1, 1],
    },
}


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


def conv2d_no_bias(inputs, filters, kernel_size, strides=1, padding="VALID", name=""):
    return Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + "conv")(
        inputs
    )


def batchnorm_with_activation(inputs, activation="swish", name=""):
    """Performs a batch normalization followed by an activation. """
    bn_axis = 1 if K.image_data_format() == "channels_first" else -1
    nn = BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name=name + "bn",
    )(inputs)
    if activation:
        nn = Activation(activation=activation, name=name + activation)(nn)
        # nn = PReLU(shared_axes=[1, 2], alpha_initializer=tf.initializers.Constant(0.25), name=name + "PReLU")(nn)
    return nn


def se_module(inputs, se_ratio=4, name=""):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    h_axis, w_axis = [2, 3] if K.image_data_format() == "channels_first" else [1, 2]

    filters = inputs.shape[channel_axis]
    # reduction = _make_divisible(filters // se_ratio, 8)
    reduction = filters // se_ratio
    # se = GlobalAveragePooling2D()(inputs)
    # se = Reshape((1, 1, filters))(se)
    se = tf.reduce_mean(inputs, [h_axis, w_axis], keepdims=True)
    se = Conv2D(reduction, kernel_size=1, use_bias=True, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + "1_conv")(se)
    # se = PReLU(shared_axes=[1, 2])(se)
    se = Activation("swish")(se)
    se = Conv2D(filters, kernel_size=1, use_bias=True, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + "2_conv")(se)
    se = Activation("sigmoid")(se)
    return Multiply()([inputs, se])


def MBConv(inputs, output_channel, stride, expand_ratio, shortcut, drop_rate=0, use_se=0, is_fused=False, name=""):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    input_channel = inputs.shape[channel_axis]

    if is_fused and expand_ratio != 1:
        nn = conv2d_no_bias(inputs, input_channel * expand_ratio, (3, 3), strides=stride, padding="same", name=name + "sortcut_")
        nn = batchnorm_with_activation(nn, name=name + "sortcut_")
    elif expand_ratio != 1:
        nn = conv2d_no_bias(inputs, input_channel * expand_ratio, (1, 1), strides=(1, 1), padding="same", name=name + "sortcut_")
        nn = batchnorm_with_activation(nn, name=name + "sortcut_")
    else:
        nn = inputs

    if not is_fused:
        # nn = keras.layers.ZeroPadding2D(padding=1, name=name + "pad")(nn)
        nn = DepthwiseConv2D((3, 3), padding="same", strides=stride, use_bias=False, depthwise_initializer=CONV_KERNEL_INITIALIZER, name=name + "MB_dw_")(nn)
        nn = batchnorm_with_activation(nn, name=name + "MB_dw_")

    if use_se:
        nn = se_module(nn, se_ratio=4 * expand_ratio, name=name + "se_")

    # pw-linear
    if is_fused and expand_ratio == 1:
        nn = conv2d_no_bias(nn, output_channel, (3, 3), strides=stride, padding="same", name=name + "fu_")
        nn = batchnorm_with_activation(nn, name=name + "fu_")
    else:
        nn = conv2d_no_bias(nn, output_channel, (1, 1), strides=(1, 1), padding="same", name=name + "MB_pw_")
        nn = batchnorm_with_activation(nn, activation=None, name=name + "MB_pw_")

    if shortcut:
        if drop_rate > 0:
            nn = Dropout(drop_rate, noise_shape=(None, 1, 1, 1), name=name + "drop")(nn)
        return Add()([inputs, nn])
    else:
        return nn


def EfficientNetV2(
    model_type,
    input_shape=(None, None, 3),
    num_classes=1000,
    dropout=0.2,
    first_strides=2,
    drop_connect_rate=0,
    classifier_activation="softmax",
    pretrained="imagenet",
    model_name="EfficientNetV2",
    kwargs=None,    # Not used, just recieving parameter
):
    blocks_config = BLOCK_CONFIGS.get(model_type.lower(), BLOCK_CONFIGS["s"])
    expands = blocks_config["expands"]
    out_channels = blocks_config["out_channels"]
    depthes = blocks_config["depthes"]
    strides = blocks_config["strides"]
    use_ses = blocks_config["use_ses"]
    first_conv_filter = blocks_config.get("first_conv_filter", out_channels[0])
    output_conv_filter = blocks_config.get("output_conv_filter", 1280)

    inputs = Input(shape=input_shape)
    out_channel = _make_divisible(first_conv_filter, 8)
    nn = conv2d_no_bias(inputs, out_channel, (3, 3), strides=first_strides, padding="same", name="stem_")
    nn = batchnorm_with_activation(nn, name="stem_")

    pre_out = out_channel
    global_block_id = 0
    total_blocks = sum(depthes)
    for id, (expand, out_channel, depth, stride, se) in enumerate(zip(expands, out_channels, depthes, strides, use_ses)):
        out = _make_divisible(out_channel, 8)
        is_fused = True if se == 0 else False
        for block_id in range(depth):
            stride = stride if block_id == 0 else 1
            shortcut = True if out == pre_out and stride == 1 else False
            name = "stack_{}_block{}_".format(id, block_id)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            nn = MBConv(nn, out, stride, expand, shortcut, block_drop_rate, se, is_fused, name=name)
            pre_out = out
            global_block_id += 1

    output_conv_filter = _make_divisible(output_conv_filter, 8)
    nn = conv2d_no_bias(nn, output_conv_filter, (1, 1), strides=(1, 1), padding="valid", name="post_")
    nn = batchnorm_with_activation(nn, name="post_")

    if num_classes > 0:
        nn = GlobalAveragePooling2D(name="avg_pool")(nn)
        if dropout > 0 and dropout < 1:
            nn = Dropout(dropout)(nn)
        nn = Dense(num_classes, activation=classifier_activation, dtype="float32", name="predictions")(nn)

    model = Model(inputs=inputs, outputs=nn, name=model_name)
    reload_model_weights(model, model_type, pretrained)
    return model


def reload_model_weights(model, model_type, pretrained="imagenet"):
    pretrained_dd = {"imagenet": "imagenet", "imagenet21k": "21k", "imagenet21k-ft1k": "21k-ft1k"}
    if not pretrained in pretrained_dd:
        print(">>>> No pretraind available, model will be randomly initialized")
        return

    pre_url = "https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-{}-{}.h5"
    url = pre_url.format(model_type, pretrained_dd[pretrained])
    file_name = os.path.basename(url)
    try:
        pretrained_model = keras.utils.get_file(file_name, url, cache_subdir="models/efficientnetv2")
    except:
        print("[Error] will not load weights, url not found or download failed:", url)
        return
    else:
        print(">>>> Load pretraind from:", pretrained_model)
        model.load_weights(pretrained_model, by_name=True, skip_mismatch=True)


def EfficientNetV2B0(input_shape=(224, 224, 3), num_classes=1000, dropout=0.2, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return EfficientNetV2(model_type="b0", model_name="EfficientNetV2B0", **locals(), **kwargs)


def EfficientNetV2B1(input_shape=(240, 240, 3), num_classes=1000, dropout=0.2, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return EfficientNetV2(model_type="b1", model_name="EfficientNetV2B1", **locals(), **kwargs)


def EfficientNetV2B2(input_shape=(260, 260, 3), num_classes=1000, dropout=0.3, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return EfficientNetV2(model_type="b2", model_name="EfficientNetV2B2", **locals(), **kwargs)


def EfficientNetV2B3(input_shape=(300, 300, 3), num_classes=1000, dropout=0.3, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return EfficientNetV2(model_type="b3", model_name="EfficientNetV2B3", **locals(), **kwargs)


def EfficientNetV2T(input_shape=(320, 320, 3), num_classes=1000, dropout=0.2, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return EfficientNetV2(model_type="t", model_name="EfficientNetV2T", **locals(), **kwargs)


def EfficientNetV2S(input_shape=(384, 384, 3), num_classes=1000, dropout=0.2, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return EfficientNetV2(model_type="s", model_name="EfficientNetV2S", **locals(), **kwargs)


def EfficientNetV2M(input_shape=(480, 480, 3), num_classes=1000, dropout=0.3, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return EfficientNetV2(model_type="m", model_name="EfficientNetV2M", **locals(), **kwargs)


def EfficientNetV2L(input_shape=(480, 480, 3), num_classes=1000, dropout=0.4, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return EfficientNetV2(model_type="l", model_name="EfficientNetV2L", **locals(), **kwargs)


def EfficientNetV2XL(input_shape=(512, 512, 3), num_classes=1000, dropout=0.4, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return EfficientNetV2(model_type="xl", model_name="EfficientNetV2XL", **locals(), **kwargs)


def get_actual_drop_connect_rates(model):
    return [ii.rate for ii in model.layers if isinstance(ii, keras.layers.Dropout)]
