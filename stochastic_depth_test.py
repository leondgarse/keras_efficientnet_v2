import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import backend

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def batchnorm_with_activation(inputs, activation="relu", zero_gamma=False, name=""):
    """Performs a batch normalization followed by an activation. """
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1
    gamma_initializer = tf.zeros_initializer() if zero_gamma else tf.ones_initializer()
    nn = layers.BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        gamma_initializer=gamma_initializer,
        name=name + "bn",
    )(inputs)
    if activation:
        nn = layers.Activation(activation=activation, name=name + activation)(nn)
    return nn


def conv2d_no_bias(inputs, filters, kernel_size, strides=1, padding="VALID", name=""):
    return layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False, name=name + "conv")(inputs)


def resnet_block_sd(featuremap, proj_factor=4, activation="relu", strides=1, target_dimension=2048, survival_probability=None, name="resnet_block"):
    """
    Should be like Add(name=name + '_add')([shortcut, residual])
    """
    if strides != 1 or featuremap.shape[-1] != target_dimension:
        padding = "SAME" if strides == 1 else "VALID"
        shortcut = conv2d_no_bias(featuremap, target_dimension, 1, strides=strides, padding=padding, name=name + "_0_")
        shortcut = batchnorm_with_activation(shortcut, activation=None, zero_gamma=False, name=name + "_0_")
    else:
        shortcut = featuremap

    bottleneck_dimension = target_dimension // proj_factor

    nn = conv2d_no_bias(featuremap, bottleneck_dimension, 1, strides=strides, padding="VALID", name=name + "_1_")
    nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name=name + "_1_")
    nn = conv2d_no_bias(nn, bottleneck_dimension, 3, strides=1, padding="SAME", name=name + "_2_")
    nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name=name + "_2_")
    nn = conv2d_no_bias(nn, target_dimension, 1, strides=1, padding="VALID", name=name + "_3_")
    nn = batchnorm_with_activation(nn, activation=None, zero_gamma=True, name=name + "_3_")

    if survival_probability is not None and survival_probability < 1:
        from tensorflow_addons.layers import StochasticDepth

        nn = StochasticDepth(float(survival_probability), name=name + "_stochastic_depth")([shortcut, nn])
    else:
        nn = layers.Add(name=name + "_add")([shortcut, nn])
    return layers.Activation(activation, name=name + "_out")(nn)


def resnet_stack_sd(featuremap, target_dimension=2048, num_layers=3, strides=2, activation="relu", proj_factor=4, survivals=None, name="resnet_block_sd"):
    for ii in range(num_layers):
        block_strides = strides if ii == 0 else 1
        block_name = name + "_block{}".format(ii + 1)
        survival_probability = None if survivals is None else survivals[ii]
        featuremap = resnet_block_sd(featuremap, proj_factor, activation, block_strides, target_dimension, survival_probability, block_name)
    return featuremap


def ResNetSD(
    stack_fn,
    preact,
    use_bias,
    model_name="resnet",
    activation="relu",
    include_top=True,
    weights=None,
    input_shape=None,
    classes=1000,
    classifier_activation="softmax",
):
    img_input = layers.Input(shape=input_shape)

    nn = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name="conv1_pad")(img_input)
    nn = layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name="conv1_conv")(nn)

    if not preact:
        nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name="conv1_")
    nn = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="pool1_pad")(nn)
    nn = layers.MaxPooling2D(3, strides=2, name="pool1_pool")(nn)

    nn = stack_fn(nn)
    if preact:
        nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name="post_")
    if include_top:
        nn = layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        nn = layers.Dense(classes, activation=classifier_activation, name="predictions")(nn)
    return keras.models.Model(img_input, nn, name=model_name)


def ResNet50SD(activation="relu", include_top=True, weights=None, input_shape=(224, 224, 3), classes=1000, survivals=None, **kwargs):
    """
    survivals can be a constant value like `0.5` or `0.8`,
        or a tuple like `(1, 0.5)` indicates the survival probability changes from `1 --> 0.5` for `top --> bottom` layers.
        A higher the value means a higher probability will keep the conv branch.
    """
    num_layers = [3, 4, 6, 3]
    total_layers = sum(num_layers)
    if isinstance(survivals, float):
        survivals = [survivals] * total_layers
    elif isinstance(survivals, (list, tuple)) and len(survivals) == 2:
        start, end = survivals
        survivals = [start + (end - start) * ii / (total_layers - 1) for ii in range(total_layers)]
    else:
        survivals = [None] * total_layers
    survivals = [survivals[int(sum(num_layers[:id])) : sum(num_layers[: id + 1])] for id in range(len(num_layers))]

    def stack_fn(nn, num_layers=num_layers, survivals=survivals):
        nn = resnet_stack_sd(nn, 64 * 4, num_layers[0], strides=1, activation=activation, survivals=survivals[0], name="conv2")
        nn = resnet_stack_sd(nn, 128 * 4, num_layers[1], activation=activation, survivals=survivals[1], name="conv3")
        nn = resnet_stack_sd(nn, 256 * 4, num_layers[2], activation=activation, survivals=survivals[2], name="conv4")
        nn = resnet_stack_sd(nn, 512 * 4, num_layers[3], activation=activation, survivals=survivals[3], name="conv5")
        return nn

    return ResNetSD(stack_fn, False, True, "resnet50", activation, include_top, weights, input_shape, classes, **kwargs)
