import pytest
import tensorflow as tf
from tensorflow import keras
from skimage.data import chelsea

import sys

sys.path.append(".")
import keras_efficientnet_v2


def test_model_predict_t_imagenet():
    """ Run a single forward pass with EfficientNetV2T on imagenet """
    model = keras_efficientnet_v2.EfficientNetV2T(pretrained="imagenet")
    imm = tf.image.resize(chelsea(), model.input_shape[1:3])  # Chelsea the cat
    pred = model(tf.expand_dims(imm / 255, 0)).numpy()
    out = keras.applications.imagenet_utils.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"
    assert abs(out[2] - 0.9630692) <= 1e-5


def test_model_predict_b3_imagenet21k_ft1k():
    """ Run a single forward pass with EfficientNetV2B3 on imagenet21k-ft1k """
    model = keras_efficientnet_v2.EfficientNetV2B3(pretrained="imagenet21k-ft1k")
    imm = tf.image.resize(chelsea(), model.input_shape[1:3])  # Chelsea the cat
    pred = model(tf.expand_dims(imm / 255, 0)).numpy()
    out = keras.applications.imagenet_utils.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"
    assert abs(out[2] - 0.63709414) <= 1e-5


def test_model_predict_s_imagenet21k():
    """ Run a single forward pass with EfficientNetV2S on imagenet21k """
    model = keras_efficientnet_v2.EfficientNetV2S(num_classes=21843, pretrained="imagenet21k")
    imm = tf.image.resize(chelsea(), model.input_shape[1:3])  # Chelsea the cat
    pred = model(tf.expand_dims(imm / 255, 0)).numpy()

    assert pred.argmax() == 200
    assert abs(pred.max() - 0.43293855) <= 1e-5