import pytest
import tensorflow as tf
from tensorflow import keras
from skimage.data import chelsea

import sys

sys.path.append(".")
import keras_efficientnet_v2


def test_model_predict_b0_imagenet():
    model = keras_efficientnet_v2.EfficientNetV2B0(pretrained="imagenet")
    imm = tf.image.resize(chelsea(), model.input_shape[1:3])  # Chelsea the cat
    pred = model(tf.expand_dims(imm / 128 - 1, 0)).numpy()
    out = keras.applications.imagenet_utils.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"
    assert abs(out[2] - 0.76896363) <= 1e-5


def test_model_predict_b1_imagenet_preprocessing():
    model = keras_efficientnet_v2.EfficientNetV2B1(pretrained="imagenet", include_preprocessing=True)
    imm = tf.image.resize(chelsea(), model.input_shape[1:3])  # Chelsea the cat
    pred = model(tf.expand_dims(imm, 0)).numpy()
    out = keras.applications.imagenet_utils.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"
    assert abs(out[2] - 0.76861376) <= 1e-5


def test_model_predict_b2_imagenet21k_ft1k():
    model = keras_efficientnet_v2.EfficientNetV2B2(pretrained="imagenet21k-ft1k")
    imm = tf.image.resize(chelsea(), model.input_shape[1:3])  # Chelsea the cat
    pred = model(tf.expand_dims(imm / 128 - 1, 0)).numpy()
    out = keras.applications.imagenet_utils.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"
    assert abs(out[2] - 0.58329606) <= 1e-5


def test_model_predict_s_imagenet_preprocessing():
    model = keras_efficientnet_v2.EfficientNetV2S(pretrained="imagenet", include_preprocessing=True)
    imm = tf.image.resize(chelsea(), model.input_shape[1:3])  # Chelsea the cat
    pred = model(tf.expand_dims(imm, 0)).numpy()
    out = keras.applications.imagenet_utils.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"
    assert abs(out[2] - 0.8642885) <= 1e-5


def test_model_predict_t_imagenet():
    """ Run a single forward pass with EfficientNetV2T on imagenet """
    model = keras_efficientnet_v2.EfficientNetV2T(pretrained="imagenet")
    imm = tf.image.resize(chelsea(), model.input_shape[1:3])  # Chelsea the cat
    pred = model(tf.expand_dims(imm / 128 - 1, 0)).numpy()
    out = keras.applications.imagenet_utils.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"
    assert abs(out[2] - 0.8502904) <= 1e-5


def test_model_predict_s_imagenet21k():
    """ Run a single forward pass with EfficientNetV2S on imagenet21k """
    model = keras_efficientnet_v2.EfficientNetV2S(num_classes=21843, pretrained="imagenet21k")
    imm = tf.image.resize(chelsea(), model.input_shape[1:3])  # Chelsea the cat
    pred = model(tf.expand_dims(imm / 128 - 1, 0)).numpy()

    assert pred.argmax() == 2389
    assert abs(pred.max() - 0.15546332) <= 1e-5


def test_model_m_defination():
    model = keras_efficientnet_v2.EfficientNetV2M(num_classes=0, pretrained=None)
    assert model.output_shape == (None, 15, 15, 1280)


def test_model_l_defination():
    model = keras_efficientnet_v2.EfficientNetV2L(num_classes=0, pretrained=None)
    assert model.output_shape == (None, 15, 15, 1280)


def test_model_xl_defination():
    model = keras_efficientnet_v2.EfficientNetV2XL(num_classes=0, pretrained=None)
    assert model.output_shape == (None, 16, 16, 1280)


def test_model_predict_v1_b0_imagenet():
    """ Run a single forward pass with EfficientNetV1B2 on imagenet """
    model = keras_efficientnet_v2.EfficientNetV1B0(pretrained="imagenet")
    imm = tf.image.resize(chelsea(), model.input_shape[1:3])  # Chelsea the cat
    pred = model(tf.expand_dims(imm / 128 - 1, 0)).numpy()
    out = keras.applications.imagenet_utils.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"
    assert abs(out[2] - 0.64605427) <= 1e-5


def test_model_predict_v1_b1_noisy_student():
    """ Run a single forward pass with EfficientNetV1B2 on imagenet """
    model = keras_efficientnet_v2.EfficientNetV1B1(pretrained="noisy_student")
    imm = tf.image.resize(chelsea(), model.input_shape[1:3])  # Chelsea the cat
    pred = model(tf.expand_dims(imm / 128 - 1, 0)).numpy()
    out = keras.applications.imagenet_utils.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"
    assert abs(out[2] - 0.8223327) <= 1e-5


def test_model_predict_v1_b2_imagenet():
    """ Run a single forward pass with EfficientNetV1B2 on imagenet """
    model = keras_efficientnet_v2.EfficientNetV1B2(pretrained="imagenet")
    imm = tf.image.resize(chelsea(), model.input_shape[1:3])  # Chelsea the cat
    pred = model(tf.expand_dims(imm / 128 - 1, 0)).numpy()
    out = keras.applications.imagenet_utils.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"
    assert abs(out[2] - 0.5294576) <= 1e-5


def test_model_predict_v1_b3_noisy_student_preprocessing():
    """ Run a single forward pass with EfficientNetV1B6 on noisy_student """
    model = keras_efficientnet_v2.EfficientNetV1B3(pretrained="noisy_student", include_preprocessing=True)
    imm = tf.image.resize(chelsea(), model.input_shape[1:3])  # Chelsea the cat
    pred = model(tf.expand_dims(imm, 0)).numpy()
    out = keras.applications.imagenet_utils.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"
    assert abs(out[2] - 0.8770545) <= 1e-5


def test_model_predict_v1_b4_noisy_student():
    """ Run a single forward pass with EfficientNetV1B6 on noisy_student """
    model = keras_efficientnet_v2.EfficientNetV1B4(pretrained="noisy_student")
    imm = tf.image.resize(chelsea(), model.input_shape[1:3])  # Chelsea the cat
    pred = model(tf.expand_dims(imm / 128 - 1, 0)).numpy()
    out = keras.applications.imagenet_utils.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"
    assert abs(out[2] - 0.67979187) <= 1e-5


def test_model_v1_b5_defination():
    model = keras_efficientnet_v2.EfficientNetV1B5(num_classes=0, pretrained=None)
    assert model.output_shape == (None, 15, 15, 2048)


def test_model_v1_b6_defination():
    model = keras_efficientnet_v2.EfficientNetV1B6(num_classes=0, pretrained=None)
    assert model.output_shape == (None, 17, 17, 2304)


def test_model_v1_b7_defination():
    model = keras_efficientnet_v2.EfficientNetV1B7(num_classes=0, pretrained=None)
    assert model.output_shape == (None, 19, 19, 2560)


def test_model_v1_l2_defination():
    model = keras_efficientnet_v2.EfficientNetV1L2(num_classes=0, pretrained=None)
    assert model.output_shape == (None, 25, 25, 5504)
