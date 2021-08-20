from keras_efficientnet_v2.efficientnet_v2 import (
    EfficientNetV2,
    EfficientNetV2B0,
    EfficientNetV2B1,
    EfficientNetV2B2,
    EfficientNetV2B3,
    EfficientNetV2T,
    EfficientNetV2S,
    EfficientNetV2M,
    EfficientNetV2L,
    EfficientNetV2XL,
)

__head_doc__ = """
Github source [leondgarse/keras_efficientnet_v2](https://github.com/leondgarse/keras_efficientnet_v2).
Keras implementation of [Official efficientnetv2](https://github.com/google/automl/tree/master/efficientnetv2).
Paper [arXiv 2104.00298 EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) by Mingxing Tan, Quoc V. Le.
"""

__tail_doc__ = """  input_shape: it should have exactly 3 inputs channels. Set `(None, None, 3)` for dynamic.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  dropout: dropout rate if top layers is included.
  first_strides: is used in the first Conv2D layer.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      If not `0`, will add a `Dropout` layer for each deep branch changes from `0 --> drop_connect_rate` for `top --> bottom` layers.
      or `0` to disable (default).
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
      Default is `softmax`.
  pretrained: value in {pretrained}.
      Will try to download and load pre-trained model weights if not None.
      Save path is `~/.keras/models/efficientnetv2/`.

Returns:
    A `keras.Model` instance.
"""

EfficientNetV2.__doc__ = __head_doc__ + """
Args:
  model_type: is the pre-defined model, value in ["t", "s", "m", "l", "b0", "b1", "b2", "b3"].
  model_name: string, model name.
""" + __tail_doc__.format(pretrained=[None, "imagenet", "imagenet21k", "imagenet21k-ft1k"]) + """
Model architectures:
  | Model             | Params | 1K Top1 acc |
  | ----------------- | ------ | ----------- |
  | EfficientNetV2B0  | 7.1M   | 78.7%       |
  | EfficientNetV2B1  | 8.1M   | 79.8%       |
  | EfficientNetV2B2  | 10.1M  | 80.5%       |
  | EfficientNetV2B3  | 14.4M  | 82.1%       |
  | EfficientNetV2T   | 13.6M  | 82.5%       |
  | EfficientNetV2S   | 21.5M  | 84.9%       |
  | EfficientNetV2M   | 54.1M  | 86.2%       |
  | EfficientNetV2L   | 119.5M | 86.9%       |
  | EfficientNetV2XL  | 206.8M | 87.2%       |
"""

EfficientNetV2B0.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__.format(pretrained=[None, "imagenet", "imagenet21k", "imagenet21k-ft1k"])

EfficientNetV2B1.__doc__ = EfficientNetV2B0.__doc__
EfficientNetV2B2.__doc__ = EfficientNetV2B0.__doc__
EfficientNetV2B3.__doc__ = EfficientNetV2B0.__doc__
EfficientNetV2T.__doc__ = __head_doc__ + """Architecture and weights from [Github rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models#july-5-9-2021).

Args:
""" + __tail_doc__.format(pretrained=[None, "imagenet"])
EfficientNetV2S.__doc__ = EfficientNetV2B0.__doc__
EfficientNetV2M.__doc__ = EfficientNetV2B0.__doc__
EfficientNetV2L.__doc__ = EfficientNetV2B0.__doc__
EfficientNetV2XL.__doc__ = EfficientNetV2B0.__doc__
