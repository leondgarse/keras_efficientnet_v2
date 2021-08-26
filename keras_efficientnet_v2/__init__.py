from .version import __version__
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
    EfficientNetV1B0,
    EfficientNetV1B1,
    EfficientNetV1B2,
    EfficientNetV1B3,
    EfficientNetV1B4,
    EfficientNetV1B5,
    EfficientNetV1B6,
    EfficientNetV1B7,
    EfficientNetV1L2,
)

__v2_head_doc__ = """
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
  include_preprocessing: Boolean value if add preprocessing `Rescale + Normalization` after `Input`. Default `False`.
      `True` means using input value in range `[0, 255]`.
      `False` means using input value in range `[-1, 1]`.
  pretrained: value in {pretrained}.
      Will try to download and load pre-trained model weights if not None.
      Save path is `~/.keras/models/efficientnetv2/`.

Returns:
    A `keras.Model` instance.
"""

EfficientNetV2.__doc__ = __v2_head_doc__ + """
Args:
  model_type: is the pre-defined model, value in
      v2: ["t", "s", "m", "l", "b0", "b1", "b2", "b3"].
      v1: ["v1-b0", "v1-b1", "v1-b2", "v1-b3", "v1-b4", "v1-b5", "v1-b6", "v1-b7", "v1-l2"]
  model_name: string, model name.
""" + __tail_doc__.format(pretrained=[None, "imagenet", "imagenet21k", "imagenet21k-ft1k", "noisy_student"]) + """
Model architectures:
  | V2 Model          | Params | 1K Top1 acc |
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

  | V1 Model          | Params  | 1K Top1 Acc |
  | ----------------- | ------- | ----------- |
  | EfficientNetV1B0  | 5.3M    | 78.8        |
  | EfficientNetV1B1  | 7.8M    | 81.5        |
  | EfficientNetV1B2  | 9.1M    | 82.4        |
  | EfficientNetV1B3  | 12.2M   | 84.1        |
  | EfficientNetV1B4  | 19.3M   | 85.3        |
  | EfficientNetV1B5  | 30.4M   | 86.1        |
  | EfficientNetV1B6  | 43.0M   | 86.4        |
  | EfficientNetV1B7  | 66.3M   | 86.9        |
  | EfficientNetV1L2  | 480.3M  | 88.4        |
"""

EfficientNetV2B0.__doc__ = __v2_head_doc__ + """
Args:
""" + __tail_doc__.format(pretrained=[None, "imagenet", "imagenet21k", "imagenet21k-ft1k"])

EfficientNetV2B1.__doc__ = EfficientNetV2B0.__doc__
EfficientNetV2B2.__doc__ = EfficientNetV2B0.__doc__
EfficientNetV2B3.__doc__ = EfficientNetV2B0.__doc__
EfficientNetV2T.__doc__ = __v2_head_doc__ + """Architecture and weights from [Github rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models#july-5-9-2021).

Args:
""" + __tail_doc__.format(pretrained=[None, "imagenet"])
EfficientNetV2S.__doc__ = EfficientNetV2B0.__doc__
EfficientNetV2M.__doc__ = EfficientNetV2B0.__doc__
EfficientNetV2L.__doc__ = EfficientNetV2B0.__doc__
EfficientNetV2XL.__doc__ = EfficientNetV2B0.__doc__

__v1_head_doc__ = """
Github source [leondgarse/keras_efficientnet_v2](https://github.com/leondgarse/keras_efficientnet_v2).
Keras implementation of [Github tensorflow/tpu/efficientnet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet).
Paper [PDF 1911.04252 Self-training with Noisy Student improves ImageNet classification](https://arxiv.org/pdf/1911.04252.pdf).
"""

EfficientNetV1B0.__doc__ = __v1_head_doc__ + """
Args:
""" + __tail_doc__.format(pretrained=[None, "imagenet", "noisy_student"])

EfficientNetV1B1.__doc__ = EfficientNetV1B0.__doc__
EfficientNetV1B2.__doc__ = EfficientNetV1B0.__doc__
EfficientNetV1B3.__doc__ = EfficientNetV1B0.__doc__
EfficientNetV1B4.__doc__ = EfficientNetV1B0.__doc__
EfficientNetV1B5.__doc__ = EfficientNetV1B0.__doc__
EfficientNetV1B6.__doc__ = EfficientNetV1B0.__doc__
EfficientNetV1B7.__doc__ = EfficientNetV1B0.__doc__
EfficientNetV1L2.__doc__ = EfficientNetV1B0.__doc__
