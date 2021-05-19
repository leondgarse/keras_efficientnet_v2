## Keras_efficientnet_v2_test
***
  - My own keras implementation of [Official efficientnetv2](https://github.com/google/automl/tree/master/efficientnetv2).
  - Article [arXiv 2104.00298 EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) by Mingxing Tan, Quoc V. Le.
  - `h5` model weights converted from official publication.

    | Model           | ImageNet21K weight | ImageNet21K weight no top |
    | --------------- | ------------------ | ------------------------- |
    | EfficientNetV2S | [efficientnetv2-s-21k.h5](https://drive.google.com/file/d/1onSbAdvSYuvZzDdEg1rAXs7UIIR-cutB/view?usp=sharing) | [efficientnetv2-s-21k-notop.h5](https://drive.google.com/file/d/1bw79TEh4teW_HDtbnmiF42LmOjeeQXNU/view?usp=sharing) |
    | EfficientNetV2M | [efficientnetv2-m-21k.h5](https://drive.google.com/file/d/1lXERhhTczTl5RJDJ8JfC6WlZr103MQxp/view?usp=sharing) | [efficientnetv2-m-21k-notop.h5](https://drive.google.com/file/d/1cxHyIMzHQZLqf1qfv0JFGHSfullOBsZt/view?usp=sharing) |
    | EfficientNetV2L | [efficientnetv2-l-21k.h5](https://drive.google.com/file/d/1apIx_tNxworcMhWFK384RNdLDCvuQ4o3/view?usp=sharing) | [efficientnetv2-l-21k-notop.h5](https://drive.google.com/file/d/1yNulcVfpB-0f1IoTF45RI_nJZzIl7c8A/view?usp=sharing) |
  - **Output compare**
    ```py
    import tensorflow as tf
    import numpy as np

    # Original official efficientnetv2-s model
    import brain_automl.efficientnetv2.infer
    model = brain_automl.efficientnetv2.infer.create_model('efficientnetv2-s', 'imagenet21k')
    len(model(tf.ones([1, 224, 224, 3]), False))
    ckpt = tf.train.latest_checkpoint('models/efficientnetv2-s-21k')
    model.load_weights(ckpt)
    orign_out = model(tf.ones([1, 224, 224, 3]))[0]

    # Converted EfficientNetV2S model. For ImageNet21k, dropout_rate=0.000001, survival_prob=1.0
    from Keras_efficientnet_v2_test import efficientnet_v2
    converted_model = efficientnet_v2.EfficientNetV2S(survivals=None, dropout=1e-6, classes=21843, classifier_activation=None)
    converted_model.load_weights('models/efficientnetv2-s-21k.h5')
    # Or just use: converted_model = tf.keras.models.load_model('models/efficientnetv2-s-21k.h5')
    converted_out = converted_model(tf.ones([1, 224, 224, 3]))

    # Compare result
    print('Allclose:', np.allclose(orign_out.numpy(), converted_out.numpy()))
    # Allclose: True
    ```
  - [Colab efficientnetV2_basic_test.ipynb](https://colab.research.google.com/drive/1QYfgaqEWwaOCsGnPsD9Xu5-8wNbrD6Dj?usp=sharing)
  - EfficientNetV2-S architecture

    | Stage | Operator               | Stride | #Channels | #Layers |
    | ----- | ---------------------- | ------ | --------- | ------- |
    | 0     | Conv3x3                | 2      | 24        | 1       |
    | 1     | Fused-MBConv1, k3x3    | 1      | 24        | 2       |
    | 2     | Fused-MBConv4, k3x3    | 2      | 48        | 4       |
    | 3     | Fused-MBConv4, k3x3    | 2      | 64        | 4       |
    | 4     | MBConv4, k3x3, SE0.25  | 2      | 128       | 6       |
    | 5     | MBConv6, k3x3, SE0.25  | 1      | 160       | 9       |
    | 6     | MBConv6, k3x3, SE0.25  | 2      | 256       | 15      |
    | 7     | Conv1x1 & Pooling & FC | -      | 1280      | 1       |

  - Progressive training settings for EfficientNetV2
    |              | S min | S max | M min | M max | L min | M max |
    | ------------ | ----- | ----- | ----- | ----- | ----- | ----- |
    | Image Size   | 128   | 300   | 128   | 380   | 128   | 380   |
    | RandAugment  | 5     | 15    | 5     | 20    | 5     | 25    |
    | Mixup alpha  | 0     | 0     | 0     | 0.2   | 0     | 0.4   |
    | Dropout rate | 0.1   | 0.3   | 0.1   | 0.4   | 0.1   | 0.5   |

  - Imagenet training detail
    - RMSProp optimizer with decay 0.9 and momentum 0.9
    - batch norm momentum 0.99; weight decay 1e-5
    - Each model is trained for 350 epochs with total batch size 4096
    - Learning rate is first warmed up from 0 to 0.256, and then decayed by 0.97 every 2.4 epochs
    - We use exponential moving average with 0.9999 decay rate
    - RandAugment (Cubuk et al., 2020)
    - Mixup (Zhang et al., 2018)
    - Dropout (Srivastava et al., 2014)
    - and stochastic depth (Huang et al., 2016) with 0.8 survival probability
***

## Related Projects
  - [tfa.layers.StochasticDepth](https://www.tensorflow.org/addons/api_docs/python/tfa/layers/StochasticDepth)
  - [Official efficientnetv2](https://github.com/google/automl/tree/master/efficientnetv2)
***
