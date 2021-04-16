## Keras_efficientnet_v2_test
***
  - Creates an EfficientNetV2 Model using Tensorflow keras, as defined in [arXiv preprint arXiv:2104.00298 EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) by Mingxing Tan, Quoc V. Le.
  - Follows implementation in [Github rosinality/vision-transformers-pytorch](https://github.com/rosinality/vision-transformers-pytorch/blob/main/models/efficientnet.py).
  - This is **NOT** an official implementation, and as the [Official version](https://github.com/google/automl/tree/master/efficientnetv2) still not published, I'm not sure about this architecture.
  - Architecture only, haven't trained on imagenet.
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
    | 6     | MBConv6, k3x3, SE0.25  | 2      | 272       | 15      |
    | 7     | Conv1x1 & Pooling & FC | -      | 1792      | 1       |

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
  - [Github rosinality/vision-transformers-pytorch](https://github.com/rosinality/vision-transformers-pytorch/blob/main/models/efficientnet.py)
  - [Github d-li14/efficientnetv2.pytorch](https://github.com/d-li14/efficientnetv2.pytorch)
  - [Github jahongir7174/EffcientNetV2](https://github.com/jahongir7174/EffcientNetV2)
***
