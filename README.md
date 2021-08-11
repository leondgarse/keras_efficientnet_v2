# ___Keras EfficientNetV2___
***
## Table of Contents
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [___Keras EfficientNetV2___](#keras-efficientnetv2)
	- [Table of Contents](#table-of-contents)
	- [Summary](#summary)
	- [Usage](#usage)
	- [Training detail from article](#training-detail-from-article)
	- [Detailed conversion procedure](#detailed-conversion-procedure)
	- [Progressive train test on cifar10](#progressive-train-test-on-cifar10)
	- [Related Projects](#related-projects)

<!-- /TOC -->
***

## Summary
  - My own keras implementation of [Official efficientnetv2](https://github.com/google/automl/tree/master/efficientnetv2). Article [arXiv 2104.00298 EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) by Mingxing Tan, Quoc V. Le.
  - `h5` model weights converted from official publication.

  | Model       | Params |1K Top1 acc | ImageNet21K weight | Imagenet21k-ft1k weight |
  | ----------- | ----- | -------- | ------------------ | ----------------------- |
  | EffNetV2-B0 | 7.1M  | 78.7% | [efficientnetv2-b0-21k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/v1.0.0/efficientnetv2-b0-21k.h5)|[efficientnetv2-b0-21k-ft1k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/v1.0.0/efficientnetv2-b0-21k-ft1k.h5)|
  | EffNetV2-B1 | 8.1M  | 79.8% | [efficientnetv2-b1-21k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/v1.0.0/efficientnetv2-b1-21k.h5)|[efficientnetv2-b1-21k-ft1k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/v1.0.0/efficientnetv2-b1-21k-ft1k.h5)|
  | EffNetV2-B2 | 10.1M | 80.5% | [efficientnetv2-b2-21k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/v1.0.0/efficientnetv2-b2-21k.h5)|[efficientnetv2-b2-21k-ft1k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/v1.0.0/efficientnetv2-b2-21k-ft1k.h5)|
  | EffNetV2-B3 | 14.4M | 82.1% | [efficientnetv2-b3-21k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/v1.0.0/efficientnetv2-b3-21k.h5)|[efficientnetv2-b3-21k-ft1k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/v1.0.0/efficientnetv2-b3-21k-ft1k.h5)|
  | EffNetV2S   | 21.5M | 84.9% | [efficientnetv2-s-21k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/v1.0.0/efficientnetv2-s-21k.h5) |[efficientnetv2-s-21k-ft1k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/v1.0.0/efficientnetv2-s-21k-ft1k.h5)|
  | EffNetV2M   | 54.1M | 86.2% | [efficientnetv2-m-21k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/v1.0.0/efficientnetv2-m-21k.h5) |[efficientnetv2-m-21k-ft1k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/v1.0.0/efficientnetv2-m-21k-ft1k.h5)|
  | EffNetV2L   | 119.5M| 86.9% | [efficientnetv2-l-21k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/v1.0.0/efficientnetv2-l-21k.h5) |[efficientnetv2-l-21k-ft1k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/v1.0.0/efficientnetv2-l-21k-ft1k.h5)|
  | EffNetV2XL  | 206.8M| 87.2% | [efficientnetv2-xl-21k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/v1.0.0/efficientnetv2-xl-21k.h5)|[efficientnetv2-xl-21k-ft1k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/v1.0.0/efficientnetv2-xl-21k-ft1k.h5)|
## Usage
  - This repo can be installed as a pip package, or just `git clone` it.
    ```py
    pip install -U git+https://github.com/leondgarse/keras_efficientnet_v2
    ```
  - **Define model and load pretrained weights** Parameter `pretrained` is added in value `[None, "imagenet", "imagenet21k", "imagenet21k-ft1k"]`, default is `imagenet21k-ft1k`.
    ```py
    # Will download and load `imagenet` pretrained weights.
    # Model weight is loaded with `by_name=True, skip_mismatch=True`.
    import keras_efficientnet_v2
    model = keras_efficientnet_v2.EfficientNetV2S(survivals=None, pretrained="imagenet")

    # Run prediction
    from skimage.data import chelsea
    imm = tf.image.resize(chelsea(), model.input_shape[1:3]) # Chelsea the cat
    pred = model(tf.expand_dims(imm / 255, 0)).numpy()
    print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
    # [('n02124075', 'Egyptian_cat', 0.89163685), ('n02123045', 'tabby', 0.01682318), ...]
    ```
    Or download `h5` model and load directly
    ```py
    mm = keras.models.load_model('efficientnetv2-b3-21k-ft1k.h5')
    ```
    For `"imagenet21k"` pre-trained model, actual `num_classes` is `21843`.
  - **Exclude model top layers** by set `num_classes=0`.
    ```py
    import keras_efficientnet_v2
    model = keras_efficientnet_v2.EfficientNetV2B0(survivals=None, dropout=1e-6, num_classes=0, pretrained="imagenet21k")
    print(model.output_shape)
    # (None, 7, 7, 1280)

    model.save('efficientnetv2-b0-21k-notop.h5')
    ```
  - **Use dynamic input resolution** by set `input_shape=(None, None, 3)`.
    ```py
    import keras_efficientnet_v2
    model = keras_efficientnet_v2.EfficientNetV2M(input_shape=(None, None, 3), survivals=(1, 0.8), num_classes=0, pretrained="imagenet21k-ft1k")

    print(model(np.ones([1, 224, 224, 3])).shape)
    # (1, 7, 7, 1280)
    print(model(np.ones([1, 512, 512, 3])).shape)
    # (1, 16, 16, 1280)
    ```
## Training detail from article
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
## Detailed conversion procedure
  - [convert_effnetv2_model.py](convert_effnetv2_model.py) is a modified version of [the orignal effnetv2_model.py](https://github.com/google/automl/blob/master/efficientnetv2/effnetv2_model.py). Check detail by `vimdiff convert_effnetv2_model.py ../automl/efficientnetv2/effnetv2_model.py`
    - Delete some `names`, as they may cause confliction in keras.
    - Use `.call` directly calling `se` modules and other blocks, so they will not be `blocks` in `model.summary()`
    - Just use `Add` layer instead of `utils.drop_connect`, as when `is_training=False`, `utils.drop_connect` functions like `Add`.
    - Add a `num_classes` parameter outside of `mconfig`.
    - Add `__main__` part, which makes this can be run as a script. Refer to it for converting detail.
  - Depends on official repo
    ```sh
    ../
    ├── automl  # Official repo
    ├── keras_efficientnet_v2  # This one
    ```
  - **Procedure**
    ```py
    # See help info
    CUDA_VISIBLE_DEVICES='-1' python convert_effnetv2_model.py -h

    # Convert by specific model_type and dataset type
    CUDA_VISIBLE_DEVICES='-1' python convert_effnetv2_model.py -m b0 -d imagenet21k

    # Convert by specific model_type and all its datasets ['imagenet', 'imagenet21k', 'imagenetft']
    CUDA_VISIBLE_DEVICES='-1' python convert_effnetv2_model.py -m s -d all

    # Convert all model_types and and all datasets
    CUDA_VISIBLE_DEVICES='-1' python convert_effnetv2_model.py -m all -d all
    ```
## Progressive train test on cifar10
  - [Colab efficientnetV2_basic_test.ipynb](https://colab.research.google.com/drive/1vmAEfF9tUgK2gkrS5qVftadTyUcX343D?usp=sharing)
  ```py
  import keras_efficientnet_v2
  from tensorflow import keras
  from keras_efficientnet_v2 import progressive_train_test

  num_classes = 10
  ev2_s = keras_efficientnet_v2.EfficientNetV2("s", input_shape=(None, None, 3), num_classes=0)
  out = ev2_s.output

  nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(out)
  nn = keras.layers.Dropout(0.1)(nn)
  nn = keras.layers.Dense(num_classes, activation="softmax", name="predictions", dtype="float32")(nn)
  model = keras.models.Model(ev2_s.inputs[0], nn)

  lr_scheduler = None
  optimizer = "adam"
  loss = "categorical_crossentropy"
  model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

  hhs = progressive_train_test.progressive_with_dropout_randaug(
      model,
      data_name="cifar10",
      lr_scheduler=lr_scheduler,
      total_epochs=36,
      batch_size=64,
      dropout_layer=-2,
      target_shapes=[128, 160, 192, 224], # [128, 185, 242, 300] for final shape (300, 300)
      dropouts=[0.1, 0.2, 0.3, 0.4],
      magnitudes=[5, 8, 12, 15],
  )

  with open("history_ev2s_imagenet_progressive_224.json", "w") as ff:
      json.dump(hhs, ff)
  ```
  ![](cifar10_progressive_train.svg)
## Related Projects
  - [tfa.layers.StochasticDepth](https://www.tensorflow.org/addons/api_docs/python/tfa/layers/StochasticDepth)
  - [Official efficientnetv2](https://github.com/google/automl/tree/master/efficientnetv2)
***
