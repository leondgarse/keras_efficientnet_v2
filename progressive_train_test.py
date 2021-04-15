import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler

import efficientnet_v2


class RandomProcessImage:
    def __init__(self, target_shape=(300, 300), magnitude=0, keep_shape=False):
        self.target_shape, self.magnitude, self.keep_shape = target_shape, magnitude, keep_shape
        if magnitude > 0:
            import augment

            translate_const, cutout_const = 100, 40
            # translate_const = int(target_shape[0] * 10 / magnitude)
            # cutout_const = int(target_shape[0] * 40 / 224)
            print(
                ">>>> RandAugment: magnitude = %d, translate_const = %d, cutout_const = %d"
                % (magnitude, translate_const, cutout_const)
            )
            aa = augment.RandAugment(magnitude=magnitude, translate_const=translate_const, cutout_const=cutout_const)
            # aa.available_ops = ["AutoContrast", "Equalize", "Invert", "Rotate", "Posterize", "Solarize", "Color", "Contrast", "Brightness", "Sharpness", "ShearX", "ShearY", "TranslateX", "TranslateY", "Cutout", "SolarizeAdd"]
            self.process = lambda img: aa.distort(img)
        elif magnitude == 0:
            self.process = lambda img: tf.image.random_flip_left_right(img)
        else:
            self.process = lambda img: img

    def __call__(self, datapoint):
        image = datapoint["image"]
        if self.keep_shape:
            cropped_shape = tf.reduce_min(tf.keras.backend.shape(image)[:2])
            image = tf.image.random_crop(image, (cropped_shape, cropped_shape, 3))

        input_image = tf.image.resize(image, self.target_shape)
        label = datapoint["label"]
        input_image = self.process(input_image)
        input_image = (tf.cast(input_image, tf.float32) - 127.5) / 128
        return input_image, label


def init_dataset(target_shape=(300, 300), batch_size=64, buffer_size=1000, info_only=False, magnitude=0, keep_shape=False):
    dataset, info = tfds.load("food101", with_info=True)
    num_classes = info.features["label"].num_classes
    total_images = info.splits["train"].num_examples
    if info_only:
        return total_images, num_classes, steps_per_epoch

    AUTOTUNE = tf.data.AUTOTUNE
    train_process = RandomProcessImage(target_shape, magnitude, keep_shape=keep_shape)
    train = dataset["train"].map(lambda xx: train_process(xx), num_parallel_calls=AUTOTUNE)
    test_process = RandomProcessImage(target_shape, magnitude=-1, keep_shape=keep_shape)
    test = dataset["validation"].map(lambda xx: test_process(xx))

    as_one_hot = lambda xx, yy: (xx, tf.one_hot(yy, num_classes))
    train_dataset = train.shuffle(buffer_size).batch(batch_size).map(as_one_hot).prefetch(buffer_size=AUTOTUNE)
    test_dataset = test.batch(batch_size).map(as_one_hot)
    return train_dataset, test_dataset, total_images, num_classes


def exp_scheduler(epoch, lr_base=0.256, decay_step=2.4, decay_rate=0.97, lr_min=0, warmup=10):
    if epoch < warmup:
        lr = (lr_base - lr_min) * (epoch + 1) / (warmup + 1)
    else:
        lr = lr_base * decay_rate ** ((epoch - warmup) / decay_step)
        lr = lr if lr > lr_min else lr_min
    print("Learning rate for iter {} is {}".format(epoch + 1, lr))
    return lr


def progressive_with_dropout_randaug(
    compiled_model,
    lr_scheduler=None,
    total_epochs=36,
    stages=1,
    target_shapes=[128],
    dropouts=[0.4],
    dropout_layer=-2,
    magnitudes=[0],
):
    histories = []
    for stage, target_shape, dropout, magnitude in zip(range(stages), target_shapes, dropouts, magnitudes):
        if len(dropouts) > 1 and isinstance(model.layers[dropout_layer], keras.layers.Dropout):
            model.layers[dropout_layer].rate = dropout
            optimizer = model.optimizer
            model = keras.models.clone_model(model)  # Make sure it do changed
            model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["acc"])

        target_shape = (target_shape, target_shape)
        train_dataset, test_dataset, total_images, num_classes = init_dataset(target_shape=target_shape, magnitude=magnitude)

        initial_epoch = stage * total_epochs // stages
        epochs = (stage + 1) * total_epochs // stages
        history = model.fit(
            train_dataset,
            epochs=epochs,
            initial_epoch=initial_epoch,
            validation_data=test_dataset,
            callbacks=[lr_scheduler],
        )
        histories.append(history)
    hhs = {kk: np.ravel([hh.history[kk] for hh in histories]).tolist() for kk in history.history.keys()}
    return hhs


def image_crop_3(datapoint, target_shape=300):
    image, label = datapoint["image"], datapoint["label"]
    height, width = image.shape[:2]
    if height == width:
        resize_shape = (target_shape, target_shape)
        crops = [tf.image.resize(image, resize_shape)] * 3
    elif height < width:
        resize_shape = (target_shape, target_shape * width // height)
        image = tf.image.resize(image, resize_shape)
        crops = [
            image[:, :target_shape],
            image[:, (image.shape[1] - target_shape) // 2 : (image.shape[1] + target_shape) // 2],
            image[:, -target_shape:],
        ]
    else:
        resize_shape = (target_shape * height // width, target_shape)
        image = tf.image.resize(image, resize_shape)
        crops = [
            image[:target_shape, :],
            image[(image.shape[0] - target_shape) // 2 : (image.shape[0] + target_shape) // 2, :],
            image[-target_shape:, :],
        ]
    return np.array(crops), label


def model_pred_3(model, image_batch):
    preds = model((np.array(image_batch) - 127.5) / 127)
    pred_values, pred_classes = np.max(preds, -1), np.argmax(preds, -1)
    pred_values, pred_classes = pred_values.reshape(-1, 3), pred_classes.reshape(-1, 3)

    voted_classes, voted_values = [], []
    for pred_value, pred_class in zip(pred_values, pred_classes):
        if pred_class[0] == pred_class[1]:
            voted_class = pred_class[0]
            voted_value = max(pred_value[0], pred_value[1])
        elif pred_class[0] == pred_class[2]:
            voted_class = pred_class[0]
            voted_value = max(pred_value[0], pred_value[2])
        elif pred_class[1] == pred_class[2]:
            voted_class = pred_class[1]
            voted_value = max(pred_value[1], pred_value[2])
        else:
            voted_class = pred_class[np.argmax(pred_value)]
            voted_value = np.max(pred_value)
        voted_classes.append(voted_class)
        voted_values.append(voted_value)
    return voted_classes, voted_values, preds


def model_validation_3(model, batch_size=64):
    from tqdm import tqdm

    dataset, info = tfds.load("food101", with_info=True)
    batch_size = 64
    test_gen = dataset["validation"].as_numpy_iterator()
    total_test = info.splits["validation"].num_examples

    voted_classes, voted_values, labels, image_batch = [], [], [], []
    batch_size *= 3
    for id, datapoint in tqdm(enumerate(test_gen), total=total_test):
        crops, label = image_crop_3(datapoint)
        image_batch.extend(crops)
        labels.append(label)
        if id + 1 == total_test or len(image_batch) == batch_size:
            batch_voted_classes, batch_voted_values, batch_preds = model_pred_3(model, image_batch)
            voted_classes.extend(batch_voted_classes)
            voted_values.extend(batch_voted_values)
            image_batch = []

    voted_classes, voted_values, labels = np.array(voted_classes), np.array(voted_values), np.array(labels)
    print("crop_3_predict accuray:", (voted_classes == labels).sum() / labels.shape[0])
    return voted_classes, voted_values, labels


def plot_hists(hists, names=None):
    import os
    import json
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for id, hist in enumerate(hists):
        with open(hist, "r") as ff:
            aa = json.load(ff)
        name = names[id] if names != None else os.path.splitext(os.path.basename(hist))[0]

        axes[0].plot(aa["loss"], label=name + " loss")
        axes[0].plot(aa["val_loss"], label=name + " val_loss")
        axes[1].plot(aa["accuracy"], label=name + " accuracy")
        axes[1].plot(aa["val_accuracy"], label=name + " val_accuracy")
        axes[2].plot(aa["lr"], label=name + " lr")
    for ax in axes:
        ax.legend()
        ax.grid()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import json

    train_dataset, test_dataset, total_images, num_classes = init_dataset()
    print("total_images: %s, num_classes: %s" % (total_images, num_classes))

    total_epochs = 36
    lr_scheduler = LearningRateScheduler(
        lambda epoch: exp_scheduler(epoch, lr_base=0.256, decay_step=1, decay_rate=0.88, warmup=2)
    )

    print(">>>> Basic input_shape=(128, 128, 3), dropout=0")
    eb2 = efficientnet_v2.EfficientNetV2(input_shape=(None, None, 3), include_top=True, classes=num_classes)
    inited_weights = eb2.get_weights()
    # optmizer = keras.optimizers.RMSprop(learning_rate=0.256, momentum=0.9, decay=0.9)
    optmizer = keras.optimizers.SGD(momentum=0.9)
    eb2.compile(loss="sparse_categorical_crossentropy", optimizer=optmizer, metrics=["accuracy"])
    history = eb2.fit(train_dataset, epochs=total_epochs, validation_data=test_dataset, callbacks=[lr_scheduler])
    with open("basic_dropout_0.json", "w") as ff:
        json.dump(history.history, ff)

    print(">>>> Basic input_shape=(128, 128, 3), dropout=0.4")
    eb2 = efficientnet_v2.EfficientNetV2(input_shape=(None, None, 3), include_top=True, classes=num_classes, dropout=0.4)
    eb2.set_weights(inited_weights)
    optmizer = keras.optimizers.SGD(momentum=0.9)
    eb2.compile(loss="sparse_categorical_crossentropy", optimizer=optmizer, metrics=["accuracy"])
    history = eb2.fit(train_dataset, epochs=total_epochs, validation_data=test_dataset, callbacks=[lr_scheduler])
    with open("basic_dropout_0.4.json", "w") as ff:
        json.dump(history.history, ff)

    print(">>>> Progressive input_shape=[56, 80, 104, 128], dropout=[0.1, 0.2, 0.3, 0.4]")
    eb2 = efficientnet_v2.EfficientNetV2(input_shape=(None, None, 3), include_top=True, classes=num_classes, dropout=0.1)
    eb2.set_weights(inited_weights)
    optmizer = keras.optimizers.SGD(momentum=0.9)
    eb2.compile(loss="sparse_categorical_crossentropy", optimizer=optmizer, metrics=["accuracy"])
    hhs = progressive_with_dropout_randaug(
        eb2, 36, stages=4, target_shapes=[56, 80, 104, 128], dropouts=[0.1, 0.2, 0.3, 0.4], magnitudes=[5, 5, 5, 5]
    )
    with open("progressive.json", "w") as ff:
        json.dump(hhs, ff)
elif __name__ == "__train_test__":
    import json

    keras.mixed_precision.set_global_policy("mixed_float16")

    train_dataset, test_dataset, total_images, num_classes = init_dataset(
        target_shape=(300, 300), magnitude=15, keep_shape=True
    )
    # model = keras.applications.MobileNet(input_shape=(None, None, 3), include_top=False, weights='imagenet')
    model = keras.applications.InceptionV3(input_shape=(None, None, 3), weights="imagenet", include_top=False)
    # model = keras.applications.ResNet50(input_shape=(None, None, 3), weights='imagenet', include_top=False)
    inputs = model.inputs[0]
    nn = model.outputs[0]
    nn = keras.layers.GlobalAveragePooling2D()(nn)
    nn = keras.layers.Dropout(0.4)(nn)
    nn = keras.layers.Activation("linear", dtype="float32")(nn)
    nn = keras.layers.Dense(
        num_classes, kernel_regularizer=keras.regularizers.l2(0.0005), activation="softmax", name="predictions", dtype="float32"
    )(nn)
    model = keras.models.Model(inputs, nn)

    total_epochs = 52
    lr_scheduler = LearningRateScheduler(
        lambda epoch: exp_scheduler(epoch, lr_base=0.01, decay_step=1, decay_rate=0.9, warmup=4)
    )
    optmizer = keras.optimizers.SGD(momentum=0.9)
    # loss = "categorical_crossentropy"
    loss = keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.1)
    model.compile(loss=loss, optimizer=optmizer, metrics=["accuracy"])
    history = model.fit(train_dataset, epochs=total_epochs, validation_data=test_dataset, callbacks=[lr_scheduler])

    hhs = {kk: np.array(vv, "float").tolist() for kk, vv in history.history.items()}
    with open("inceptionV3_magnitude_10.json", "w") as ff:
        json.dump(hhs, ff)
    _ = model_validation_3(model)
    plot_hist(["inceptionV3_magnitude_15_keep_shape_true_ls_01.json"], names=["aa"])
