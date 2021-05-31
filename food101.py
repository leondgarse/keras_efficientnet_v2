import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np


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


def init_dataset(data_name="food101", target_shape=(300, 300), batch_size=64, buffer_size=1000, info_only=False, magnitude=0, keep_shape=False):
    dataset, info = tfds.load(data_name, with_info=True)
    num_classes = info.features["label"].num_classes
    total_images = info.splits["train"].num_examples
    if info_only:
        return total_images, num_classes

    AUTOTUNE = tf.data.AUTOTUNE
    train_process = RandomProcessImage(target_shape, magnitude, keep_shape=keep_shape)
    train = dataset["train"].map(lambda xx: train_process(xx), num_parallel_calls=AUTOTUNE)
    test_process = RandomProcessImage(target_shape, magnitude=-1, keep_shape=keep_shape)
    if "validation" in dataset:
        test = dataset["validation"].map(lambda xx: test_process(xx))
    elif "test" in dataset:
        test = dataset["test"].map(lambda xx: test_process(xx))

    as_one_hot = lambda xx, yy: (xx, tf.one_hot(yy, num_classes))
    train_dataset = train.shuffle(buffer_size).batch(batch_size).map(as_one_hot).prefetch(buffer_size=AUTOTUNE)
    test_dataset = test.batch(batch_size).map(as_one_hot)
    return train_dataset, test_dataset, total_images, num_classes


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
        if isinstance(hist, str):
            with open(hist, "r") as ff:
                hist = json.load(ff)
        name = names[id] if names != None else os.path.splitext(os.path.basename(hist))[0]

        axes[0].plot(hist["loss"], label=name + " loss")
        color = axes[0].lines[-1].get_color()
        axes[0].plot(hist["val_loss"], label=name + " val_loss", color=color, linestyle="--")
        axes[1].plot(hist["accuracy"], label=name + " accuracy")
        color = axes[1].lines[-1].get_color()
        axes[1].plot(hist["val_accuracy"], label=name + " val_accuracy", color=color, linestyle="--")
        axes[2].plot(hist["lr"], label=name + " lr")
    for ax in axes:
        ax.legend()
        ax.grid()
    fig.tight_layout()
    return fig
