import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler
import efficientnet_v2
import food101


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
    dropout_layer=-3,
    magnitudes=[0],
):
    histories = []
    for stage, target_shape, dropout, magnitude in zip(range(stages), target_shapes, dropouts, magnitudes):
        if len(dropouts) > 1 and isinstance(model.layers[dropout_layer], keras.layers.Dropout):
            print(">>>> Changing dropout rate to:", dropout)
            model.layers[dropout_layer].rate = dropout
            # loss, optimizer, metrics = model.loss, model.optimizer, model.metrics
            # model = keras.models.clone_model(model)  # Make sure it do changed
            # model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        target_shape = (target_shape, target_shape)
        train_dataset, test_dataset, total_images, num_classes = food101.init_dataset(
            target_shape=target_shape, magnitude=magnitude, keep_shape=True
        )

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
    hhs = {kk: np.ravel([hh.history[kk] for hh in histories]).astype("float").tolist() for kk in history.history.keys()}
    return hhs


if __name__ == "__main__":
    import json

    train_dataset, test_dataset, total_images, num_classes = food101.init_dataset()
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

    train_dataset, test_dataset, total_images, num_classes = food101.init_dataset(
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
    optimizer = keras.optimizers.SGD(momentum=0.9)
    loss = "categorical_crossentropy"
    # loss = keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.1)
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    history = model.fit(train_dataset, epochs=total_epochs, validation_data=test_dataset, callbacks=[lr_scheduler])

    hhs = {kk: np.array(vv, "float").tolist() for kk, vv in history.history.items()}
    with open("inceptionV3_magnitude_10.json", "w") as ff:
        json.dump(hhs, ff)
    _ = food101.model_validation_3(model)
    food101.plot_hist(["inceptionV3_magnitude_15_keep_shape_true_ls_01.json"], names=["aa"])

    hhs = progressive_with_dropout_randaug(
        model, lr_scheduler, 52, stages=4, target_shapes=[128, 185, 242, 300], dropouts=[0.1, 0.2, 0.3, 0.4], magnitudes=[5, 8, 12, 15]
    )
