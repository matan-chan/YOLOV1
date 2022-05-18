from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, BatchNormalization, MaxPool2D
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.models import Sequential, Model


def build_model():
    """
    :return: the discriminator model.
    """
    img_shape = (448, 448, 3)
    model = Sequential(name="classifier")
    #                                                           3
    model.add(Conv2D(64, 7, 2, input_shape=img_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(MaxPool2D(2, 2))

    model.add(Conv2D(192, 3, 1, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(MaxPool2D(2, 2))

    model.add(Conv2D(128, 1, 1, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(256, 3, 1, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(256, 1, 1, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(512, 3, 1, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(MaxPool2D(2, 2))

    for _ in range(4):
        model.add(Conv2D(256, 1, 1, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv2D(512, 3, 1, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(512, 1, 1, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(1024, 3, 1, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(MaxPool2D(2, 2))

    for _ in range(2):
        model.add(Conv2D(512, 1, 1, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv2D(1024, 3, 1, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(1024, 1, 1, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(1024, 2, 1, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(1024, 1, 1, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(1024, 1, 1, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Flatten())

    model.add(Dense(4096))
    model.add(LeakyReLU(alpha=0.1))
    S, B, C = 7, 2, 20
    model.add(Dense(S * S * (C + B * 5)))
    model.add(Reshape((S, S, C + B * 5)))

    model.summary()
    image = Input(shape=img_shape)
    validity = model(image)
    return Model(image, validity)
