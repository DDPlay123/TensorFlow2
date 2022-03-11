"""
「WGAN-GP」網路模型的程式碼
"""
from tensorflow import keras


def Generator(input_shape=(1, 1, 128), name='Generator'):
    inputs = keras.Input(shape=input_shape)

    # 1: Convolution Transpose Block(1), 1 x 1 -> 4 x 4
    x = keras.layers.Conv2DTranspose(512, 4, strides=1, padding='valid', use_bias=False)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    # 2: Convolution Transpose Block(2), 4 x 4 -> 8 x 8
    x = keras.layers.Conv2DTranspose(256, 4, strides=2, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    # 3: Convolution Transpose Block(3), 8 x 8 -> 16 x 16
    x = keras.layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    # 4: Convolution Transpose Block(4), 16 x 16 -> 32 x 32
    x = keras.layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    # 5: Convolution Transpose + Tanh, 32 x 32 -> 64 x 64
    x = keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same', use_bias=False)(x)
    outputs = keras.layers.Activation('tanh')(x)
    return keras.Model(inputs=inputs, outputs=outputs, name=name)


def Discriminator(input_shape=(64, 64, 3), name='Discriminator'):
    inputs = keras.Input(shape=input_shape)

    # 1: Convolution + LeakReLU, 64 x 64 -> 32 x 32
    x = keras.layers.Conv2D(64, 4, strides=2, padding='same')(inputs)
    x = keras.layers.LeakyReLU()(x)
    # 2: Convolution Block(1), 32 x 32 -> 16 x 16
    x = keras.layers.Conv2D(128, 4, strides=2, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    # 3: Convolution Block(2), 16 x 16 -> 8 x 8
    x = keras.layers.Conv2D(256, 4, strides=2, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    # 4: Convolution Block(3), 8 x 8 -> 4 x 4
    x = keras.layers.Conv2D(512, 4, strides=2, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    # 5: Convolution, 4 x 4 -> 1 x 1
    outputs = keras.layers.Conv2D(1, 4, strides=1, padding='valid')(x)
    return keras.Model(inputs=inputs, outputs=outputs, name=name)
