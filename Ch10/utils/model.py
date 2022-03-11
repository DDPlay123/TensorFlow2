"""
「VAE網路模型」和「自定義網路層」的程式碼
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras


def create_vae_model(input_shape, latent_dim):
    """
    包含 Encoder 和 Decoder
    """
    # 定義 Encoder 模型
    img_inputs = keras.Input(input_shape)
    x = keras.layers.Conv2D(32, 3, padding='same', activation='relu')(img_inputs)
    x = keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    # 儲存Flatten前的特徵大小
    shape_before_flatten = x.shape
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    # 輸出平均值 μ
    z_mean = keras.layers.Dense(latent_dim)(x)
    # 輸出變異數 σ^2
    z_log_var = keras.layers.Dense(latent_dim)(x)
    # 自定義網路層
    z = Sampling()([z_mean, z_log_var])
    # 建立 Encoder Model
    encoder = keras.Model(inputs=img_inputs, outputs=z, name='encoder')
    encoder.summary()

    # 定義 Decoder 模型
    latent_inputs = keras.Input((latent_dim,))
    # 產生和 Encoder Flatten 前的一樣大小的特徵
    x = keras.layers.Dense(np.prod(shape_before_flatten[1:]), activation='relu')(latent_inputs)
    # 將特徵 Reshape 成 Encoder Flatten 前的形狀
    x = keras.layers.Reshape(target_shape=shape_before_flatten[1:])(x)
    x = keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    img_outputs = keras.layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)
    # 建立 Decoder Model
    decoder = keras.Model(inputs=latent_inputs, outputs=img_outputs, name='decoder')
    decoder.summary()

    # 建立 VAE Model
    z = encoder(img_inputs)
    img_outputs = decoder(z)
    vae = keras.Model(inputs=img_inputs, outputs=img_outputs, name='vae')

    # 建立 KL Loss，對 VAE 中間層的輸出計算損失值，使用 vae.add_loss 來加入損失函數
    kl_loss = 0.5 * tf.reduce_mean(tf.exp(z_log_var) - (1 + z_log_var) + tf.square(z_mean))
    vae.add_loss(kl_loss)

    return vae


class Sampling(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        # Batch 大小
        batch = tf.shape(z_mean)[0]
        # Code 維度
        dim = tf.shape(z_mean)[1]
        # 產生標準常態分佈
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(z_log_var) * epsilon
