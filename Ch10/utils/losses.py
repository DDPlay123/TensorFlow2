"""
「自定義損失函數」的程式碼
"""
import tensorflow as tf


def reconstruction_loss(y_true, y_pred):
    """
    客製化 Reconstruction Loss 函數
    """
    # 對生成影像與輸入影像的每一個像素計算 BCE
    bce = -(y_true * tf.math.log(y_pred + 1e-07) +
            (1 - y_true) * tf.math.log(1 - y_pred + 1e-07))
    # tf.reduce_sum: 將每一個像素加總
    # tf.reduce_mean: 將 Batch 的資料計算平均值
    return tf.reduce_mean(tf.reduce_sum(bce, axis=[1, 2, 3]))
