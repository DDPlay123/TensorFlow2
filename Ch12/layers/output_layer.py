"""
將輸出形狀reshape到(batch, grid_h, grid_w, num_anchors, classes + 5)大小
"""
import tensorflow as tf


class YoloOutputLayer(tf.keras.layers.Layer):
    def __init__(self, num_anchors, num_classes, **kwargs):
        super(YoloOutputLayer, self).__init__(**kwargs)
        self.num_anchors = num_anchors
        self.num_classes = num_classes

    def build(self, input_shape):
        self.input_h, self.input_w = input_shape[1:3]

    def call(self, inputs, **kwargs):
        if self.input_h is None or self.input_w is None:
            # 前一層輸入長、寬為None的話，則輸出大小浮動
            inputs = tf.reshape(inputs, (-1, tf.shape(inputs)[1], tf.shape(inputs)[2],
                                self.num_anchors, self.num_classes + 5))
        else:
            # 前一層輸入長、寬有值傳入，則長、寬固定大小
            inputs = tf.reshape(inputs, (-1, self.input_h, self.input_w,
                                self.num_anchors, self.num_classes + 5))
        return inputs

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_anchors': self.num_anchors,
            'num_classes': self.num_classes
        })
        return config

