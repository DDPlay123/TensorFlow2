"""
將預測輸出(tx, ty, tw, th)轉換成(x1, y1, x2, y2)
"""
import tensorflow as tf


class YoloOutputBoxLayer(tf.keras.layers.Layer):
    def __init__(self, anchors, output_layer=1, num_classes=80, training=False, **kwargs):
        super(YoloOutputBoxLayer, self).__init__(**kwargs)
        self.anchors = anchors
        self.num_classes = num_classes
        self.training = training
        # 輸入image size與輸出grid的倍率關係，例如:第一層輸出(output_layer=1)
        # 輸出grid大小為(13, 13)，乘上32可以還原回原輸入大小(416, 416)
        if output_layer == 1:
            self.grid_to_img_scale = 32
        elif output_layer == 2:
            self.grid_to_img_scale = 16
        else:
            self.grid_to_img_scale = 8

    def build(self, input_shape):
        self.grid_h, self.grid_w = input_shape[1:3]

    def call(self, inputs, **kwargs):
        """
        :param inputs:  (batch, grid_h, grid_w, anchors, (x, y, w, h, obj, ...classes))
        :param kwargs: None
        :return:
            bbox: (batch, grid_h, grid_w, anchors, (x1, y1, x2, y2))
            box_confidence: (batch, grid_h, grid_w, anchors, 1)
            box_class_probs: (batch, grid_h, grid_w, anchors, classes)
        """
        # 前一層網路層沒有輸出大小，則用tf.shape動態取得前一層輸出大小
        if self.grid_h is None:
            grid_h, grid_w = tf.shape(inputs)[1], tf.shape(inputs)[2]
        else:
            grid_h, grid_w = self.grid_h, self.grid_w

        box_xy, box_wh, box_confidence, box_class_probs = tf.split(inputs, (2, 2, 1, self.num_classes), axis=-1)
        # box_xy: (batch, grid_h, grid_w, anchors, [tx, ty])
        box_xy = tf.sigmoid(box_xy)  # scale to 0~1
        # box_confidence: (batch, grid_h, grid_w, anchors, confidence)
        box_confidence = tf.sigmoid(box_confidence)  # scale to 0~1
        # box_class_probs: (batch, grid_h, grid_w, anchors, classes)
        box_class_probs = tf.sigmoid(box_class_probs)  # scale to 0~1
        # pred_box: (batch, grid_h, grid_w, anchors, [tx, ty, tw, th]) 計算損失
        pred_box = tf.concat((box_xy, box_wh), axis=-1)

        grid = tf.meshgrid(tf.range(grid_w), tf.range(grid_h))
        grid = tf.stack(grid, axis=-1)  # (gx, gy, 2)
        grid = tf.expand_dims(grid, axis=2)  # (gx, gy, 1, 2)

        # box_xy: (batch, grid_h, grid_w, anchors, (x, y))
        box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast((grid_w, grid_h), tf.float32)
        # 計算輸入圖片大小
        img_w, img_h = (grid_w * self.grid_to_img_scale, grid_h * self.grid_to_img_scale)
        # box_wh: (batch, grid_h, grid_w, anchors, (w, h))
        box_wh = self.anchors * tf.exp(box_wh) / (img_w, img_h)

        # bbox: (x1, y1, x2, y2)
        box_x1y1 = box_xy - box_wh / 2
        box_x2y2 = box_xy + box_wh / 2
        bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)
        if self.training:
            return tf.concat([bbox, box_confidence, box_class_probs, pred_box], axis=-1)

        return bbox, box_confidence, box_class_probs

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'anchors': self.anchors,
            'num_classes': self.num_classes,
            'training': self.training
        })
        return config
