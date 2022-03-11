"""
YOLO V3 的損失函數
"""
import tensorflow as tf
from tensorflow.python.keras.losses import binary_crossentropy, sparse_categorical_crossentropy


def yolo_loss(y_true, y_pred, anchors, num_classes=80, ignore_thresh=0.5):
    """
    :param y_true: labels inputs
    :param y_pred: predict outputs
    :param anchors: layer anchors size:(3, 2)
    :param num_classes: number of classes in dataset
    :param ignore_thresh: if (IoU < threshold) and ignore
    :return: total loss (xy_loss + wh_loss + confidence_loss + class_loss)
    """
    # 1. 轉換預測輸出
    # y_pred: (batch_size, grid_h, grid_w, anchors, [x1, y1, x2, y2, obj, ...classes, tx, ty, tw, th])
    # y_pred為YoloOutputBoxLayer層的輸出，這裡分成4個部分:
    # pred_box: (batch_size, grid_h, grid_w, anchors, [x1, y1, x2, y2])用於計算與真實框的IoU
    # pred_obj: (batch_size, grid_h, grid_w, anchors, obj)用於計算confidence_loss
    # pred_class: (batch_size, grid_h, grid_w, anchors, classes)用於計算class_loss
    # pred_xywh: (batch_size, grid_h, grid_w, anchors, [tx, ty, tw, th])用於計算xy_loss, wh_loss
    pred_box, pred_obj, pred_class, pred_xywh = tf.split(y_pred, (4, 1, num_classes, 4), axis=-1)

    pred_xy = pred_xywh[..., 0:2]
    pred_wh = pred_xywh[..., 2:4]

    # 2. 轉換真實值(x1, y1, x2, y2) -> (x, y, w, h)
    true_box, true_obj, true_class_idx = tf.split(y_true, (4, 1, 1), axis=-1)
    true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
    true_wh = true_box[..., 2:4] - true_box[..., 0:2]
    # 因為小物件計算的損失值較小，所以給小物件乘上"權重係數"
    box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

    # 3. 將真實值(x1, y1, x2, y2)轉成(tx, ty, tw, th)並與pred_xywh計算損失
    grid_h, grid_w = tf.shape(y_true)[1], tf.shape(y_true)[2]
    grid = tf.meshgrid(tf.range(grid_w), tf.range(grid_h))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
    true_xy = true_xy * (grid_h, grid_w) - tf.cast(grid, true_xy.dtype)
    true_wh = tf.math.log(true_wh / anchors)
    true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)

    # 4. 產生遮罩(有無物件) shape(batch_size, grid_h, grid_w, anchors)
    obj_mask = tf.squeeze(true_obj, -1)     # (batch_size, grid, grid, anchors)
    # 5. 產生負樣本遮罩，無視iou超過threshold無視false positive
    # 取得有物件存在的物件框，true_box_flat = (N, [x1, y1, x2, y2])
    true_box_flat = tf.boolean_mask(true_box, tf.cast(obj_mask, tf.bool))
    # 計算預測真實物件與預測物件的iou
    best_iou = tf.reduce_max(broadcast_iou(pred_box, true_box_flat), axis=-1)
    # 產生遮罩，如果iou < ignore_thresh
    ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

    # 6. 計算損失函數
    xy_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
    wh_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
    obj_loss = binary_crossentropy(true_obj, pred_obj)
    confidence_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss
    class_loss = obj_mask * sparse_categorical_crossentropy(true_class_idx, pred_class)

    # 7. 總和所有預測框的損失值 (batch_size, grid_h, grid_w, anchors) -> (batch_size, 1)
    xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
    wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
    confidence_loss = tf.reduce_sum(confidence_loss, axis=(1, 2, 3))
    class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

    return xy_loss + wh_loss + confidence_loss + class_loss


# 再透過繼承tf.keras.losses.LossFunctionWrapper物件將yolo_loss損失函數封裝
class YoloLoss(tf.python.keras.losses.LossFunctionWrapper):
    def __init__(self,
                 anchors,
                 num_classes=80,
                 ignore_thresh=0.5,
                 name='yolo_loss'):
        super(YoloLoss, self).__init__(
            yolo_loss,
            name=name,
            anchors=anchors,
            num_classes=num_classes,
            ignore_thresh=ignore_thresh)


def broadcast_iou(pred_box, true_box):
    """
    計算真實框和預測框之間的IoU
    :param pred_box: size(b, gx, gy, 3, 4)
    :param true_box: size(n, 4)
    :return: Intersection over Union(IoU)
    """
    # broadcast boxes
    pred_box = tf.expand_dims(pred_box, -2)   # (b, gx, gy, 3, 1, 4)
    true_box = tf.expand_dims(true_box, 0)    # (1, n, 4)
    # new_shape: (b, gx, gy, 3, n, 4)
    new_shape = tf.broadcast_dynamic_shape(tf.shape(pred_box), tf.shape(true_box))
    pred_box = tf.broadcast_to(pred_box, new_shape)
    true_box = tf.broadcast_to(true_box, new_shape)

    # Overlap: (b, gx, gy, 3, n)
    int_w = tf.maximum(tf.minimum(pred_box[..., 2], true_box[..., 2]) -
                       tf.maximum(pred_box[..., 0], true_box[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(pred_box[..., 3], true_box[..., 3]) -
                       tf.maximum(pred_box[..., 1], true_box[..., 1]), 0)
    int_area = int_w * int_h

    # box size: w * h
    box_1_area = (pred_box[..., 2] - pred_box[..., 0]) * \
        (pred_box[..., 3] - pred_box[..., 1])
    box_2_area = (true_box[..., 2] - true_box[..., 0]) * \
        (true_box[..., 3] - true_box[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)