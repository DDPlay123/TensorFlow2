"""
YOLO v3架構
"""
from tensorflow.keras import layers, Model, Input
from layers.output_layer import YoloOutputLayer
from layers.output_box_layer import YoloOutputBoxLayer
from layers.nms_layer import NMSLayer
from model.darknet import darknet_body, darknetconv2d_bn_leaky, darknetconv2d
import config


yolo_anchors = config.yolo_anchors
def yolov3(input_size, anchors=yolo_anchors, num_classes=80, iou_threshold=0.5, score_threshold=0.5, training=False):
    """ 建立YOLO_V3網路模型(訓練和測試模式) """
    # 將 anchors boxes 平均分配給三個輸出
    num_anchors = len(anchors) // 3
    # 建立輸入層
    inputs = Input(input_size)
    # 建立 Darknet-53，並輸出三種不同尺度的網路層
    x_26, x_43, x = darknet_body(name='Yolo_DarkNet')(inputs)
    # 輸出層 y1 shape: (13, 13, 3, classes + 5)
    x, y1 = make_last_layers(x, 512, num_anchors, num_classes)

    x = darknetconv2d_bn_leaky(x, 256, (1, 1))
    # 上採樣 (13, 13, 256) -> (26, 26, 256)
    x = layers.UpSampling2D(2)(x)
    # Concat (26, 26, 256) + (26, 26, 512) = (26, 26, 768)
    x = layers.Concatenate()([x, x_43])
    # 輸出層 y2 shape: (26, 26, 3, classes + 5)
    x, y2 = make_last_layers(x, 256, num_anchors, num_classes)

    x = darknetconv2d_bn_leaky(x, 128, (1, 1))
    # 上採樣 (26, 26, 128) -> (52, 52, 128)
    x = layers.UpSampling2D(2)(x)
    # Concat (52, 52, 128) + (52, 52, 256) = (52, 52, 384)
    x = layers.Concatenate()([x, x_26])
    # 輸出層 y3 shape: (52, 52, 3, classes + 5)
    x, y3 = make_last_layers(x, 128, num_anchors, num_classes)

    # 如果為測試模式，加入以下幾層自訂層
    h, w, _ = input_size
    # 將輸出 anchors box 參數(tx, ty, tw, th)轉換成(x1, y1, x2, y2)
    y1 = YoloOutputBoxLayer(anchors[6:], 1, num_classes, training)(y1)
    y2 = YoloOutputBoxLayer(anchors[3:6], 2, num_classes, training)(y2)
    y3 = YoloOutputBoxLayer(anchors[0:3], 3, num_classes, training)(y3)

    # 如果為訓練模式，建立訓練用網路
    if training:
        return Model(inputs, (y1, y2, y3), name='Yolo-V3')
    # 移除重複的預測框
    outputs = NMSLayer(num_classes, iou_threshold, score_threshold)([y1, y2, y3])
    # 建立測試(預設)用的網路模型
    return Model(inputs, outputs, name='Yolo-V3')


def make_last_layers(x, num_filters, num_anchors, num_classes):
    """ 使用在 YOLO v3 最後一層卷積層 """
    out_filters = num_anchors * (num_classes + 5)
    x = darknetconv2d_bn_leaky(x, num_filters, (1, 1))
    x = darknetconv2d_bn_leaky(x, num_filters * 2, (3, 3))
    x = darknetconv2d_bn_leaky(x, num_filters, (1, 1))
    x = darknetconv2d_bn_leaky(x, num_filters * 2, (3, 3))
    x = darknetconv2d_bn_leaky(x, num_filters, (1, 1))
    y = darknetconv2d_bn_leaky(x, num_filters * 2, (3, 3))
    # 將特徵圖壓縮到(batch, grid_h, grid_w, num_anchors * (classes + 5))大小
    y = darknetconv2d(y, out_filters, (1, 1), num_classes=num_classes)
    # 將輸出reshape到(batch, grid_h, grid_w, num_anchors, classes + 5)大小
    y = YoloOutputLayer(num_anchors, num_classes)(y)
    return x, y
