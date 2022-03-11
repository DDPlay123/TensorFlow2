"""
測試物件偵測的程式碼
"""
import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import config
from model.yolo import yolov3
from utils.dataset import parse_fn_test
from utils.utils import trainable_model


# 使用測試集共4952筆資料
test_data = tfds.load("voc", split="test")
# 載入模型
# weight_file = 'model_data/yolo_weights.h5'
weight_file = 'logs_yolo/models/best_010.h5'

if weight_file == 'model_data/yolo_weights.h5':
    # COCO weights
    classes_list = config.coco_classes
    num_classes = len(config.coco_classes)
    freeze = False
else:
    # VOC 2007 weights
    classes_list = config.voc_classes
    num_classes = len(config.voc_classes)
    if int(os.path.splitext(weight_file)[0].split('_')[-1]) <= 100:
        freeze = True

# 建立 YOLO v3 網路模型
model = yolov3((config.size_h, config.size_w, 3), num_classes=num_classes, training=False)


# 如果權重為step1凍結Darknet-53所有訓練參數
if freeze:
    # Freeze all layers in except last layer
    trainable_model(model, trainable=False)
    model.get_layer('conv2d_last_layer1_20').trainable = True
    model.get_layer('conv2d_last_layer2_20').trainable = True
    model.get_layer('conv2d_last_layer3_20').trainable = True
# 載入模型權重
model.load_weights(weight_file)


def test_and_show_result(model, test_number=10):
    for img_count, data in enumerate(test_data.take(test_number)):
        # 讀取影像數據，作為顯示用
        org_img = data['image'].numpy()
        h, w, _ = data['image'].shape
        # 將數據進行前處理
        img, bboxes = parse_fn_test(data)
        # 預測影像
        boxes, scores, classes, nums = model.predict(tf.expand_dims(img, axis=0))
        boxes, scores, classes, nums = boxes[0], scores[0], classes[0], int(nums[0])
        for i in range(nums):
            # 將預測的物件框標示在圖片上
            x1y1 = tuple((np.array(boxes[i][0:2]) * (w, h)).astype(np.int32))
            x2y2 = tuple((np.array(boxes[i][2:4]) * (w, h)).astype(np.int32))
            cv2.rectangle(org_img, x1y1, x2y2, (255, 0, 0), 2)
            # 將預測的物件類別顯示在圖片上
            cv2.putText(org_img,
                        '{} {:.4f}'.format(classes_list[int(classes[i])], scores[i]),
                        x1y1,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2)
        plt.figure()
        plt.imshow(org_img)
        plt.imsave('output_images/output_{}.png'.format(img_count), org_img)
    plt.show()


test_and_show_result(model, test_number=5)
