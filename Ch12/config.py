"""
設定模型參數、訓練參數或權重路徑的地方
"""
import numpy as np


# YOLO Anchor boxes 大小設定
yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)], np.float32)
# YOLO Anchor boxes遮罩，ex: 6,7,8給第一層輸出，3,4,5給第二層輸出
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
# YOLO 模型輸入設定
size_h = 416
size_w = 416
# 訓練分為兩個step:訓練 Darknet-53 以外的網路層 和 訓練整個網路
step1_batch_size = 32
step1_learning_rate = 1e-3
step1_start_epochs = 0
step1_end_epochs = 10  # 100
step2_batch_size = 8
step2_learning_rate = 1e-4
step2_start_epochs = step1_end_epochs
step2_end_epochs = step1_end_epochs + 10  # 100

# Pre-Train weights(透過convert.py轉換權重檔)
yolo_weights = 'model_data/yolo_weights.h5'

# coco 資料集的類別 (list順序會對應到yolo_weights.h5的輸出類別)
coco_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush']
voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
