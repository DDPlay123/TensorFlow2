"""
訓練物件偵測的程式碼
"""
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import config
from losses.yolo_loss import YoloLoss
from model.yolo import yolov3
from utils.utils import trainable_model
from utils.dataset import parse_aug_fn, parse_fn, transform_targets


# 讀取各層輸出的Anchor boxes遮罩
anchor_masks = config.yolo_anchor_masks
# 讀取dataset的類別數量
num_classes = len(config.voc_classes)


# 建立YOLO v3 網路模型
model = yolov3((config.size_h, config.size_w, 3), num_classes=num_classes, training=True)
# 載入預訓練權重
model.load_weights(config.yolo_weights, by_name=True)


# 建立logs目錄
log_dir = 'logs_yolo'
model_dir = log_dir + '/models'
os.makedirs(model_dir, exist_ok=True)
# 儲存訓練記錄檔
model_tb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
# 儲存最好的網路模型權重
model_mckp = tf.keras.callbacks.ModelCheckpoint(model_dir + '/best_{epoch:03d}.h5',
                                                monitor='val_loss',
                                                save_best_only=True,
                                                mode='min')
# 當10個epoch val_loss沒有持續下降，降低Learning rate
model_rlr = tf.keras.callbacks.ReduceLROnPlateau(verbose=1)


# YOLO v3 訓練函數
def training_model(model, callbacks, num_classes=80, step=1):
    # 設定訓練參數
    if step == 1:
        batch_size = config.step1_batch_size
        learning_rate = config.step1_learning_rate
        start_epochs = config.step1_start_epochs
        end_epochs = config.step1_end_epochs
    else:
        batch_size = config.step2_batch_size
        learning_rate = config.step2_learning_rate
        start_epochs = config.step2_start_epochs
        end_epochs = config.step2_end_epochs

    anchors = config.yolo_anchors / 416

    AUTOTUNE = tf.data.experimental.AUTOTUNE  # 自動調整模式

    # Training dataset setting
    # 使用訓練集和驗證集做為訓練資料，共5011筆資料
    train_data, info = tfds.load("voc", split="train+validation", with_info=True)
    # 打散資料
    train_data = train_data.shuffle(1000)
    # 資料標準化和資料增強，CPU 數量為自動調整模式
    train_data = train_data.map(lambda dataset: parse_aug_fn(dataset), num_parallel_calls=AUTOTUNE)
    # 設定一筆訓練資料的數量
    train_data = train_data.batch(batch_size)
    # 訓練目標轉換，CPU 數量為自動調整模式
    train_data = train_data.map(lambda x, y: transform_targets(x, y, anchors, anchor_masks), num_parallel_calls=AUTOTUNE)
    # 開啟prefetch模式(暫存空間為自動調整模式)
    train_data = train_data.prefetch(buffer_size=AUTOTUNE)

    # Validation dataset setting
    # 使用測試集做為訓練資料，共4952筆資料
    valid_data = tfds.load("voc", split="test")
    # 資料標準化，CPU 數量為自動調整模式
    valid_data = valid_data.map(lambda dataset: parse_fn(dataset), num_parallel_calls=AUTOTUNE)
    # 設定一筆測試資料的數量
    valid_data = valid_data.batch(batch_size)
    # 訓練目標轉換，CPU 數量為自動調整模式
    valid_data = valid_data.map(lambda x, y: transform_targets(x, y, anchors, anchor_masks), num_parallel_calls=AUTOTUNE)
    # 開啟prefetch模式(暫存空間為自動調整模式)
    valid_data = valid_data.prefetch(buffer_size=AUTOTUNE)

    # 設定優化器
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    # 設定損失函數
    model.compile(optimizer=optimizer, loss=[YoloLoss(anchors[mask],
                                                      num_classes=num_classes) for mask in anchor_masks],
                                                      run_eagerly=False)
    # 訓練網路模型
    model.fit(train_data,
              epochs=end_epochs,
              callbacks=callbacks,
              validation_data=valid_data,
              initial_epoch=start_epochs)


# 訓練 YOLO v3 Step1
# 除了最後一層，凍結YOLO v3所有網路層
trainable_model(model, trainable=False)
model.get_layer('conv2d_last_layer1_20').trainable = True
model.get_layer('conv2d_last_layer2_20').trainable = True
model.get_layer('conv2d_last_layer3_20').trainable = True

# Step1: 訓練YOLO v3網路模型最後一層
training_model(model,
               callbacks=[model_tb, model_mckp, model_rlr],
               num_classes=num_classes,
               step=1)


# 訓練 YOLO v3 Step2
# 解凍YOLO v3所有網路層
trainable_model(model, trainable=True)
# Step2: 訓練整個YOLO v3網路模型
training_model(model,
               callbacks=[model_tb, model_mckp, model_rlr],
               num_classes=num_classes,
               step=2)
