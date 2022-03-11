"""
訓練 VAE 模型的程式碼
"""
import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from utils.model import create_vae_model
from utils.losses import reconstruction_loss
from utils.callbacks import SaveDecoderModel, SaveDecoderOutput


# 資料預處理，因為是非監督式學習，所以輸入及答案都是相同的影像資料
def parse_fn(dataset, input_size=(28, 28)):
    x = tf.cast(dataset['image'], tf.float32)
    # 將影像 resize 成網路輸入大小
    x = tf.image.resize(x, input_size)
    # 將影像標準化，縮放到 0 ~ 1 間
    x = x / 255.
    # 回傳訓練資料及答案
    return x, x


# 載入 MNIST 資料集
train_data, info = tfds.load('mnist', split="train", with_info=True)
valid_data = tfds.load('mnist', split="test")


# 設定 Datasets
AUTOTUNE = tf.data.experimental.AUTOTUNE  # 自動調整模式
batch_size = 16  # 批次大小
train_num = info.splits['train'].num_examples  # 訓練資料數量


# 打散資料
train_data = train_data.shuffle(train_num)
# 載入預處理 parse_fn()，CPU數量為自動調整模式
train_data = train_data.map(parse_fn, num_parallel_calls=AUTOTUNE)
# 設定批次大小為16，並將 prefetch 模式開啟(暫存空間為自動調整模式)
train_data = train_data.batch(batch_size).prefetch(buffer_size=AUTOTUNE)


# 載入預處理 parse_fn()，CPU數量為自動調整模式
valid_data = valid_data.map(parse_fn, num_parallel_calls=AUTOTUNE)
# 設定批次大小為16，並將 prefetch 模式開啟(暫存空間為自動調整模式)
valid_data = valid_data.batch(batch_size).prefetch(buffer_size=AUTOTUNE)


# 建立 Callback function
log_dirs = 'logs_vae'  # 建立儲存模型目錄
model_dir = log_dirs + '/models'
os.makedirs(model_dir, exist_ok=True)


# 將訓練紀錄存成TensorBoard的記錄檔
model_tb = keras.callbacks.TensorBoard(log_dir=log_dirs)
# 儲存最好的網路模型權重
model_sdw = SaveDecoderModel(model_dir + '/best_model.h5', monitor='val_loss')
# 儲存 Decoder 產生的影像，到 TensorBoard 記錄檔中
model_sdo = SaveDecoderOutput(28, log_dir=log_dirs)


# 建立 VAE 網路模型
input_shape = (28, 28, 1)  # 輸入大小
# 定義 Encoder 壓縮到多少維的空間向量
latent_dim = 2
vae_model = create_vae_model(input_shape, latent_dim)


# 設定訓練使用的優化器和損失函數
optimizer = tf.keras.optimizers.RMSprop()
vae_model.compile(optimizer, loss=reconstruction_loss)


# 訓練網路模型
vae_model.fit(train_data,
              epochs=20,
              validation_data=valid_data,
              callbacks=[model_tb, model_sdw, model_sdo])
