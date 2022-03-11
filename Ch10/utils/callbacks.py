"""
「自定義回調函數」的程式碼
此部分建立兩個回調函數
1. SaveDecoderModel: 每一個 epoch 都會去檢查網路模型有無進步，有的話儲存 VAE 中的 Decoder 模型(類似keras.callbacks.ModelCheckpoint)
2. SaveDecoderOutput: 每一個 epoch 都會產生225張影像，並儲存在 TensorBoard 記錄檔中，之後可以開啟 TensorBoard 觀察每個 epoch 的輸出變化
"""
import os.path
import numpy as np
import tensorflow as tf


class SaveDecoderModel(tf.keras.callbacks.Callback):
    def __init__(self, weights_file, monitor='loss', save_weights_only=False):
        super(SaveDecoderModel, self).__init__()
        # Decoder 模型儲存路徑
        self.weights_file = weights_file
        # 設定 best 為無限大
        self.best = np.Inf
        # 要監測的項目
        self.monitor = monitor
        # 儲存模型或模型權重
        self.save_weights_only = save_weights_only

    def on_epoch_end(self, epoch, logs=None):
        """
        在每個 epoch 結束時執行，如果網路模型有進步則儲存
        """
        loss = logs.get(self.monitor)
        if loss < self.best:
            if self.save_weights_only:
                # 儲存 Decoder 模型的權重
                self.model.get_layer('decoder').save_weights(self.weights_file)
            else:
                # 儲存 Decoder 模型
                self.model.get_layer('decoder').save(self.weights_file)
            self.best = loss


class SaveDecoderOutput(tf.keras.callbacks.Callback):
    def __init__(self, image_size, log_dir):
        super(SaveDecoderOutput, self).__init__()
        self.writer = None
        # 產生影像的大小
        self.size = image_size
        # TensorBoard 記錄檔儲存路徑
        self.log_dir = log_dir
        # 產生 (15 x 15) 個影像
        n = 15
        # 影像儲存及顯示的最大陣列 (可放入225張影像)
        self.save_images = np.zeros((image_size * n, image_size * n, 1))
        # 線性抽取15個數值，作為 Code 的 x
        self.grid_x = np.linspace(-1.5, 1.5, n)
        # 線性抽取15個數值，作為 Code 的 y
        self.grid_y = np.linspace(-1.5, 1.5, n)

    def on_train_begin(self, logs=None):
        """
        在開始訓練前會建立 TensorBoard 的紀錄檔
        """
        path = os.path.join(self.log_dir, 'images')
        self.writer = tf.summary.create_file_writer(path)

    def on_epoch_end(self, epoch, logs=None):
        """
        在每個 epoch 結束時執行，每次產生225張影像，並儲存在記錄檔
        """
        for i, yi in enumerate(self.grid_x):
            for j, xi in enumerate(self.grid_y):
                # 產生一組 Code
                z_sample = np.array([[xi, yi]])
                # Decoder 透過 Code 產生影像
                img = self.model.get_layer('decoder')(z_sample)
                # 將影像儲存到影像儲存顯示的陣列中
                self.save_images[i * self.size: (i + 1) * self.size,
                                 j * self.size: (j + 1) * self.size] = img.numpy()[0]
        # 將生成的225張影像儲存到 TensorBoard 記錄檔中
        with self.writer.as_default():
            tf.summary.image("Decoder output", [self.save_images], step=epoch)
