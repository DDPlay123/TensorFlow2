"""
測試 VAE 模型的程式碼
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# 輸出圖片大小
size = 28
# 產生 (15 x 15)個影像
n = 15
# 影像儲存及顯示的最大陣列 (可以放入225張影像)
save_images = np.zeros((size * n, size * n, 1))
# 線性抽取15個數值，作為 Code 的 x
grid_x = np.linspace(-1.5, 1.5, n)
# 線性抽取15個數值，作為 Code 的 y
grid_y = np.linspace(-1.5, 1.5, n)


# 載入訓練好的 Decoder 模型
model = tf.keras.models.load_model('logs_vae/models/best_model.h5')
for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        # 產生一組 Code
        z_sample = np.array([[xi, yi]])
        # Decoder 透過 Code 產生影像
        img = model(z_sample)
        # 將影像儲存到影像儲存顯示的陣列中
        save_images[i * size: (i + 1) * size,
                    j * size: (j + 1) * size] = img.numpy()[0]
# 顯示 Decoder 的預測結果，總共預測出225張圖片
plt.imshow(save_images[..., 0], cmap='gray')
plt.show()
