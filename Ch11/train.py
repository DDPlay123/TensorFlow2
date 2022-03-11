"""
訓練 WGAN-GP 模型的程式碼
"""
import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from functools import partial
from utils.dataset import parse_fn
from utils.losses import generator_loss, discriminator_loss, gradient_penalty
from utils.models import Generator, Discriminator


dataset = 'aflw2k3d'  # aflw2k3d、celeb_a
# 設定訓練參數
batch_size = 64
lr = 0.0001  # Learning Rate
z_dim = 128  # 輸入 Generator 的 domain 大小為128維
n_dis = 5  # 每訓練5次 Discriminator，訓練一次Generator
gradient_penalty_weight = 10.0  # 設定 gradient_penalty 係數，通常設定為 10


# 載入 CelebA Dataset，並設定 Dataset
train_data, info = tfds.load(dataset, split='train', with_info=True)
AUTOTUNE = tf.data.experimental.AUTOTUNE  # 自動調整模式
buffer_size = 1000  # 緩存空間

# 打散資料集
train_data = train_data.shuffle(buffer_size)
# 載入預處理 parse_fn()，CPU數量為自動調整模式
train_data = train_data.map(parse_fn, num_parallel_calls=AUTOTUNE)
# 設定批次大小為64，如果最後一批資料小於64則捨棄，並將 prefetch 模式開啟(暫存空間為自動調整模式)
train_data = train_data.batch(batch_size, drop_remainder=True).prefetch(buffer_size=AUTOTUNE)


# 建立 Generator 和 Discriminator 網路模型
generator = Generator((1, 1, z_dim))
discriminator = Discriminator((64, 64, 3))


# 設定 Generator 和 Discriminator 訓練優化器
g_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.9)
d_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.9)


# 建立 Generator 訓練函數
@tf.function
def train_generator():
    """
    函數會去計算 Generator 的損失函數，並計算梯度更新 Generator 網路權重。
    @tf.function:
        TensorFlow2預設為「Eager Execution」動態圖模式，這個模式一但運算執行，就會立刻回傳數值，提供更靈活的使用，但可能會犧牲一些效能。
        為了使效能最佳化，TensorFlow2推出「@tf.function」修飾器，在這個修飾器下的函數，會透過「AutoGraph」的工具，將程式碼轉為靜態計算圖。
    """
    with tf.GradientTape() as tape:
        # 從常態分佈中產生 128維 的 Random vector 作為 Generator
        random_vector = tf.random.normal(shape=(batch_size, 1, 1, z_dim))
        # 從 Generator 中產生假的圖片
        fake_img = generator(random_vector, training=True)
        # 使用 Discriminator 評估生成圖片
        fake_logit = discriminator(fake_img, training=True)
        # 計算 Generator Loss
        g_loss = generator_loss(fake_logit)
    # 計算梯度
    gradients = tape.gradient(g_loss, generator.trainable_variables)
    # 更新 Generator 權重
    g_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
    return g_loss


# 建立 Discriminator 訓練函數
@tf.function
def train_discriminator(real_img):
    """
    函數會去計算 Discriminator 的損失函數，並計算梯度更新 Generator 網路權重。
    """
    with tf.GradientTape() as tape:
        # 從常態分佈中產生 128維 的 Random vector 作為 Generator
        random_vector = tf.random.normal(shape=(batch_size, 1, 1, z_dim))
        # 從 Generator 中產生假的圖片
        fake_img = generator(random_vector, training=True)
        # 使用 Discriminator 評估真實圖片
        real_logit = discriminator(real_img, training=True)
        # 使用 Discriminator 評估生成圖片
        fake_logit = discriminator(fake_img, training=True)
        # 計算真實圖片和生成圖片的損失
        real_loss, fake_loss = discriminator_loss(real_logit, fake_logit)
        # 計算 Gradient Penalty
        gp_loss = gradient_penalty(partial(discriminator, training=True), real_img, fake_img)
        # 計算 Discriminator 權重
        d_loss = (real_loss + fake_loss) + gp_loss * gradient_penalty_weight
    # 計算梯度
    gradients = tape.gradient(d_loss, discriminator.trainable_variables)
    # 更新 Discriminator 權重
    d_optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
    return real_loss + fake_loss, gp_loss


# 生成100張圖片
def combine_images(images, col=10, row=10):
    # 為了讓生成圖片正常顯示，將圖片從 -1 ~ +1 之間，縮放到 0 ~ 1 之間。
    images = (images + 1) / 2
    # 將 TensorFlow 格式轉換成 Numpy 格式
    images = images.numpy()
    # 取得生成圖片的形狀，shape=(batch_size, height, width, channel)
    b, h, w, _ = images.shape
    # 建立 10 x 10 的大陣列儲存100張圖片
    images_combine = np.zeros(shape=(h * col, w * row, 3))
    # 將100張圖片放入 10 x 10 的大陣列中
    for y in range(col):
        for x in range(row):
            images_combine[y * h: (y + 1) * h, x * w: (x + 1) * w] = images[x + y * row]
    return images_combine


# WGAN-GP訓練程序
def train_wgan():
    # 建立儲存 Generator 模型目錄
    log_dirs = 'logs_wgan'
    model_dir = log_dirs + '/models/'
    os.makedirs(model_dir, exist_ok=True)
    # 建立 TensorBoard 記錄檔
    summary_writer = tf.summary.create_file_writer(log_dirs)
    # 從常態分佈中產生一組固定的Random vector(作為驗證用)
    sample_random_vector = tf.random.normal((100, 1, 1, z_dim))
    # 總共訓練25個epoch
    for epoch in range(25):
        # 讀取訓練資料(真實圖片)
        for step, real_img in enumerate(train_data):
            # 訓練 Discriminator
            d_loss, gp = train_discriminator(real_img)
            # 儲存 Discriminator 的損失值到 TensorBoard 記錄檔
            with summary_writer.as_default():
                tf.summary.scalar('discriminator_loss', d_loss, d_optimizer.iterations)
                tf.summary.scalar('gradient_penalty', gp, d_optimizer.iterations)
            # 每訓練 5 次 Discriminator，執行 1 次 Generator 訓練
            if d_optimizer.iterations.numpy() % n_dis == 0:
                # 訓練 Generator
                g_loss = train_generator()
                # 儲存 Generator 的損失值到 TensorBoard 記錄檔
                with summary_writer.as_default():
                    tf.summary.scalar('generator_loss', g_loss, g_optimizer.iterations)
                # 顯示當前 Generator、Discriminator和gradient penalty的損失值
                print('G Loss: {:.2f}\tD Loss: {:.2f}\tGP LOSS: {:.2f}'.format(g_loss, d_loss, gp))
                # Generator每訓練100次，會產生100張圖片並儲存到TensorBoard
                if g_optimizer.iterations.numpy() % 100 == 0:
                    # 從 Generator 中生成100張圖片
                    x_fake = generator(sample_random_vector, training=False)
                    # 將生成的100張圖片放入10 x 10的大陣列中顯示
                    save_img = combine_images(x_fake)
                    # 儲存100張生成圖片到TensorBoard記錄檔
                    with summary_writer.as_default():
                        tf.summary.image(dataset, [save_img], step=g_optimizer.iterations)
        # 每一次 epoch 儲存一次 Generator 模型權重
        if epoch != 0:
            generator.save_weights(model_dir + "generator-epochs-{}.h5".format(epoch))


if __name__ == '__main__':
    train_wgan()
