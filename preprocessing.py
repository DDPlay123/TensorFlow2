import tensorflow as tf

# 水平翻轉
def flip(x):
    x = tf.image.random_flip_left_right(x)
    return x

# 顏色轉換
def color(x):
    x = tf.image.random_hue(x, 0.08) # 色調
    x = tf.image.random_saturation(x, 0.6, 1.6) # 飽和度
    x = tf.image.random_brightness(x, 0.05) # 亮度
    x = tf.image.random_contrast(x, 0.7, 1.3) # 對比度
    return x

# 影像旋轉
def rotate(x):
    # 隨機旋轉 n 次，minval和maxval設定 n 的範圍，每次旋轉90度
    x = tf.image.rot90(x, tf.random.uniform(shape=[], minval=1, maxval=4, dtype=tf.int32))
    return x

# 影像縮放
def zoom(x, scale_min=0.6, scale_max=1.4):
    h, w, c = x.shape
    # 隨機縮放比例
    scale = tf.random.uniform([], scale_min, scale_max)
    sh = h * scale # 縮放後的影像高度
    sw = w * scale # 縮放後的影像寬度
    # 縮放影像
    x = tf.image.resize(x, (sh, sw))
    # 影像縮減或填補
    x = tf.image.resize_with_crop_or_pad(x, h ,w)
    return x

# 資料預處理
def parse_aug_fn(dataset):
    # 影像標準化
    x = tf.cast(dataset['image'], tf.float32) / 255.
    # 資料增強
    x = flip(x) # 水平翻轉
    x = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: color(x), lambda:x) # 50%機率顏色轉換
    x = tf.cond(tf.random.uniform([], 0, 1) > 0.75, lambda: rotate(x), lambda:x) # 25%機率影像旋轉
    x = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: zoom(x), lambda:x) # 50%機率影像縮放
    # 標籤轉One-hot Encoding
    y = tf.one_hot(dataset['label'], 10)
    return x,y

def parse_fn(dataset):
    # 影像標準化
    x = tf.cast(dataset['image'], tf.float32) / 255.
    # 標籤轉One-hot Encoding
    y = tf.one_hot(dataset['label'], 10)
    return x,y