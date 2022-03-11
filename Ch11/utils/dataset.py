"""
「資料預處理」的程式碼
"""
import tensorflow as tf


def parse_fn(dataset, input_size=(64, 64)):
    """
    Data Prepossessing:
        將圖片大小調整到 64 x 64，每個像素由 0 ~ 255，正規化到 -1 ~ +1 之間。
    """
    x = tf.cast(dataset['image'], tf.float32)
    crop_size = 300  # celeb_a: 108、aflw2k3d: 300
    # 圖片大小celeb_a:(218, 178, 3)、aflw2k3d:(450, 450, 3)
    h, w, _ = x.shape
    # 從圖片中間裁切(108, 108, 3)大小作為新的圖片
    x = tf.image.crop_to_bounding_box(x, (h-crop_size)//2, (w-crop_size)//2, crop_size, crop_size)
    # 將圖片Resize: (108, 108, 3) --> (64, 64, 3)
    x = tf.image.resize(x, input_size)
    # 將圖片像素值標準化到 -1 ~ +1 之間，步驟: [0, 255]/127.5 --> [0, 2], [0, 2]-1 --> [-1, 1]
    x = x / 127.5 - 1
    return x
