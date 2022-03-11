"""
「自定義損失函數」的程式碼
"""
import tensorflow as tf


def generator_loss(fake_logit):
    """
    Generator Loss損失函數 :
        讓 Generator 生成的圖片，使 Discriminator 辨識為接近真實的圖片。
    """
    g_loss = - tf.reduce_mean(fake_logit)
    return g_loss


def discriminator_loss(real_logit, fake_logit):
    """
    Discriminator Loss損失函數 :
        降低real_loss項，可以讓 Discriminator 判斷真實圖片越真。
        降低fake_loss項，可以讓 Discriminator 判斷生成圖片越假。
    """
    real_loss = -tf.reduce_mean(real_logit)
    fake_loss = tf.reduce_mean(fake_logit)
    return real_loss, fake_loss


def gradient_penalty(discriminator, real_img, fake_img):
    """
    Gradient Penalty方法:
        可以使 Discriminator 滿足 1-Lipschitz function。
    """
    def _interpolate(a, b):
        shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
        alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
        inter = (alpha * a) + ((1 - alpha) * b)
        inter.set_shape(a.shape)
        return inter
    # 將產生的生成圖片與真實圖片做線性內插取得 x'
    x_img = _interpolate(real_img, fake_img)
    with tf.GradientTape() as tape:
        # 確保 x_img 可以被 tape 追蹤
        tape.watch(x_img)
        # Discriminator 會判別 x_img 的真假
        pred_logit = discriminator(x_img)
    # 計算梯度
    grad = tape.gradient(pred_logit, x_img)
    # 計算梯度的範數
    norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
    # L2 正規化，希望損失值越接近 1 越好
    gp_loss = tf.reduce_mean((norm - 1.)**2)
    return gp_loss
