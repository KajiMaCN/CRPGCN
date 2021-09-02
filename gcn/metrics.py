import tensorflow.compat.v1 as tf


def masked_accuracy(preds, labels, mask, negative_mask):
    """Accuracy with masking."""
    preds = tf.cast(preds, tf.float32)# 转换数据类型
    labels = tf.cast(labels, tf.float32)
    error = tf.square(preds-labels)#是对括号里的每一个元素求平方
    mask += negative_mask
    mask = tf.cast(mask, dtype=tf.float32) 
    error *= mask
    return tf.sqrt(tf.reduce_mean(error))#先算error平均值，再开根号

def euclidean_loss(preds, labels):
    euclidean_loss = tf.sqrt(tf.reduce_sum(tf.square(preds-labels),0))
    return euclidean_loss
