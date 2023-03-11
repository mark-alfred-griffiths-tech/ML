import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from set_seed import *

class LogRegLayer(tf.keras.layers.Layer):
    def __init__(self, units, num_features, bias_regularizer, kernel_regularizer,  *args, **kwargs):
        super(LogRegLayer, self).__init__(*args, **kwargs)
        self.units = int(units)
        self.num_features = int(num_features)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        SetSeed(1234)

    def log_reg_accuracy(self, labels, y_hat):
        correct_prediction = tf.equal(tf.argmax(labels, 1), tf.cast(y_hat, tf.int64))
        return K.mean(tf.cast(correct_prediction, tf.float32))

    def log_reg_loss(self, labels, y_hat):
        epsilon = tf.keras.backend.epsilon()
        y_hat_clipped = tf.clip_by_value(y_hat, epsilon, 1 - epsilon)
        y_hat_log = tf.math.log(y_hat_clipped)
        cross_entropy = -tf.reduce_sum(labels * y_hat_log, axis=1)
        loss_f = tf.reduce_mean(cross_entropy)
        return loss_f

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[1], self.units),
            regularizer=self.kernel_regularizer,
            initializer="random_normal",
            trainable=True,
            name="w",
            dtype=tf.dtypes.float32,
        )
        self.b = self.add_weight(
            shape=(self.units,), regularizer=self.bias_regularizer ,initializer="random_normal", trainable=True, name="b", dtype=tf.dtypes.float32,
        )

    def call(self, images, labels):
        labels = tf.one_hot(labels, self.units)
        y_hat = tf.nn.softmax(tf.add(tf.matmul(tf.cast(images, dtype=tf.dtypes.float32), tf.cast(self.w, dtype=tf.dtypes.float32)), tf.cast(self.b, dtype=tf.dtypes.float32)))
        loss = self.log_reg_loss(tf.cast(labels, dtype=tf.dtypes.float32), tf.cast(y_hat, dtype=tf.dtypes.float32))
        self.add_loss(loss)
        acc = self.log_reg_accuracy(tf.cast(labels, dtype=tf.dtypes.int32), tf.cast(y_hat, dtype=tf.dtypes.int32))
        self.add_metric(acc, name='lr_accuracy')
        return y_hat

    def get_config(self):
        config = super(LogRegLayer, self).get_config()
        config.update({"units": self.units})
        config.update({"num_features": self.num_features})
        config.update({"kernel_regularizer": self.kernel_regularizer})
        config.update({"bias_regularizer": self.bias_regularizer})
        config.update({"activity_regularizer": self.activity_regularizer})
        return config


def log_reg_loss(labels, y_hat):
    epsilon = tf.keras.backend.epsilon()
    y_hat_clipped = tf.clip_by_value(y_hat, epsilon, 1 - epsilon)
    y_hat_log = tf.math.log(y_hat_clipped)
    cross_entropy = -tf.reduce_sum(labels * y_hat_log, axis=1)
    loss_f = tf.reduce_mean(cross_entropy)
    return loss_f
