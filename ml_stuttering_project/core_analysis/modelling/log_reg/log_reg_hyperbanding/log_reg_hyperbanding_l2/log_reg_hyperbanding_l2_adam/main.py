from tensorflow.keras import callbacks
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
import keras_tuner as kt
from instantiate_data import *
from initialise_settings_and_dim_x import *
from standard_scaler import *
from set_seed import *
from create_output_dir import *
import numpy as np
from kerastuner_tensorboard_logger import (
    TensorBoardLogger,
    setup_tb)


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

def wire_model(kernel_regularizer, bias_regularizer):
    data = InstantiateData(data_dir = '/scratch/users/k1754828/DATA/')
    num_features = data.xtrain.shape[1]
    num_classes = len(np.unique(data.ytrain))
    images = tf.keras.Input(shape=(num_features,), name="images", dtype=tf.dtypes.int32)
    labels = tf.keras.Input(shape=(1,), name="labels", dtype=tf.dtypes.int32)
    y_hat = LogRegLayer(num_classes, num_features, kernel_regularizer, bias_regularizer)(images, labels)
    log_reg_model = tf.keras.Model(inputs=[images, labels], outputs=y_hat, name="log_reg_model")
    return log_reg_model

def log_reg_loss(labels, y_hat):
    epsilon = tf.keras.backend.epsilon()
    y_hat_clipped = tf.clip_by_value(y_hat, epsilon, 1 - epsilon)
    y_hat_log = tf.math.log(y_hat_clipped)
    cross_entropy = -tf.reduce_sum(labels * y_hat_log, axis=1)
    loss_f = tf.reduce_mean(cross_entropy)
    return loss_f

def build_model_l2_adam(hp):
    adam_learning_rate = hp.Choice('adam_learning_rate', [1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
    adam_beta_1 = hp.Float(name='adam_beta_1', min_value=0.1, max_value=0.9, step=0.1)
    adam_beta_2 = hp.Float(name='adam_beta_2', min_value=0.001, max_value=0.999, step=0.001)
    adam_epsilon = hp.Choice('epsilon', [1e-05, 1e-06, 1e-07])
    adam_amsgrad = hp.Boolean('adam_amsgrad')
    kernel_regularizer_l1 = hp.Choice('kernel_regularizer_l1', [1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
    kernel_regularizer_l2 = hp.Choice('kernel_regularizer_l2', [1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
    bias_regularizer_value = hp.Choice('bias_regularizer_value', [1e-3, 1e-4, 1e-5, 1e-6, 1e-7])

    model = call_built_code_l2_adam(bias_regularizer_value, kernel_regularizer_l1, kernel_regularizer_l2,
                                    adam_learning_rate, adam_beta_1, adam_beta_2, adam_epsilon, adam_amsgrad)
    return model


def call_built_code_l2_adam(bias_regularizer_value, kernel_regularizer_l1, kernel_regularizer_l2, adam_learning_rate,
                            adam_beta_1, adam_beta_2, adam_epsilon, adam_amsgrad):
    data = InstantiateData(data_dir='/scratch/users/k1754828/DATA/')
    kernel_regularizer = regularizers.l1_l2(l1=kernel_regularizer_l1, l2=kernel_regularizer_l2)
    bias_regularizer = regularizers.l2(bias_regularizer_value)
    num_features = data.xtrain.shape[1]
    num_classes = len(np.unique(data.ytrain))
    images = tf.keras.Input(shape=(num_features,), name="images", dtype=tf.dtypes.int32)
    labels = tf.keras.Input(shape=(1,), name="labels", dtype=tf.dtypes.int32)
    y_hat = LogRegLayer(num_classes, num_features, kernel_regularizer, bias_regularizer)(images, labels)
    log_reg_model = tf.keras.Model(inputs=[images, labels], outputs=y_hat, name="log_reg_model")
    log_reg_model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=adam_learning_rate,
        beta_1=adam_beta_1,
        beta_2=adam_beta_2,
        epsilon=adam_epsilon,
        amsgrad=adam_amsgrad,
        name="Adam"), loss=[log_reg_loss])
    return log_reg_model


class run_tuner_get_best_hyperparameters:
    def __init__(self, data, max_epochs, log_reg_dir, min_delta, *args, **kwargs):
        super(run_tuner_get_best_hyperparameters, self).__init__(*args, **kwargs)
        self.max_epochs = max_epochs
        self.log_reg_l2_adam_hyperbanding_pretraining = log_reg_dir.log_reg_l2_adam_hyperbanding_pretraining
        self.log_reg_l2_adam_hyperbanding_tensorboard = log_reg_dir.log_reg_l2_adam_hyperbanding_tensorboard
        self.log_reg_l2_adam_hyperbanding_partial_models = log_reg_dir.log_reg_l2_adam_hyperbanding_partial_models
        self.min_delta = min_delta
        self.data = data
        self.training_data = {
            "images": np.array(self.data.xtrain),
            "labels": self.data.ytrain
            ,
        }
        self.validation_data = {
            "images": np.array(self.data.xtest),
            "labels": self.data.ytest
            ,
        }
        self.run_tuner()

    def run_tuner(self):
        self.tuner = kt.Hyperband(build_model_l2_adam,
                                  objective=kt.Objective('val_lr_accuracy',direction='max'),
                                  max_epochs=self.max_epochs,
                                  factor=3,
                                  # distribution_strategy=tf.distribute.MirroredStrategy(),
                                  overwrite=False,
                                  directory=self.log_reg_l2_adam_hyperbanding_pretraining,
                                  project_name='log_reg_tf_l2_adam_hyperbanding_tensorboard',
                                  logger=TensorBoardLogger(metrics=["loss","accuracy", "val_accuracy", "val_loss",], logdir=self.log_reg_l2_adam_hyperbanding_pretraining+"/log_reg_tf_l2_adam_hyperbanding_tensorboard/hparams"))
        setup_tb(self.tuner)
        tensorflow_board = tf.keras.callbacks.TensorBoard(self.log_reg_l2_adam_hyperbanding_tensorboard)
        partial_models = tf.keras.callbacks.ModelCheckpoint(filepath=self.log_reg_l2_adam_hyperbanding_partial_models +
                                                                     '/l1_adam_model.{epoch:02d}.h5')
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', min_delta=self.min_delta,
                                                      patience=5)
        self.tuner.search(self.training_data, validation_data=(self.validation_data),
                          callbacks=[stop_early, partial_models, tensorflow_board])
        return self


# RUN_ALL

data_dir = '/scratch/users/k1754828/DATA/'
results_dir = '/scratch/users/k1754828/RESULTS/'
log_reg_dir = CreateOutputDirectory(results_dir)
data = InstantiateData(data_dir)
data = DimXNumCats(data)
data = ConductSklearnStandardScaling(data)
min_delta = 0.0001
batch_size = 14000
max_epochs = 10000
run_tuner_get_best_hyperparameters(data, max_epochs, log_reg_dir, min_delta)()
