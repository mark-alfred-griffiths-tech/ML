import numpy as np
import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras import regularizers
from instantiate_data import *
from log_reg_layer import *

def log_reg_loss(labels, y_hat):
    epsilon = tf.keras.backend.epsilon()
    y_hat_clipped = tf.clip_by_value(y_hat, epsilon, 1 - epsilon)
    y_hat_log = tf.math.log(y_hat_clipped)
    cross_entropy = -tf.reduce_sum(labels * y_hat_log, axis=1)
    loss_f = tf.reduce_mean(cross_entropy)
    return loss_f


def build_l1_rmsprop_model(hp):
    rmsprop_learning_rate = hp.Choice('rmsprop_learning_rate', [1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
    rmsprop_rho = hp.Float(name='rmsprop_rho', min_value=0.1, max_value=0.9, step=0.1)
    rmsprop_momentum = hp.Float(name='rmsprop_momentum', min_value=0.1, max_value=0.9, step=0.1)
    rmsprop_epsilon = hp.Choice('rmsprop_epsilon', [1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
    centred = hp.Boolean('centred')
    kernel_regularizer_l1 = hp.Choice('kernel_regularizer_l1', [1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
    kernel_regularizer_l2 = hp.Choice('kernel_regularizer_l2', [1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
    bias_regularizer_value = hp.Choice('bias_regularizer_value', [1e-3, 1e-4, 1e-5, 1e-6, 1e-7])

    model = call_built_code_l1_rmsprop(bias_regularizer_value, kernel_regularizer_l1, kernel_regularizer_l2,
                                       rmsprop_learning_rate, rmsprop_rho, rmsprop_momentum, rmsprop_epsilon, centred)
    return model


def call_built_code_l1_rmsprop(bias_regularizer_value, kernel_regularizer_l1, kernel_regularizer_l2,
                               rmsprop_learning_rate, rmsprop_rho, rmsprop_momentum, rmsprop_epsilon, centred):
    data = InstantiateData(data_dir='/home/debian/DATA')
    kernel_regularizer = regularizers.l1_l2(l1=kernel_regularizer_l1, l2=kernel_regularizer_l2)
    bias_regularizer = regularizers.l1(bias_regularizer_value)
    num_features = data.xtrain.shape[1]
    num_classes = len(np.unique(data.ytrain))
    images = tf.keras.Input(shape=(num_features,), name="images", dtype=tf.dtypes.int32)
    labels = tf.keras.Input(shape=(1,), name="labels", dtype=tf.dtypes.int32)
    y_hat = LogRegLayer(num_classes, num_features, kernel_regularizer, bias_regularizer)(images, labels)
    log_reg_model = tf.keras.Model(inputs=[images, labels], outputs=y_hat, name="log_reg_model")
    log_reg_model.compile(optimizer=tf.keras.optimizers.RMSprop(
        learning_rate=rmsprop_learning_rate,
        rho=rmsprop_rho,
        momentum=rmsprop_momentum,
        epsilon=rmsprop_epsilon,
        centered=centred,
        name="RMSprop"), loss=[log_reg_loss])
    return log_reg_model

def run_tuner_get_best_hyperparameters_l1_rmsprop(model_dir, project_name, epochs):
    tuner = kt.Hyperband(build_l1_rmsprop_model, objective='accuracy', max_epochs=epochs, factor=3, overwrite=False, directory=model_dir.log_reg_l1_rmsprop_hyperbanding_pretraining,
                         project_name=project_name)

    best_hps = tuner.get_best_hyperparameters(1)[0]

    return best_hps


