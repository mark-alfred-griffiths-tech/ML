##!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
from tensorflow.keras.layers import Dropout, BatchNormalization, InputLayer
from tensorflow.keras.optimizers import SGD
from tensorflow.python.ops.init_ops import glorot_uniform_initializer
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential
from instantiatedata import *
from initialise_settings_add_dim_x import *
from setseed import *
from loadparams import *
from reformatdata import *
from output_class_report_conf_mat import *
from output_summary_stats import *
from loss_accuracy_output import *
from get_name import *
from create_output_directory import *
from run_tuner_get_hyperparameters import *
from set_seed import *
from run_best_hps_hyperparameters import *
from random_fourier_features_layer import *
from create_mlp_output_dir import *


def build_model(hp):
    kernel_initializer = hp.Choice(name="kernel_initializer", values=["gaussian", "laplacian"])
    scale = hp.Float(name="scale", min_value=0.1, max_value=10, step=0.1)
    output_dim = hp.Int(name="output_dim", min_value=10, max_value=1000, step=10)
    optimizer_momentum_float_value = hp.Float("optimizer_momentum_float_value", min_value=0.0, max_value=0.9, step=0.1)
    optimizer_clipnorm_float_value = hp.Float("optimizer_clipnorm_float_value", min_value=0.0, max_value=1.0, step=0.1)

    svm_model = call_existing_code(kernel_initializer, scale, output_dim, optimizer_momentum_float_value,
                                   optimizer_clipnorm_float_value)
    return svm_model


def call_existing_code(kernel_initializer, scale, output_dim, optimizer_momentum_float_value,
                       optimizer_clipnorm_float_value):
    data = InstantiateData(data_dir='')
    data = DimXNumCats(data)
    SetSeed(seed_value=1234)
    mlp_four_dir = CreateMlpFourDirectory()
    best_hps = run_tuner_get_best_hyperparameters(model_dir=mlp_four_dir.mlp_tf_four_layer_pretraining,
                                                  project_name='mlp_tf_four_layer_tensorboard', epochs=10000)
    model = Sequential()
    # ADD INPUT LAYER
    model.add(InputLayer(input_shape=(data.dim_x,)))

    model.add(Dense(best_hps.get('nk_neurons'),
                    kernel_initializer=glorot_uniform_initializer(),
                    bias_initializer='zeros',
                    name='nk_neurons_layer'))
    model.add(Activation(best_hps.get('nk_neurons_activation')))
    model.add(Dropout(best_hps.get('nk_neurons_dropout_value')))

    model.add(Dense(best_hps.get('nl_neurons'),
                    kernel_initializer=glorot_uniform_initializer(),
                    bias_initializer='zeros',
                    name='nl_neurons_layer'))
    model.add(Activation(best_hps.get('nl_neurons_activation')))
    model.add(Dropout(best_hps.get('nl_neurons_dropout_value')))
    model.add(BatchNormalization(momentum=best_hps.get('batch_normalisation_momentum'),
                                 epsilon=best_hps.get('batch_normalisation_epsilon')))

    # ADD RANDOM FOURIER FEATURES LAYER
    model.add(random_fourier_features_layer(output_dim, scale, kernel_initializer, name='quasi_svm_layer'))

    # ADD OUTPUT
    model.add(Dense(units=data.num_cats, name='output_layer'))

    # USE HINGE LOSS IN COMPILE
    model.compile(optimizer=SGD(momentum=optimizer_momentum_float_value,
                                clipnorm=optimizer_clipnorm_float_value),
                  loss=tf.keras.losses.hinge,
                  metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")])

    return model
