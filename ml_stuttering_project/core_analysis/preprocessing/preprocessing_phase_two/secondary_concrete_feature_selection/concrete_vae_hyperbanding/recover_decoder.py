#!/usr/bin/env python
# coding: utf-8
from instantiate_data import InstantiateData
from add_dim_x_num_cats import DimXNumCats
import keras_tuner as kt
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from concrete_autoencoder import ConcreteAutoencoderFeatureSelector
from decoder_pretraining import decoder_base


def get_best_concrete_autoencoder_decoder_pretrain_hyperparameters(model_dir):
    tuner = kt.Hyperband(decoder_base, objective='accuracy', max_epochs=1000, factor=3, directory=model_dir,
                         project_name='concrete_autoencoder_decoder_pretrain')

    best_hps = tuner.get_best_hyperparameters(1)[0]

    return best_hps


def full_autoencoder(num_feats, optimizer_momentum_float_value, optimizer_clipnorm_float_value):
    data = InstantiateData()
    data = DimXNumCats(data)
    best_hps = get_best_concrete_autoencoder_decoder_pretrain_hyperparameters(
        model_dir='/users/k1754828/RESULTS/concrete_autoencoder/')

    decoder = Sequential()

    decoder.add(Dense(data.dim_x, activation=best_hps.get('ni_neurons_activation'), name='ni_neurons'))
    decoder.add(
        Dense(best_hps.get('nj_neurons'), activation=best_hps.get('nj_neurons_activation'), name='nj_neurons_layer'))
    decoder.add(
        Dense(best_hps.get('nk_neurons'), activation=best_hps.get('nk_neurons_activation'), name='nk_neurons_layer'))
    decoder.add(
        Dense(best_hps.get('nl_neurons'), activation=best_hps.get('nl_neurons_activation'), name='nl_neurons_layer'))
    decoder.add(Dense(data.dim_x, activation=best_hps.get('nm_neurons_activation'), name='nm_neurons_layer'))

    autoencoder = Sequential()

    autoencoder.add(ConcreteAutoencoderFeatureSelector(K=num_feats, output_function=decoder, num_epochs=50))
    autoencoder.compile(loss="categorical_crossentropy",
                        optimizer=SGD(momentum=optimizer_momentum_float_value, clipnorm=optimizer_clipnorm_float_value))

    return autoencoder


def build_full_autoencoder_model(hp):
    num_feats = hp.Int(name="num_feats", min_value=1, max_value=55, step=1)
    optimizer_momentum_float_value = hp.Float("optimizer_momentum_float_value", min_value=0.0, max_value=0.9, step=0.1)
    optimizer_clipnorm_float_value = hp.Float("optimizer_clipnorm_float_value", min_value=0.0, max_value=1.0, step=0.1)

    autoencoder = full_autoencoder(num_feats, optimizer_momentum_float_value, optimizer_clipnorm_float_value)
    return autoencoder