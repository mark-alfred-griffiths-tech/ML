#!/usr/bin/env python
# coding: utf-8
from kerastuner_tensorboard_logger import (
    TensorBoardLogger,
    setup_tb
)
from instantiate_data import *
from add_dim_x_num_cats import *
from create_con_vae_output_directory import *
from set_seed import *
import keras_tuner as kt
from tensorflow.python.ops.init_ops import glorot_uniform_initializer
from reformat_data import *
from tensorflow.keras.optimizers import SGD
from standard_scaler import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
import os
import tensorflow as tf

class instantiate_data:
    def __init__(self, *args, **kwargs):
        super(instantiate_data, self).__init__(*args, **kwargs)
        data_dir = '/scratch/users/k1754828/DATA'
        self.set_data_root(data_dir)
        xytrain, xytest = self.get_files()
        self.split_files(xytrain, xytest)

    def set_data_root(self, data_dir):
        self.data_dir = Path(data_dir)
        return self

    def get_files(self):
        os.chdir(self.data_dir)
        xytrain = pd.read_csv('train.csv')
        xytest = pd.read_csv('test.csv')
        return xytrain, xytest

    def split_files(self, xytrain, xytest):
        self.ytrain = xytrain.loc[:, xytrain.columns == 'stutter']
        self.xtrain = xytrain.loc[:, xytrain.columns != 'stutter']
        self.ytest = xytest.loc[:, xytest.columns == 'stutter']
        self.xtest = xytest.loc[:, xytest.columns != 'stutter']
        return self


class get_dim_x:
    def __init__(self, data, *args, **kwargs):
        super(get_dim_x, self).__init__(*args, **kwargs)
        self.xtrain = data.xtrain
        self.ytrain = data.ytrain
        self.xtest = data.xtest
        self.ytest = data.ytest
        self.dim_x = pd.DataFrame(data.xtrain).shape[1]


def build_decoder_model(hp):
    ni_neurons_num = hp.Int(name="ni_neurons", min_value=32, max_value=128, step=32)
    nj_neurons_num = hp.Int(name="nj_neurons", min_value=32, max_value=128, step=32)
    nk_neurons_num = hp.Int(name="nk_neurons", min_value=32, max_value=128, step=32)
    nl_neurons_num = hp.Int(name="nl_neurons", min_value=32, max_value=128, step=32)

    ni_neurons_activation = hp.Choice(name="ni_neurons_activation", values=["relu", "elu", "tanh"])
    nj_neurons_activation = hp.Choice(name="nj_neurons_activation", values=["relu", "elu", "tanh"])
    nk_neurons_activation = hp.Choice(name="nk_neurons_activation", values=["relu", "elu", "tanh"])
    nl_neurons_activation = hp.Choice(name="nl_neurons_activation", values=["relu", "elu", "tanh"])
    nm_neurons_activation = hp.Choice(name="nm_neurons_activation", values=["sigmoid"])

    optimizer_momentum_float_value = hp.Float("optimizer_momentum_float_value", min_value=0.0, max_value=0.9, step=0.1)
    optimizer_clipnorm_float_value = hp.Float("optimizer_clipnorm_float_value", min_value=0.0, max_value=1.0, step=0.1)

    model = decoder_base(ni_neurons_num, nj_neurons_num, nk_neurons_num, nl_neurons_num,
                            ni_neurons_activation, nj_neurons_activation,
                            nk_neurons_activation, nl_neurons_activation, nm_neurons_activation,
                            optimizer_momentum_float_value, optimizer_clipnorm_float_value)
    return model


def decoder_base(ni_neurons_num, nj_neurons_num, nk_neurons_num, nl_neurons_num,
                    ni_neurons_activation, nj_neurons_activation, nk_neurons_activation,
                    nl_neurons_activation, nm_neurons_activation, optimizer_momentum_float_value,
                    optimizer_clipnorm_float_value):
    data = instantiate_data()
    data = get_dim_x(data)
    SetSeed(set_seed=1234)

    model = Sequential()
    model.add(Dense(ni_neurons_num, activation=ni_neurons_activation,
                    kernel_initializer=glorot_uniform_initializer(),
                    bias_initializer='zeros',
                    input_shape=(data.dim_x,),
                    name="ni_neurons"))
    model.add(Dense(nj_neurons_num, activation=nj_neurons_activation,
                    kernel_initializer=glorot_uniform_initializer(),
                    bias_initializer='zeros',
                    name="nj_neurons"))
    model.add(Dense(nk_neurons_num,
                    kernel_initializer=glorot_uniform_initializer(),
                    bias_initializer='zeros',
                    activation=nk_neurons_activation,
                    name="nk_neurons"))
    model.add(Dense(nl_neurons_num,
                    kernel_initializer=glorot_uniform_initializer(),
                    bias_initializer='zeros',
                    activation=nl_neurons_activation,
                    name="nl_neurons"))
    model.add(Dense(data.dim_x, activation=nm_neurons_activation, name="nm_neurons"))
    model.compile(loss="categorical_crossentropy",
                  optimizer=SGD(momentum=optimizer_momentum_float_value, clipnorm=optimizer_clipnorm_float_value))

    return model


class RunTuneGetBestDecoderHyperparameters:
    def __init__(self, max_epochs, min_delta, batch_size, seed_value, *args, **kwargs):
        super(RunTuneGetBestDecoderHyperparameters, self).__init__(*args, **kwargs)
        self.max_epochs = max_epochs
        self.min_delta = min_delta
        self.seed_value = seed_value
        self.batch_size = batch_size

        data = InstantiateData(data_dir='/scratch/users/k1754828/DATA/')
        data = DimXNumCats(data)
        data = ConductSklearnStandardScaling(data)
        data = ReformatData(data, batch_size=self.batch_size)

        con_vae_dir = CreateConVAEDirectory(results_dir='/scratch/users/k1754828/RESULTS/')

        SetSeed(seed_value=self.seed_value)

        self.xytrain = data.xytrain
        self.xytest = data.xytest

        self.con_vae_dir_tf = con_vae_dir.con_vae_dir_tf_tensorboard
        self.con_vae_dir_tf_pretraining = con_vae_dir.con_vae_tf_pretraining
        self.con_vae_dir_tf_partial_models = con_vae_dir.con_vae_tf_partial_models
        self.con_vae_dir_tf_tensorboard = con_vae_dir.con_vae_tf_tensorboard
        self.run_tuner()

    def run_tuner(self):
        self.tuner = kt.Hyperband(build_decoder_model,
                                  objective=kt.Objective('val_accuracy', direction='max'),
                                  max_epochs=self.max_epochs,
                                  factor=3,
                                  # distribution_strategy=tf.distribute.MirroredStrategy(),
                                  overwrite=False,
                                  directory=self.con_vae_dir_tf_pretraining,
                                  project_name='concrete_autoencoder_decoder_pretrain',
                                  logger=TensorBoardLogger(metrics=["loss", "accuracy", "val_accuracy", "val_loss", ],
                                                           logdir=self.con_vae_dir_tf_tensorboard + "/con_vae_dir_tf_tensorboard/hparams")
                                  )
        setup_tb(self.tuner)
        tensorflow_board = tf.keras.callbacks.TensorBoard(self.con_vae_dir_tf_tensorboard)
        partial_models = tf.keras.callbacks.ModelCheckpoint(filepath=self.con_vae_dir_tf_partial_models +
                                                                     '/model.{epoch:02d}.h5')
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', min_delta=self.min_delta,
                                                      patience=5)
        self.tuner.search(self.xytrain, validation_data=self.xytest, batch_size=self.batch_size,
                          callbacks=[stop_early, partial_models, tensorflow_board,
                                     tf.keras.callbacks.ReduceLROnPlateau(patience=4),
                                     tf.keras.callbacks.EarlyStopping(patience=8)])
        return self


