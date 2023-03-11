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
from recover_decoder import full_autoencoder
from reformat_data import *
from standard_scaler import *
import tensorflow as tf

class RunTuneGetBestFullAutoencoderHyperparameters:
    def __init__(self, max_epochs, min_delta, batch_size, seed_value, *args, **kwargs):
        super(RunTuneGetBestFullAutoencoderHyperparameters, self).__init__(*args, **kwargs)
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
        self.tuner = kt.Hyperband(full_autoencoder,
                                  objective=kt.Objective('val_accuracy', direction='max'),
                                  max_epochs=self.max_epochs,
                                  factor=3,
                                  # distribution_strategy=tf.distribute.MirroredStrategy(),
                                  overwrite=False,
                                  directory=self.con_vae_dir_tf_pretraining,
                                  project_name='full_autoencoder_tensorboard',
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
