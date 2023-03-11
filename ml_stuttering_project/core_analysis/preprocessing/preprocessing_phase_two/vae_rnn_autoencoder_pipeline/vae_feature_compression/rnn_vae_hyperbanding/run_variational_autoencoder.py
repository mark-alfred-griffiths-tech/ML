# !/usr/bin/env python
# coding: utf-8
import tensorflow as tf
from create_output_rnn_four_directory import *
from instantiate_data import *
from add_dim_x_num_cats import *
from standard_scaler import *
from reformat_data import *
from variational_autoencoder import *
from datetime import datetime
import sys
import keras_tuner as kt
from kerastuner_tensorboard_logger import (
    TensorBoardLogger,
    setup_tb
)



class RunTuneGetBestRnnHyperparametersFour:
    def __init__(self, data, epochs, min_delta, batch_size, seed_value, *args, **kwargs):
        super(RunTuneGetBestRnnHyperparametersFour, self).__init__(*args, **kwargs)
        self.data = data
        self.epochs = epochs
        self.min_delta = min_delta
        self.batch_size = batch_size
        self.seed_value = seed_value

        data = InstantiateData(data_dir='/scratch/users/k1754828/DATA/')
        data = DimXNumCats(data)
        data = ConductSklearnStandardScaling(data)
        self.data = ReformatData(data, batch_size=self.batch_size)

        vae_dir = CreateVaeDirectory(results_dir='/scratch/users/k1754828/RESULTS/')

        self.xfull = self.data.xfull

        self.vae_tf = vae_dir.vae_tf
        self.vae_tf_pretraining = vae_dir.vae_tf_pretraining
        self.vae_tf_partial_models = vae_dir.vae_tf_partial_models
        self.vae_tf_tensorboard = vae_dir.vae_tf_tensorboard
        self.run_tuner()

    def run_tuner(self):
        self.tuner = kt.Hyperband(VAE(data=self.data, epochs=self.epochs, min_delta=self.min_delta,
                                            batch_size=self.batch_size, seed_value=self.seed_value),
                                  objective=kt.Objective('val_accuracy', direction='max'),
                                  max_epochs=self.epochs,
                                  factor=3,
                                  # distribution_strategy=tf.distribute.MirroredStrategy(),
                                  overwrite=False,
                                  directory=self.vae_tf_pretraining,
                                  project_name='vae_tf_tensorboard',
                                  logger=TensorBoardLogger(metrics=["loss", "accuracy", "val_accuracy", "val_loss", ],
                                                           logdir=self.vae_tf_pretraining + "/vae_tf_tensorboard/hparams")
                                  )
        setup_tb(self.tuner)
        tensorflow_board = tf.keras.callbacks.TensorBoard(self.vae_tf_tensorboard)
        partial_models = tf.keras.callbacks.ModelCheckpoint(filepath=self.vae_tf_partial_models +
                                                                     '/model.{epoch:02d}.h5')
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', min_delta=self.min_delta,
                                                      patience=5)
        self.tuner.search(self.xfull, batch_size=self.batch_size,
                          callbacks=[stop_early, partial_models, tensorflow_board,
                                     tf.keras.callbacks.ReduceLROnPlateau(patience=4),
                                     tf.keras.callbacks.EarlyStopping(patience=8)])
        return self


def run_rnn_four():
    start_time = datetime.now()

    epochs = 10000
    min_delta = 0.001
    batch_size = 14000
    seed_value = 1234

    RunTuneGetBestRnnHyperparametersFour(max_epochs=epochs, min_delta=min_delta, batch_size=batch_size,
                                         seed_value=seed_value)

    time_delta = datetime.now() - start_time
    exit_message = 'VAE RAN SUCCESSFULLY IN: ' + str(time_delta)
    sys.exit(exit_message)


# RUN ALL
run_rnn_four()
