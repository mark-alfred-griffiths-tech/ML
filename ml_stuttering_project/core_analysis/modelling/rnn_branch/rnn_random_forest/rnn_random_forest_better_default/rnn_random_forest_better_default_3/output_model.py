#!/usr/bin/env python
# coding: utf-8
import os
import tensorflow as tf


class OutputModel:
    def __init__(self, data, random_forest_dir, df_and_nn_model, name, *args, **kwargs):
        super(OutputModel, self).__init__(*args, **kwargs)
        self.rf_nn_phase_three_models = random_forest_dir.random_forest_phase_two_models
        self.df_and_nn_model = df_and_nn_model
        input = tf.keras.layers.InputLayer(data.dim_x)
        self.df_and_nn_model = df_and_nn_model(input)
        self.name = name
        self.save_model()

    def save_model(self):
        os.chdir(self.rf_tf_rank_one_three_layer_final_model)
        self.df_and_nn_model.save(str(str(self.name) + '.model'))
        return self


def compile_model(model):
    model.compile(metrics=['accuracy'])
    return model
