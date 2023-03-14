#!/usr/bin/env python
# coding: utf-8
from kerastuner import HyperModel
from tensorflow.keras.optimizers import SGD
from tensorflow.python.ops.init_ops import glorot_uniform_initializer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class HyperDecoder(HyperModel):
    def __init__(self, data, *args, **kwargs):
        super(HyperDecoder, self).__init__(*args, **kwargs)
        self.decoder = None
        self.data = data
    def build(self,hp):
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

        decoder = Sequential()
        decoder.add(Dense(ni_neurons_num, activation=ni_neurons_activation,
                        kernel_initializer=glorot_uniform_initializer(),
                        bias_initializer='zeros',
                        input_shape=(self.data.dim_x,),
                        name="ni_neurons"))
        decoder.add(Dense(nj_neurons_num, activation=nj_neurons_activation,
                        kernel_initializer=glorot_uniform_initializer(),
                        bias_initializer='zeros',
                        name="nj_neurons"))
        decoder.add(Dense(nk_neurons_num,
                        kernel_initializer=glorot_uniform_initializer(),
                        bias_initializer='zeros',
                        activation=nk_neurons_activation,
                        name="nk_neurons"))
        decoder.add(Dense(nl_neurons_num,
                        kernel_initializer=glorot_uniform_initializer(),
                        bias_initializer='zeros',
                        activation=nl_neurons_activation,
                        name="nl_neurons"))
        decoder.add(Dense(self.data.dim_x, activation=nm_neurons_activation, name="nm_neurons"))
        decoder.compile(loss="categorical_crossentropy",
                      optimizer=SGD(momentum=optimizer_momentum_float_value, clipnorm=optimizer_clipnorm_float_value))
        return decoder
