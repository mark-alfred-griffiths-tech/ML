import os
from pathlib import Path
from tensorflow.keras.layers import Activation, Dense, Dropout, BatchNormalization, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
import keras_tuner as kt
from tensorflow.keras import backend as K
from tensorflow.python.ops.init_ops import glorot_uniform_initializer
from instantiate_data import *
from dim_x_add_num_cats import *
from set_seed import *
from instantiate_data import *
from dim_x_add_num_cats import *
from standard_scaler import *
from reformat_data import *


def build_model(hp):

    nl_neurons = hp.Int(name='nl_neurons', min_value=32, max_value=128, step=32)

    nl_neurons_activation = hp.Choice(name='nl_neurons_activation', values=['relu', 'tanh'])

    nl_neurons_dropout_value = hp.Float('nl_neurons_dropout_value', min_value=0.1, max_value=0.8, step=0.1)

    nl_neurons_batch_normalisation_momentum = hp.Float(name='nl_neurons_batch_normalisation_momentum', min_value=0.1,
                                                       max_value=0.9, step=0.1)
    nl_neurons_batch_normalisation_epsilon = hp.Choice(name='nl_neurons_batch_normalisation_epsilon',
                                                       values=[1e-3, 1e-4])

    optimizer_momentum_float_value = hp.Float('optimizer_momentum_float_value', min_value=0.0, max_value=0.9, step=0.1)
    optimizer_clipnorm_float_value = hp.Float('optimizer_clipnorm_float_value', min_value=0.0, max_value=1.0, step=0.1)

    model = call_existing_code(nl_neurons, nl_neurons_activation, nl_neurons_dropout_value,
                               nl_neurons_batch_normalisation_momentum,
                               nl_neurons_batch_normalisation_epsilon,
                               optimizer_momentum_float_value, optimizer_clipnorm_float_value)

    return model


def call_existing_code(nl_neurons, nl_neurons_activation,
                       nl_neurons_dropout_value,
                       nl_neurons_batch_normalisation_momentum,
                       nl_neurons_batch_normalisation_epsilon,
                       optimizer_momentum_float_value, optimizer_clipnorm_float_value):
    seed_value = 1234

    data = InstantiateData(data_dir='/home/debian/DATA/')
    data = DimXNumCats(data)
    data = ConductSklearnStandardScaling(data)
    data = ReformatData(data, batch_size=14000)

    model = Sequential()

    model.add(InputLayer(input_shape=(data.dim_x,)))

    model.add(Dense(nl_neurons,
                    kernel_initializer=glorot_uniform_initializer(),
                    bias_initializer='zeros',
                    name='nl_neurons_layer'))
    model.add(Activation(nl_neurons_activation))

    model.add(Dropout(nl_neurons_dropout_value))

    model.add(BatchNormalization(momentum=nl_neurons_batch_normalisation_momentum,
                                     epsilon=nl_neurons_batch_normalisation_epsilon))

    model.add(Dense(data.num_cats,
                    kernel_initializer=glorot_uniform_initializer(),
                    name='output_layer'))
    model.add(Activation('softmax'))

    model.compile(optimizer=SGD(momentum=optimizer_momentum_float_value,
                                clipnorm=optimizer_clipnorm_float_value), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def run_tuner_get_best_hyperparameters(model_dir, project_name, epochs):
    tuner = kt.Hyperband(build_model, objective='accuracy', max_epochs=epochs, factor=3, directory=model_dir,
                         project_name=project_name)

    best_hps = tuner.get_best_hyperparameters(1)[0]

    return best_hps
