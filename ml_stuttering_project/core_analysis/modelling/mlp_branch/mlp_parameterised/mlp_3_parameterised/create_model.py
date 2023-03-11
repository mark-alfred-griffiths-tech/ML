from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Activation, Dense, Dropout, BatchNormalization, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.python.ops.init_ops import glorot_uniform_initializer
from set_seed import *
from instantiate_data import *
from add_dim_x_num_cats import *
from standard_scaler import *
from reformat_data import *


def build_model(hp):
    nj_neurons = hp.Int(name='nj_neurons', min_value=32, max_value=128, step=32)
    nk_neurons = hp.Int(name='nk_neurons', min_value=32, max_value=128, step=32)
    nl_neurons = hp.Int(name='nl_neurons', min_value=32, max_value=128, step=32)

    nj_neurons_activation = hp.Choice(name='ni_neurons_activation', values=['relu', 'tanh'])
    nk_neurons_activation = hp.Choice(name='nk_neurons_activation', values=['relu', 'tanh'])
    nl_neurons_activation = hp.Choice(name='nl_neurons_activation', values=['relu', 'tanh'])

    nj_neurons_dropout_value = hp.Float('nj_neurons_dropout_value', min_value=0.1, max_value=0.8, step=0.1)
    nk_neurons_dropout_value = hp.Float('nk_neurons_dropout_value', min_value=0.1, max_value=0.8, step=0.1)
    nl_neurons_dropout_value = hp.Float('nl_neurons_dropout_value', min_value=0.1, max_value=0.8, step=0.1)

    nj_neurons_batch_normalisation_momentum = hp.Float(name='nj_neurons_batch_normalisation_momentum', min_value=0.1,
                                                       max_value=0.9, step=0.1)
    nj_neurons_batch_normalisation_epsilon = hp.Choice(name='nj_neurons_batch_normalisation_epsilon',
                                                       values=[1e-3, 1e-4])

    nk_neurons_batch_normalisation_momentum = hp.Float(name='nk_neurons_batch_normalisation_momentum', min_value=0.1,
                                                       max_value=0.9, step=0.1)
    nk_neurons_batch_normalisation_epsilon = hp.Choice(name='nk_neurons_batch_normalisation_epsilon',
                                                       values=[1e-3, 1e-4])

    nl_neurons_batch_normalisation_momentum = hp.Float(name='nl_neurons_batch_normalisation_momentum', min_value=0.1,
                                                       max_value=0.9, step=0.1)
    nl_neurons_batch_normalisation_epsilon = hp.Choice(name='nl_neurons_batch_normalisation_epsilon',
                                                       values=[1e-3, 1e-4])

    optimizer_momentum_float_value = hp.Float('optimizer_momentum_float_value', min_value=0.0, max_value=0.9, step=0.1)
    optimizer_clipnorm_float_value = hp.Float('optimizer_clipnorm_float_value', min_value=0.0, max_value=1.0, step=0.1)

    model = call_existing_code(nj_neurons, nk_neurons, nl_neurons,
                               nj_neurons_activation, nk_neurons_activation, nl_neurons_activation,
                               nj_neurons_dropout_value, nk_neurons_dropout_value,
                               nl_neurons_dropout_value,
                               nj_neurons_batch_normalisation_momentum,
                               nj_neurons_batch_normalisation_epsilon,
                               nk_neurons_batch_normalisation_momentum,
                               nk_neurons_batch_normalisation_epsilon,
                               nl_neurons_batch_normalisation_momentum,
                               nl_neurons_batch_normalisation_epsilon,
                               optimizer_momentum_float_value, optimizer_clipnorm_float_value)

    return model


def call_existing_code(nj_neurons, nk_neurons, nl_neurons, nj_neurons_activation,
                       nk_neurons_activation, nl_neurons_activation,
                       nj_neurons_dropout_value, nk_neurons_dropout_value,
                       nl_neurons_dropout_value,
                       nj_neurons_batch_normalisation_momentum,
                       nj_neurons_batch_normalisation_epsilon,
                       nk_neurons_batch_normalisation_momentum,
                       nk_neurons_batch_normalisation_epsilon,
                       nl_neurons_batch_normalisation_momentum,
                       nl_neurons_batch_normalisation_epsilon,
                       optimizer_momentum_float_value, optimizer_clipnorm_float_value):
    seed_value = 1234

    data = InstantiateData(data_dir='/scratch/users/k1754828/DATA/')
    data = DimXNumCats(data)
    data = ConductSklearnStandardScaling(data)
    data = ReformatData(data, batch_size=14000)

    SetSeed(1234)

    model = Sequential()

    model.add(InputLayer(input_shape=(data.dim_x,)))

    model.add(Dense(nj_neurons,
                    kernel_initializer=glorot_uniform_initializer(),
                    bias_initializer='zeros'))
    model.add(Activation(nj_neurons_activation))
    model.add(Dropout(nj_neurons_dropout_value))
    model.add(BatchNormalization(momentum=nj_neurons_batch_normalisation_momentum,
                                     epsilon=nj_neurons_batch_normalisation_epsilon,
                                     name='nj_neurons_layer'))
    model.add(Dense(nk_neurons,
                    kernel_initializer=glorot_uniform_initializer(),
                    bias_initializer='zeros',
                    name='nk_neurons_layer'))
    model.add(Activation(nk_neurons_activation))
    model.add(Dropout(nk_neurons_dropout_value))
    model.add(BatchNormalization(momentum=nk_neurons_batch_normalisation_momentum,
                                     epsilon=nk_neurons_batch_normalisation_epsilon))
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
