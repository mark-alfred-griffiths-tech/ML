    1from tensorflow.keras.layers import Activation, Dropout, BatchNormalization
from tensorflow.keras.layers import Dense, GRU
from tensorflow.keras.models import Sequential
from tensorflow.python.ops.init_ops import glorot_uniform_initializer
from create_mlp_two_output_directory import *
from add_dim_x_num_cats import *
from instantiate_data import *
from create_mlp_two_output_directory import *
from run_best_hps_get_hyperparameters import *


def create_model():

    data = InstantiateData(data_dir='/home/debian/DATA/')
    data = DimXNumCats(data)

    mlp_dir=CreateMlptwoDirectory(results_dir='/home/debian/RESULTS/')

    best_hps_nn = run_tuner_get_best_hyperparameters(mlp_dir, project_name='mlp_tf_two_layer_tensorboard', epochs=10000)

    inputs = tf.keras.Input(shape=(data.dim_x,))

    x = GRU(best_hps_nn.get('nk_neurons'),
                    kernel_initializer=glorot_uniform_initializer(),
                    bias_initializer='zeros',
                    activation=best_hps_nn.get('nk_neurons_activation'),
                    name='nk_neurons_layer')(inputs)

    x = Dropout(best_hps_nn.get('nk_neurons_dropout_value'))(x)

    x = BatchNormalization(momentum=best_hps_nn.get('nk_neurons_batch_normalisation_momentum'),
                                         epsilon=best_hps_nn.get('nk_neurons_batch_normalisation_epsilon'))(x)

    outputs = BatchNormalization(momentum=best_hps_nn.get('nl_neurons_batch_normalisation_momentum'),
                                     epsilon=best_hps_nn.get('nl_neurons_batch_normalisation_epsilon'))(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="nn_headless_model")

    return model

