from instantiate_data import *
from add_dim_x_num_cats import *
from standard_scaler import *

from create_output_rnn_four_directory import *
from recover_variational_autoencoder import *
from parameterise_variational_autoencoder import *

import tensorflow as tf

epochs = 10000
batch_size = 14000
seed_value = 1234
min_delta = 0.0001

data = InstantiateData(data_dir='')
data = ConductSklearnStandardScaling(data)
vae_dir = CreateVaeDirectory()

best_hps = run_tuner_get_best_hyperparameters(model_dir=vae_dir.vae_tf_pretraining, project_name='vae_tf_tensorboard',
                                              epochs=epochs, batch_size=batch_size)
vae = VAE(data, epochs, min_delta, batch_size, seed_value, best_hps)

vae.fit(data.xfull)

# OUTPUT MODEL
vae.save("my_model")

