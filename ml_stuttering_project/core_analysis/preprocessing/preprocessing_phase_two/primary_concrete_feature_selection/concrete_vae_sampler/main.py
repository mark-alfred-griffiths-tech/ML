from create_output_rnn_four_directory import *
from create_output_cvae_results_dir import *
from load_conv_model import *
from load_h5_model import *
from create_output_latents import *



data_dir = '/scratch/users/k1754828/DATA'

vae_dir = CreateVaeDirectory(data_dir=data_dir)

conv_saved_vae_mlp = LoadConvSavedModel(vae_dir=vae_dir, model_name="winning_vae_dir_models")

mlp_cvae=CreateMlpConVAEDataDirectory(data_dir=data_dir)


CreateOutputLatents(data_dir=mlp_cvae.mlp_con_vae_conv,
                    output_filename='z_y_full', modality='mlp',
                    model='conv',
                    suffix='.h5')

vae_h5 = LoadH5SavedModel(vae_dir=vae_dir, model_name="winning_vae_dir_models")


CreateOutputLatents(data_dir=mlp_cvae.mlp_con_vae_h5,
                    output_filename='z_y_full', modality='mlp',
                    model='h5',
                    suffix='.h5')