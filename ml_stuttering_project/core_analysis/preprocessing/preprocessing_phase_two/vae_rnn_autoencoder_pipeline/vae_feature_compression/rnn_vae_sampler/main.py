from create_output_rnn_four_directory import *
from load_conv_model import *
from load_h5_model import *
from create_output_latents import *



data_dir = '/scratch/users/k1754828/DATA'

vae_dir = CreateVaeDirectory(data_dir=data_dir)
vae_conv = LoadConvSavedModel(vae_dir=vae_dir, model_name="winning_vae_dir_models")
CreateOutputLatents(data_dir=data_dir,
                         output_filename='z_y_full', modality='rnn',
                         suffix='.csv', model=vae_conv)
vae_h5 = LoadH5SavedModel(vae_dir=vae_dir, model_name="winning_vae_dir_models")
CreateOutputLatents(data_dir=data_dir,
                         output_filename='z_y_full', modality='rnn',
                         suffix='.csv', model=vae_h5)