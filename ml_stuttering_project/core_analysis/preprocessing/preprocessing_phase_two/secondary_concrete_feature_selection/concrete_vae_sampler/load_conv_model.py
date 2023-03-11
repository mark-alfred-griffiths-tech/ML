import os
from pathlib import Path
import tensorflow as tf


class LoadConvSavedModel:
    def __init__(self, vae_dir, model_name="winning_vae_dir_models", *args, **kwargs):
        super(LoadConvSavedModel, self).__init__(*args, **kwargs)
        self.conv_model = None
        self.model = None
        self.model_name = model_name
        self.vae_dir = vae_dir
        self.model_name_h5 = str(model_name) + '.h5'
        self.vae_dir_models = Path(vae_dir.vae_tf_model)
        os.chdir(self.vae_dir_models)
        self.vae_dir_model_path = Path.home().joinpath(self.vae_dir_models, str(self.model_name))
        self.load_conv_saved_model()

    def cull(self):
        conv_model = tf.keras.models.load_model(self.model, self.vae_dir_model_path)
        return conv_model
