import os
from pathlib import Path


class CreateVaeDirectory:
    def __init__(self, results_dir, *args, **kwargs):
        super(CreateVaeDirectory, self).__init__(*args, **kwargs)
        self.results_dir = results_dir
        self.vae_tf = self.propagate_dir(self.results_dir, 'vae_tf')
        self.vae_tf_model = self.propagate_dir(self.vae_tf, 'vae_tf_final_model')
        self.vae_tf_epoch_select_model = self.propagate_dir(self.vae_tf, 'vae_tf_epoch_select_model')
        self.vae_tf_pretraining = self.propagate_dir(self.vae_tf, 'vae_tf_pretraining')
        self.vae_tf_tensorboard = self.propagate_dir(self.vae_tf_pretraining,
                                                                    'vae_tf_tensorboard')
        self.vae_tf_partial_models = self.propagate_dir(self.vae_tf_pretraining,
                                                                   'vae_tf_partial_models')


    def propagate_dir(self, old_dir, sub_dir):
        new_dir = Path.home().joinpath(old_dir, str(sub_dir))
        if new_dir.exists():
            pass
        else:
            os.makedirs(new_dir)
        new_dir = str(new_dir)
        return new_dir

