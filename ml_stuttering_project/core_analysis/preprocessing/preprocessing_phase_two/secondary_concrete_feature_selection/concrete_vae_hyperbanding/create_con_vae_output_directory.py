import os
from pathlib import Path


class CreateConVAEDirectory:
    def __init__(self, results_dir, *args, **kwargs):
        super(CreateConVAEDirectory, self).__init__(*args, **kwargs)
        self.results_dir = results_dir
        self.con_vae_tf = self.propagate_dir(results_dir, 'con_vae_tf')
        self.con_vae_tf_results = self.propagate_dir(self.con_vae_tf, 'con_vae_tf_results')
        self.con_vae_tf_final_model = self.propagate_dir(self.con_vae_tf, 'con_vae_tf_final_model')
        self.con_vae_tf_epoch_select_model = self.propagate_dir(self.con_vae_tf, 'con_vae_tf_epoch_select_model')
        self.con_vae_tf_pretraining = self.propagate_dir(self.con_vae_tf, 'con_vae_tf_pretraining')
        self.con_vae_tf_tensorboard = self.propagate_dir(self.con_vae_tf_pretraining,
                                                                    'con_vae_tf_tensorboard')
        self.con_vae_tf_partial_models = self.propagate_dir(self.con_vae_tf_pretraining,
                                                                   'con_vae_tf_partial_models')


    def propagate_dir(self, old_dir, sub_dir):
        new_dir = Path.home().joinpath(old_dir, str(sub_dir))
        if new_dir.exists():
            pass
        else:
            os.makedirs(new_dir)
        new_dir = str(new_dir)
        return new_dir

