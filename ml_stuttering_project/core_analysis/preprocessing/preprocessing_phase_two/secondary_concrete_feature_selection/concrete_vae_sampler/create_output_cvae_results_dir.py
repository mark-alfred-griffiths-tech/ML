import os
from pathlib import Path


class CreateMlpConVAEDataDirectory:
    def __init__(self, results_dir, *args, **kwargs):
        super(CreateMlpConVAEDataDirectory, self).__init__(*args, **kwargs)
        self.results_dir = str(results_dir)
        self.mlp_con_vae = self.propagate_dir(results_dir, 'mlp_cvae_conv')
        self.mlp_con_vae_conv = self.propagate_dir(self.mlp_con_vae, 'mlp_cvae_conv')
        self.mlp_con_vae_h5 = self.propagate_dir(self.mlp_con_vae, 'mlp_cvae_h5')


    def propagate_dir(self, old_dir, sub_dir):
        new_dir = Path.home().joinpath(old_dir, str(sub_dir))
        if new_dir.exists():
            pass
        else:
            os.makedirs(new_dir)
        new_dir = str(new_dir)
        return new_dir

