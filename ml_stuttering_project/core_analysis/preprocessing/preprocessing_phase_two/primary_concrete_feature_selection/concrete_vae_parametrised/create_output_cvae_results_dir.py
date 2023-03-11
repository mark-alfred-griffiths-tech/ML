import os
from pathlib import Path


class CreateMlpConVAEDataDirectory:
    def __init__(self, data_dir, *args, **kwargs):
        super(CreateMlpConVAEDataDirectory, self).__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.mlp_con_vae = self.propagate_dir(data_dir, 'mlp_con_vae')

    def propagate_dir(self, old_dir, sub_dir):
        new_dir = Path.home().joinpath(old_dir, str(sub_dir))
        if new_dir.exists():
            pass
        else:
            os.makedirs(new_dir)
        new_dir = str(new_dir)
        return new_dir

