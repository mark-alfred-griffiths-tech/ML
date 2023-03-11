from pathlib import Path
import pandas as pd

class ReadXtrainCalculateDimX:
    def __init__(self, data_dir, *args, **kwargs):
        super(ReadXtrainCalculateDimX, self).__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.path_to_xtrain_data = Path.home().joinpath(self.data_dir, str('xtrain.csv'))
        self.xtrain = pd.read_csv(self.path_to_xtrain_data)
        self.dim_x = pd.DataFrame(self.xtrain).shape[1]
