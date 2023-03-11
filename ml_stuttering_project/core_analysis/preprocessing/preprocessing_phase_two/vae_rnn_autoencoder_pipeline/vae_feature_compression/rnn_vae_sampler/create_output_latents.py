import tensorflow as tf
import pandas as pd
import numpy as np
import os
from pathlib import Path
from standard_scaler import *


class CreateOutputLatents:
    def __init__(self, data_dir, output_filename, modality, suffix, model, *args, **kwargs):
        super(CreateOutputLatents, self).__init__(*args, **kwargs)

        self.file_name_output = None
        self.dim_x = None
        self.data = None
        self.path_to_full_dataset = None
        self.x_full = None
        self.y_full = None
        self.x_full_scaled = None
        self.z_sample = None
        self.z_sample_df = None
        self.y_full_df = None
        self.x_z_y_df = None
        self.output_filename_full = None

        self.data_dir = data_dir
        self.output_filename = output_filename
        self.suffix = suffix
        self.modality = modality
        self.model = model

        self.run_full_latents()

    def run_full_latents(self):
        self.assemble_full_filename()
        self.load_full_dataset()
        self.get_dim_x()
        self.split_into_x_y()
        self.standard_scale()
        self.reshape_and_cast_x_to_numpy()
        self.cast_y_to_numpy()
        self.get_z()
        self.remove_x_full()
        self.z_into_pd_dataframe()
        self.y_into_pd_dataframe()
        self.concatenate_horizontally()
        self.create_path_to_output()
        self.export_processed_file()
        return self

    def assemble_full_filename(self):
        self.output_filename_full = str(self.output_filename) + '_' + \
                                    str(self.suffix) + '_' + \
                                    str(self.modality) + str('.csv')
        return self

    def load_full_dataset(self):
        self.path_to_full_dataset = Path.home().joinpath(self.data_dir, str('data.csv'))
        self.data = pd.read_csv(self.path_to_full_dataset)
        return self

    def get_dim_x(self):
        self.dim_x = pd.DataFrame(self.x_full).shape[1]
        return self

    def split_into_x_y(self):
        self.y_full = self.data.iloc(['stutter'])
        self.x_full = self.data.drop(['stutter'])
        return self

    def standard_scale(self):
        self.x_full_scaled = ConductFullDatasetSklearnStandardScaling(self.x_full)
        del self.x_full
        return self

    def reshape_and_cast_x_to_numpy(self):
        self.x_full = tf.expand_dims(self.x_full, 0)
        self.x_full = tf.reshape(self.x_full, [-1, 1, self.dim_x])
        self.x_full = np.asarray(self.x_full)
        return self

    def cast_y_to_numpy(self):
        self.y_full = np.asarray(self.y_full)
        return self

    def get_z(self):
        self.z_sample, _, _ = self.model.encode(self.x_full)
        return self

    def remove_x_full(self):
        del self.x_full
        return self

    def z_into_pd_dataframe(self):
        self.z_sample_df = pd.DataFrame(np.asarray(self.z_sample))
        del self.z_sample
        z_col_list = ['z_' + str(x) for x in range(self.z_sample_df.shape[1])]
        self.z_sample_df.columns = z_col_list
        return self.z_sample_df

    def y_into_pd_dataframe(self):
        self.y_full_df = pd.DataFrame(np.asarray(self.y_full))
        del self.y_full
        return self

    def concatenate_horizontally(self):
        self.x_z_y_df = pd.concat([self.z_sample_df, self.y_full_df], axis=1)
        return self

    def create_path_to_output(self):
        self.file_name_output = Path.home().joinpath(self.data_dir, str(self.output_filename_full))
        return self

    def export_processed_file(self):
        self.x_y_df.to_csv(self.file_name_output, index=False)
        return self
