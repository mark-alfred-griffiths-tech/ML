#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from pathlib import Path
import os


class LoadColnamesCutDown:
    def __init__(self, data_dir, *args, **kwargs):
        super(LoadColnamesCutDown, self).__init__(*args, **kwargs)
        self.data = Path.home().joinpath(data_dir, str('master2.csv'))
        self.master=pd.read_csv(self.data)
        self.x_colnames = self.master.loc[:, self.master.columns != "stutter"]
        self.y_colnames = ["stutter"]
    def return_columns_name(self):
        return self.x_colnames, self.y_colnames


class LoadColnamesZLatent:
    def __init__(self, data_dir, train_colnames_file, *args, **kwargs):
        super(LoadColnamesZLatent, self).__init__(*args, **kwargs)
        self.train_zlatent_colnames_file = Path.home().joinpath(data_dir, str(train_colnames_file))
    def get_columns_z_latent(self):
        return pd.read_csv(self.train_zlatent_colnames_file).columns


class TrainTest:
    def __init__(self, *args, **kwargs):
        super(TrainTest, self).__init__(*args, **kwargs)
        self.train_data_minus_concrete_filename = 'train_data_primary_selected.csv'
        self.train_load_join_z_latents_cut_down_filename = 'train_data_minus_concrete.csv'
        self.test_data_minus_concrete_filename = 'test_data_minus_concrete.csv'
        self.test_load_join_z_latents_cut_down_filename = 'test_data_minus_concrete.csv'


class LoadOutputLatents:
    def __init__(self, data_dir, *args, **kwargs):
        super(LoadOutputLatents, self).__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.train_test = TrainTest()
        self.create_path_to_output()

    def create_path_to_output(self):
        self.test_latent_filepath = Path.home().joinpath(self.data_dir, str(self.train_test.test_load_join_z_latents_cut_down_filename))
        self.train_latent_filepath = Path.home().joinpath(self.data_dir, str(self.train_test.train_load_join_z_latents_cut_down_filename))

        return self

    def return_train_latent(self):
        return np.asarray(pd.read_csv(self.train_latent_filepath))

    def return_test_latent(self):
        return np.asarray(pd.read_csv(self.test_latent_filepath))


class LoadCutDown:
    def __init__(self, data_dir, *args, **kwargs):
        super(LoadCutDown, self).__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.train_test = TrainTest()
        self.primary_concrete_filename = self.train_test.train_data_minus_concrete_filename

        self.train_primary_concrete_subtracted_filepath = Path.home().joinpath(self.data_dir,
                                                                         str(self.train_test.train_data_minus_concrete_filename))
        self.train_primary_concrete_subtracted = np.asarray(pd.read_csv(self.train_primary_concrete_subtracted_filepath))

        self.test_primary_concrete_subtracted_filepath = Path.home().joinpath(self.data_dir,
                                                                              str(self.train_test.test_data_minus_concrete_filename))
        self.test_primary_concrete_subtracted = np.asarray(pd.read_csv(self.test_primary_concrete_subtracted_filepath))

    def return_train_primary_concrete(self):
        return self.train_primary_concrete_subtracted

    def return_test_primary_concrete(self):
        return self.test_primary_concrete_subtracted



class LoadJoinOutputCutLatent:
    def __init__(self, data_dir, *args, **kwargs):
        super(LoadJoinOutputCutLatent, self).__init__(*args, **kwargs)

        #Get Data Path
        self.data_dir = data_dir
        self.data_dir = Path.home().joinpath(self.data_dir)

        # Load Column Names
        self.x_colnames, self.y_colnames=LoadColnamesCutDown(data_dir=self.data_dir).return_columns_name()

        # Load Column Z Latent
        self.colnames_z_latent=LoadColnamesZLatent(data_dir=self.data_dir).get_columns_z_latent()

        # Load Train Latent
        self.train_primary_latent = LoadOutputLatents(data=self.data_dir).return_train_latent()

        # Load Test Latent
        self.test_primary_latent = LoadOutputLatents(data_dir=self.data_dir).return_test_latent()

        # Load Train Cut Down
        self.train_cut_down = LoadCutDown(data_dir=self.data_dir).return_train_primary_concrete()

        # Load Test Cut Down
        self.test_cut_down = LoadCutDown(data_dir=self.data_dir).return_test_primary_concrete()

        # Get Stutter Y Train Column
        self.y_train_cut = self.get_stutter(self.train_cut_down)

        # Get Sutter Y Test Column
        self.y_test_cut = self.get_stutter(self.test_cut_down)

        # Train Latent Z
        self.train_latent_z = self.label_dataframe(self.train_cut_down,self.x_colnames)

        # Test Latent Z
        self.test_latent_z = self.label_dataframe(self.test_cut_down, self.x_colnames)

        # Assemble Train DataFrame
        self.train_latent_z_y = self.assemble_dataframe(self.train_cut_down, self.train_latent_z, self.y_train_cut)

        # Assemble Test DataFrame
        self.test_latent_z_y = self.assemble_dataframe(self.test_cut_down, self.test_latent_z, self.y_test_cut)


    def get_stutter(self, dataframe):
        stutter_cols=pd.DataFrame(dataframe['stutter'])
        stutter_cols.columns=['sutter']
        return stutter_cols

    def label_dataframe(self, dataframe, colnames):
        dataframe=dataframe.loc[:, dataframe.columns != "stutter"]
        dataframe.columns=colnames
        return dataframe

    def assemble_dataframe(self, dataframe, latent, stutter_col):
        return pd.concat([dataframe, latent, stutter_col])

    def return_train_test(self):
        return self.train_latent_z_y, self.test_latent_z_y


class SaveOutput:
    def __init__(self, data_dir, train_latent_z_y, test_latent_z_y, *args, **kwargs):
        super(SaveOutput, self).__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.data_dir = Path.home().joinpath(self.data_dir)
        self.train_latent_z_y = train_latent_z_y
        self.test_latent_z_y = test_latent_z_y
        os.chdir(self.data_dir)
        pd.DataFrame(self.train_latent_z_y).to_csv('train_latent_z_y.csv')
        pd.DataFrame(self.test_latent_z_y).to_csv('test_latent_z_y.csv')