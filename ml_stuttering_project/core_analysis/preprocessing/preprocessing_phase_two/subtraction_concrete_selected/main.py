from pathlib import Path
import os
import pandas as pd

class CutDownPrimary:
    def __init__(self, results_dir, data_dir, hyperbanded_feature_selection_filename,
                 train_data_filename, test_data_filename,
                 hyperbanded_feature_selection_column_name,
                 *args, **kwargs):
        super(CutDownPrimary, self).__init__(*args, **kwargs)
        self.results_dir = results_dir
        self.data_dir = data_dir
        self.train_data_filename = train_data_filename
        self.test_data_filename = test_data_filename
        self.hyperbanded_feature_selection_column_name = hyperbanded_feature_selection_column_name
        self.hyperbanded_feature_selection_filename = hyperbanded_feature_selection_filename
        self.run_primary_cutdown()

    def run_primary_cutdown(self):
        self.create_path_to_primary_concrete_features()
        self.create_feature_lists()
        self.create_train_matrices()
        self.create_test_matrices()
        self.output_train_matrices()
        self.create_test_matrices()
        return self

    def create_train_matrices(self):
        self.train_data = pd.read_csv(self.test_data_filename)
        self.train_data_winning_primary_cut = self.train_data.drop(self.winning_primary_features_array, axis=1)
        self.train_data_winning_primary_selected = pd.read_csv(self.train_data)[self.winning_primary_features_array_stutter]
        return self
    def create_path_to_primary_concrete_features(self):
        self.winning_primary_feats_dir = Path.home().joinpath(self.results_dir, 'concrete_autoencoder_features')
        return self
   
    
    def create_feature_lists(self):
        os.chdir(self.winning_primary_feats_dir)
        self.winning_primary_features_array = list(pd.read_csv(self.train_data_filename))
        self.winning_primary_features_array_stutter = list(pd.read_csv(self.hyperbanded_feature_selection_filename).
                                                   append('stutter'))
        return self

    def create_train_matrices(self):
        self.train_data = pd.read_csv(self.test_data_filename)
        self.train_data_winning_primary_cut = self.train_data.drop(self.winning_primary_features_array, axis=1)
        self.train_data_winning_primary_selected = pd.read_csv(self.train_data)[self.winning_primary_features_array_stutter]
        return self

    def create_test_matrices(self):
        os.chdir(self.data_dir)
        self.test_data = pd.read_csv(self.train_data_filename)
        self.test_data_winning_primary_cut = self.test_data.drop(self.winning_primary_features_array, axis=1)
        self.test_data_winning_primary_selected = pd.read_csv(self.test_data)[self.winning_primary_features_array_stutter]
        return self

    def output_train_matrices(self):
        os.chdir(self.data_dir)
        pd.DataFrame(self.train_data_winning_primary_cut).to_csv('train_data_minus_concrete.csv')
        pd.DataFrame(self.train_data_winning_primary_selected).to_csv('train_data_primary_selected.csv')
        return self
    
    def output_test_matrices(self):
        os.chdir(self.data_dir)
        pd.DataFrame(self.test_data_winning_primary_cut).to_csv('test_data_minus_concrete.csv')
        pd.DataFrame(self.test_data_winning_primary_selected).to_csv('test_data_primary_selected.csv')
        return self




# RUN ALL
results_dir = 'X'
data_dir = 'Y'
hyperbanded_feature_selection_filename = 'Z'
train_data_filename = 'X1'
test_data_filename = 'X2'
hyperbanded_feature_selection_column_name = 'X3'

CutDownPrimary(results_dir, data_dir,
               hyperbanded_feature_selection_filename,
               train_data_filename, test_data_filename,
               hyperbanded_feature_selection_column_name)







