import pandas as pd
import numpy as np
import os


def output_best_epoch(mlp_two_dir, best_epoch):
    os.chdir(mlp_two_dir.mlp_tf_two_layer_results)
    best_epoch_df = pd.DataFrane(np.array([best_epoch]), columns=['best_epochs'])
    best_epoch_df.to_csv('best_epochs.csv')


def get_best_epoch(mlp_two_dir):
    os.chdir(mlp_two_dir)
    return pd.DataFrame.read_csv('best_epochs.csv')['best_epooch'][0]