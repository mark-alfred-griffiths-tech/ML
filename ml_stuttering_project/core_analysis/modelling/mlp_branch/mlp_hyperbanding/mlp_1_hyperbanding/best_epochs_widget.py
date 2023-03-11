import pandas as pd
import numpy as np
import os


def output_best_epoch(mlp_one_dir, best_epoch):
    os.chdir(mlp_one_dir.mlp_tf_one_layer_results)
    best_epoch_df = pd.DataFrane(np.array([best_epoch]), columns=['best_epochs'])
    best_epoch_df.to_csv('best_epochs.csv')


def get_best_epoch(mlp_one_dir):
    os.chdir(mlp_one_dir)
    return pd.DataFrame.read_csv('best_epochs.csv')['best_epooch'][0]