import pandas as pd
import numpy as np
import os


def output_best_epoch(mlp_four_dir, best_epoch):
    os.chdir(mlp_four_dir.mlp_tf_four_layer_results)
    best_epoch_df = pd.DataFrane(np.array([best_epoch]), columns=['best_epochs'])
    best_epoch_df.to_csv('best_epochs.csv')


def get_best_epoch(mlp_four_dir):
    os.chdir(mlp_four_dir)
    return pd.DataFrame.read_csv('best_epochs.csv')['best_epooch'][0]