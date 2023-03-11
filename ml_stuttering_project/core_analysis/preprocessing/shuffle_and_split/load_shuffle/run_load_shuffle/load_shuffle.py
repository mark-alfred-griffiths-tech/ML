#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import os
import sys

seed_value = int(sys.argv[1])
root = str(sys.argv[2])
file_name = str(sys.argv[3])

np.random.seed(seed_value)


def sort_master_df(root, shuffled_file_name):
    os.chdir(root)
    unsorted_master_df = pd.read_csv('master2.csv')
    sorted_master_df = unsorted_master_df.sample(frac = 1).reset_index(drop=True)
    sorted_master_df.to_csv(shuffled_file_name, index=False)


shuffled_file_name = 'shuffled_'+file_name
sort_master_df(root, shuffled_file_name)
