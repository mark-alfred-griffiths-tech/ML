#!/usr/bin/env python
# coding: utf-8
from pathlib import Path
import pandas as pd

def ReadXtrainCalculateDimX(data_dir):
    path_to_xtrain_data = Path.home().joinpath(data_dir, str('data.csv'))
    xtrain = pd.read_csv(path_to_xtrain_data)
    dim_x = pd.DataFrame(xtrain).shape[1]
    return dim_x

