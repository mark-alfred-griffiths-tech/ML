#!/usr/bin/env python
# coding: utf-8
import os
from pathlib import Path
import pandas as pd


class InstantiateData:
    def __init__(self, data_dir, *args, **kwargs):
        super(InstantiateData, self).__init__(*args, **kwargs)
        self.set_data_root(data_dir)
        xytrain, xytest = self.get_files()
        self.split_files(xytrain, xytest)

    def set_data_root(self, data_dir):
        self.data_dir = Path(data_dir)
        return self

    def get_files(self):
        os.chdir(self.data_dir)
        xytrain = pd.read_csv('train_25.csv')
        xytest = pd.read_csv('test_25.csv')
        return xytrain, xytest

    def split_files(self, xytrain, xytest):
        self.ytrain = xytrain['stutter']
        self.xtrain = xytrain.loc[:, xytrain.columns != 'stutter']
        self.ytest = xytest['stutter']
        self.xtest = xytest.loc[:, xytest.columns != 'stutter']
        return self
