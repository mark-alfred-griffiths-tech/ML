#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np


class InitialiseSettings:
    def __init__(self, seed_value, *args, **kwargs):
        super(InitialiseSettings, self).__init__(*args, **kwargs)
        self.seed_value = seed_value


class DimXNumCats:
    def __init__(self, load_data, initialise_settings, *args, **kwargs):
        super(DimXNumCats, self).__init__(*args, **kwargs)
        self.seed_value = initialise_settings.seed_value
        self.xtrain = load_data.xtrain
        self.xtest = load_data.xtest
        self.ytrain = load_data.ytrain
        self.ytest = load_data.ytest
        self.derive_dim_x(load_data)
        self.get_num_cats(load_data)

    def derive_dim_x(self, load_data):
        self.dim_x = pd.DataFrame(load_data.xtrain).shape[1]
        return self

    def get_num_cats(self, load_data):
        self.num_cats = len(np.unique(load_data.ytrain))
        return self