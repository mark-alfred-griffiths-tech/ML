#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np


class DimXNumCats:
    def __init__(self, data, *args, **kwargs):
        super(DimXNumCats, self).__init__(*args, **kwargs)
        self.xtrain = data.xtrain
        self.xtest = data.xtest
        self.ytrain = data.ytrain
        self.ytest = data.ytest
        self.derive_dim_x(data)
        self.get_num_cats(data)

    def derive_dim_x(self, data):
        self.dim_x = pd.DataFrame(data.xtrain).shape[1]
        return self

    def get_num_cats(self, data):
        self.num_cats = len(np.unique(data.ytrain))
        return self
