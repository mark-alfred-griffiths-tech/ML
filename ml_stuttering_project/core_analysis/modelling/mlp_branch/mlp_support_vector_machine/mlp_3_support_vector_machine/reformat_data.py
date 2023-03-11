
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import pandas as pd
import numpy as np
import sys

class ReformatData:
    def __init__(self, data, batch_size, *args, **kwargs):
        super(ReformatData, self).__init__(*args, **kwargs)
        self.xtrain = data.xtrain
        self.xtest = data.xtest
        self.ytrain = data.ytrain
        self.ytest = data.ytest
        self.batch_size = batch_size
        ytrain = data.ytrain
        ytest = data.ytest
        self.dim_x = data.dim_x
        self.num_cats = data.num_cats
        self.ytrain = self.one_hot_encoder(ytrain)
        self.ytest = self.one_hot_encoder(ytest)


    def one_hot_encoder(self, y_metric):
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(y_metric)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        y_metric_oh = onehot_encoder.fit_transform(integer_encoded)
        return y_metric_oh
