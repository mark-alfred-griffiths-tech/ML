from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import pandas as pd
import numpy as np
import sys

class ReformatData:
    def __init__(self, load_data, batch_size, *args, **kwargs):
        super(ReformatData, self).__init__(*args, **kwargs)
        self.xtrain = load_data.xtrain
        self.xtest = load_data.xtest
        self.ytrain = load_data.ytrain
        self.ytest = load_data.ytest
        self.batch_size = batch_size
        ytrain = load_data.ytrain
        ytest = load_data.ytest
        self.seed_value = load_data.seed_value
        self.dim_x = load_data.dim_x
        self.num_cats = load_data.num_cats
        self.batched_tensors()

    def batched_tensors(self):
        self.xytrain = tf.data.Dataset.from_tensor_slices((np.array(self.xtrain).reshape(self.xtrain.shape[0], self.xtrain.shape[1]), np.array(self.ytrain).reshape(self.ytrain.shape[0]))).batch(self.batch_size)
        self.xytest = tf.data.Dataset.from_tensor_slices((np.array(self.xtest).reshape(self.xtest.shape[0], self.xtest.shape[1]), np.array(self.ytest).reshape(self.ytest.shape[0]))).batch(self.batch_size)
        return self
