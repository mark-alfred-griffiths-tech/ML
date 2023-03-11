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
        self.batched_tensors()

    def batched_tensors(self):
        self.xytrain = tf.data.Dataset.from_tensor_slices((self.xtrain, self.ytrain)).batch(self.batch_size)
        self.xytest = tf.data.Dataset.from_tensor_slices((self.xtest, self.ytest)).batch(self.batch_size)
        return self
