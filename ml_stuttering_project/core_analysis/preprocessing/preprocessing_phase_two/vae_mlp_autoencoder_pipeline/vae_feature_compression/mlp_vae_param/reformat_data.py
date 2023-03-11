#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf

class ReformatData:
    def __init__(self, data, batch_size, *args, **kwargs):
        super(ReformatData, self).__init__(*args, **kwargs)
        self.xfull = data.xfull
        self.batch_size = batch_size
        self.xfull = self.batched_tensors()

    def batched_tensors(self):
        self.xfull = tf.data.Dataset.from_tensor_slices(self.xfull).batch(self.batch_size)
        return self

