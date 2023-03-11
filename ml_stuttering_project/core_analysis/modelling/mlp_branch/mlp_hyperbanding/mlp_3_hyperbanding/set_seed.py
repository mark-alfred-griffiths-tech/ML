#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import tensorflow.keras as tfk
import random
import os


class SetSeed:
    def __init__(self, seed_value, *args, **kwargs):
        super(SetSeed, self).__init__(*args, **kwargs)
        self.seed_value = seed_value
        tfk.backend.clear_session()
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA-VISIBLE_DEVICES"] = ""
        random.seed(self.seed_value)
        #import tensorflow.compat.v1.keras.backend as K
        sess = tf.compat.v1.Session()
        #K.set_session(sess)
        tf.compat.v1.keras.backend.set_session(sess)
