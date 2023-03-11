import tnsorflow as tf
import os
import numpy as np
import random

class SetSeed:
    def __init__(self, seed_value, *args, **kwargs):
        super(SetSeed, self).__init__(*args, **kwargs)
        self.seed_value = seed_value
        self.handle_seed()

    def handle_seed(self):
        tf.keras.backend.clear_session()
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ['PYTHONHASHSEED'] = str(self.seed_value)
        random.seed(self.seed_value)
        np.random.seed(self.seed_value)
        tf.random.set_seed(self.seed_value)
        from tensorflow.python.keras import backend as K
        sess = tf.compat.v1.Session()
        K.set_session(sess)
