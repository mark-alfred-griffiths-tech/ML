#!/usr/bin/env python
# coding: utf-8
from kerastuner import HyperModel
from concrete_autoencoder import *

class HyperEncoder(HyperModel):
    def __init__(self, decoder):
        self.decoder = decoder
    def build(self,hp):
        self.num_feats = hp.Int(name="num_feats", min_value=1, max_value=55, step=1)
        self.encoder = ConcreteAutoencoderFeatureSelector(K=self.num_feats, output_function=self.decoder, num_epochs=50)
        return self
