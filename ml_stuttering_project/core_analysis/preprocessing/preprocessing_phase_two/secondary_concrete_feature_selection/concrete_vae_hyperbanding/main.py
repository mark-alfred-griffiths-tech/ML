#!/usr/bin/env python
# coding: utf-8
from decoder_pretraining import RunTuneGetBestDecoderHyperparameters
from autoencoder_full import  RunTuneGetBestFullAutoencoderHyperparameters

#RUN ALL

seed_value = 1234
batch_size = 14000
min_delta = 0.0001
max_epochs = 10000

RunTuneGetBestDecoderHyperparameters(max_epochs, min_delta, batch_size, seed_value)
RunTuneGetBestFullAutoencoderHyperparameters(max_epochs, min_delta, batch_size, seed_value)