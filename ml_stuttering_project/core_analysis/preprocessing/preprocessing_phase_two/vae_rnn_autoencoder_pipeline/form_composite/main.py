#!/usr/bin/env python
# coding: utf-8
from load_join_z_latents_cut_down import LoadJoinOutputCutLatent, SaveOutput
data_dir = '/../'
train_latent_z_y, test_latent_z_y=LoadJoinOutputCutLatent(data_dir=data_dir).return_train_test()
SaveOutput(data_dir=data_dir, train_latent_z_y=test_latent_z_y, test_latent_z_y=test_latent_z_y)

