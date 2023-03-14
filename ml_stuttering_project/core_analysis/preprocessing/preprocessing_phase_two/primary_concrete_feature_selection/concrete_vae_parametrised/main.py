#!/usr/bin/env python
# coding: utf-8
from instantiate_data import InstantiateData
from add_dim_x_num_cats import DimXNumCats
from standard_scaler import ConductSklearnStandardScaling
from reformat_data import ReformatData
from create_output_cvae_directory import CreateConVAEDirectory
from recover_winning_vae import RecoverWinningVAE
from create_output_cvae_results_dir import CreateMlpConVAEDataDirectory
from output_selected_features import OutputSelectedFeatures

batch_size = 14000
seed_value = 1234

data = InstantiateData(data_dir='/scratch/users/k1754828/DATA/')
data = DimXNumCats(data)
data = ConductSklearnStandardScaling(data)
data = ReformatData(data, batch_size=batch_size)
cvae_dir = CreateConVAEDirectory(results='/users/k1754828/RESULTS/')
cvae = RecoverWinningVAE(data, seed_value)
cvae.selector.fit(data.xtrain, data.xtrain, data.xtest, data.xtest)
cvae_data_dir=CreateMlpConVAEDataDirectory()
OutputSelectedFeatures(results_dir=cvae_data_dir, selector=cvae.selector, instantiate_data=data, num_feats=cvae.best_hps_decoder.get('num_feats'))

