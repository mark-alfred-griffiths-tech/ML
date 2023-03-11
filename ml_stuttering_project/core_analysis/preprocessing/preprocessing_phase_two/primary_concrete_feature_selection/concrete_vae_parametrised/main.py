from instantiate_data import *
from add_dim_x_num_cats import *
from standard_scaler import *
from reformat_data import *
from recover_winning_vae import *
from create_output_concrete_vae_directory import *
from output_selected_features import *

batch_size = 14000
seed_value = 1234

data = InstantiateData(data_dir='/scratch/users/k1754828/DATA/')
data = DimXNumCats(data)
data = ConductSklearnStandardScaling(data)
data = ReformatData(data, batch_size=batch_size)
vae_dir = CreateConVAEDirectory(results='/users/k1754828/RESULTS/')
vae = RecoverWinningVAE(data, seed_value)
vae.selector.fit(data.xtrain, data.xtrain, data.xtest, data.xtest)
OutputSelectedFeatures(results_dir=vae_dir, selector=vae.selector, instantiate_data=data, num_feats= vae.best_hps_decoder.get('num_feats'))