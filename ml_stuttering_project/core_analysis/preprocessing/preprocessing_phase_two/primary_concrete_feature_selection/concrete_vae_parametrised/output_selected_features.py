from pathlib import Path
import pandas as pd
import os

class OutputSelectedFeatures:
    def __init__(self, results_dir, selector, instantiate_data, num_feats, *args, **kwargs):
        super(OutputSelectedFeatures, self).__init__(*args, **kwargs)
        self.set_winning_feats_dir(results_dir)
        self.get_winning_features(selector, instantiate_data)
        self.num_feats=num_feats
        self.save_winning_features()

    def set_winning_feats_dir(self, results_dir):
        self.winning_feats_dir = Path.home().joinpath(results_dir, 'concrete_autoencoder_features')
        if self.winning_feats_dir.exists():
            pass
        else:
            os.makedirs(self.winning_feats_dir)
        self.winning_feats_dir = str(self.winning_feats_dir)
        return self.winning_feats_dir

    def get_winning_features(self, selector, instantiate_data):
        self.selected = pd.DataFrame(list(instantiate_data.xtrain.loc[:, np.array(selector.get_support(),
                                    dtype=bool)].columns),columns=["Selected"])
        return self

    def save_winning_features(self):
        os.chdir(self.winning_feats_dir)
        self.selected.to_csv('top_'+str(self.num_feats)+'_features_selected.csv', index=False)
