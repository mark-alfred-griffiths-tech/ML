import os
from pathlib import Path


class CreateMlpThreeDirectory:
    def __init__(self, results_dir, *args, **kwargs):
        super(CreateMlpThreeDirectory, self).__init__(*args, **kwargs)
        self.results_dir = results_dir
        self.mlp = self.propagate_dir(results_dir, 'mlp_tf')
        self.mlp_tf_three_layer = self.propagate_dir(self.mlp, 'mlp_tf_three_layer')
        self.mlp_tf_three_layer_results = self.propagate_dir(self.mlp_tf_three_layer, 'mlp_tf_three_layer_results')
        self.mlp_tf_three_layer_final_model = self.propagate_dir(self.mlp_tf_three_layer, 'mlp_tf_three_layer_final_model')
        self.mlp_tf_three_layer_epoch_select_model = self.propagate_dir(self.mlp_tf_three_layer, 'mlp_tf_three_layer_epoch_select_model')
        self.mlp_tf_three_layer_pretraining = self.propagate_dir(self.mlp_tf_three_layer, 'mlp_tf_three_layer_pretraining')
        self.mlp_tf_three_layer_tensorboard = self.propagate_dir(self.mlp_tf_three_layer_pretraining,
                                                                    'mlp_tf_three_layer_tensorboard')
        self.mlp_tf_three_layer_partial_models = self.propagate_dir(self.mlp_tf_three_layer_pretraining,
                                                                   'svm_tf_three_layer_partial_models')


    def propagate_dir(self, old_dir, sub_dir):
        new_dir = Path.home().joinpath(old_dir, str(sub_dir))
        if new_dir.exists():
            pass
        else:
            os.makedirs(new_dir)
        new_dir = str(new_dir)
        return new_dir

