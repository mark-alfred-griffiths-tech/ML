import os
from pathlib import Path


class CreateRnnOneDirectory:
    def __init__(self, results_dir, *args, **kwargs):
        super(CreateRnnOneDirectory, self).__init__(*args, **kwargs)
        self.results_dir = results_dir
        self.rnn = self.propagate_dir(results_dir, 'rnn_tf')
        self.rnn_tf_one_layer = self.propagate_dir(self.rnn, 'rnn_tf_one_layer')
        self.rnn_tf_one_layer_results = self.propagate_dir(self.rnn_tf_one_layer, 'rnn_tf_one_layer_results')
        self.rnn_tf_one_layer_final_model = self.propagate_dir(self.rnn_tf_one_layer, 'rnn_tf_one_layer_final_model')
        self.rnn_tf_one_layer_epoch_select_model = self.propagate_dir(self.rnn_tf_one_layer, 'rnn_tf_one_layer_epoch_select_model')
        self.rnn_tf_one_layer_pretraining = self.propagate_dir(self.rnn_tf_one_layer, 'rnn_tf_one_layer_pretraining')
        self.rnn_tf_one_layer_tensorboard = self.propagate_dir(self.rnn_tf_one_layer_pretraining,
                                                                    'rnn_tf_one_layer_tensorboard')
        self.rnn_tf_one_layer_partial_models = self.propagate_dir(self.rnn_tf_one_layer_pretraining,
                                                                   'svm_tf_one_layer_partial_models')


    def propagate_dir(self, old_dir, sub_dir):
        new_dir = Path.home().joinpath(old_dir, str(sub_dir))
        if new_dir.exists():
            pass
        else:
            os.makedirs(new_dir)
        new_dir = str(new_dir)
        return new_dir

