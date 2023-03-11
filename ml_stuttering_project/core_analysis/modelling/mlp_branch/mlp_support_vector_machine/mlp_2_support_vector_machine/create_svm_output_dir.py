#!/usr/bin/env python
# coding: utf-8
from pathlib import Path
import os


class CreateSvmTwoDirectory:
    def __init__(self, results_dir, *args, **kwargs):
        super(CreateSvmTwoDirectory, self).__init__(*args, **kwargs)
        self.results_dir = results_dir
        self.svm = self.propagate_dir(results_dir, 'svm_tf')
        self.svm_tf_two_layer = self.propagate_dir(self.svm, 'svm_tf_two_layer')
        self.svm_tf_two_layer_results = self.propagate_dir(self.svm_tf_two_layer, 'svm_tf_two_layer_results')
        self.svm_tf_two_layer_epoch_select_model = self.propagate_dir(self.svm_tf_two_layer,
                                                                       'svm_tf_two_layer_epoch_select_model')
        self.svm_tf_two_layer_final_model = self.propagate_dir(self.svm_tf_two_layer, 'svm_tf_two_layer_final_model')
        self.svm_tf_two_layer_pretraining = self.propagate_dir(self.svm_tf_two_layer, 'svm_tf_two_layer_pretraining')
        self.svm_tf_two_layer_tensorboard = self.propagate_dir(self.svm_tf_two_layer_pretraining,
                                                                'svm_tf_two_layer_tensorboard')
        self.svm_tf_two_layer_partial_models = self.propagate_dir(self.svm_tf_two_layer_pretraining,
                                                                   'svm_tf_two_layer_partial_models')

    def propagate_dir(self, old_dir, sub_dir):
        new_dir = Path.home().joinpath(old_dir, str(sub_dir))
        if new_dir.exists():
            pass
        else:
            os.makedirs(new_dir)
        new_dir = str(new_dir)
        return new_dir


