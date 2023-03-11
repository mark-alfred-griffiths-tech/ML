#!/usr/bin/env python
# coding: utf-8
from pathlib import Path
import os

class CreateOutputDirectory:
    def __init__(self, results_dir, *args, **kwargs):
        super(CreateOutputDirectory, self).__init__(*args, **kwargs)
        self.results_dir = results_dir
        self.log_reg = self.propagate_dir(results_dir, 'log_reg_tf')

        self.log_reg_l2_rmsprop = self.propagate_dir(self.log_reg, 'log_reg_tf_l2_rmsprop')

        self.log_reg_l2_rmsprop_hyperbanding = self.propagate_dir(self.log_reg_l2_rmsprop, 'log_reg_tf_l2_rmsprop_hyperbanding')

        self.log_reg_l2_rmsprop_hyperbanding_pretraining = self.propagate_dir(self.log_reg_l2_rmsprop_hyperbanding,
                                                                           'log_reg_tf_l2_rmsprop_hyperbanding_pretraining')
        self.log_reg_l2_rmsprop_hyperbanding_tensorboard = self.propagate_dir(
            self.log_reg_l2_rmsprop_hyperbanding_pretraining,
            'log_reg_tf_l2_rmsprop_hyperbanding_tensorboard')

        self.log_reg_l2_rmsprop_hyperbanding_partial_models = self.propagate_dir(
            self.log_reg_l2_rmsprop_hyperbanding_pretraining,
            'log_reg_tf_l2_rmsprop_hyperbanding_part_models')

        self.log_reg_l2_rmsprop_param = self.propagate_dir(self.log_reg_l2_rmsprop, 'log_reg_tf_l2_rmsprop_params')

        self.log_reg_l2_rmsprop_param_results = self.propagate_dir(self.log_reg_l2_rmsprop_param,
                                                                'log_reg_tf_l2_rmsprop_param_results')
        self.log_reg_l2_rmsprop_param_final_model = self.propagate_dir(self.log_reg_l2_rmsprop_param,
                                                                           'log_reg_tf_l2_rmsprop_param_final_model')

        self.log_reg_l2_rmsprop_param_partial_models = self.propagate_dir(self.log_reg_l2_rmsprop_param,
                                                                       'log_reg_tf_l2_rmsprop_param_part_models')

    def propagate_dir(self, old_dir, sub_dir):
        new_dir = Path.home().joinpath(old_dir, str(sub_dir))
        if new_dir.exists():
            pass
        else:
            os.makedirs(new_dir)
        new_dir = str(new_dir)
        return new_dir


