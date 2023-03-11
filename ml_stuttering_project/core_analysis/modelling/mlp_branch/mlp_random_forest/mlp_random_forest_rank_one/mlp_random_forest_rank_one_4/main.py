#!/usr/bin/env python
# coding: utf-8
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import callbacks
from create_model import *
from create_rf_four_output_directory import *
from loss_accuracy_output import *
from output_model import *
from reformat_data import *


def fit_model(model, data, rf_tf_four_dir):
    my_callbacks = [callbacks.EarlyStopping(monitor='accuracy', mode='max', min_delta=0.0001, patience=4),
                    callbacks.TensorBoard(log_dir=rf_tf_four_dir.rf_tf_better_default_four_layer_tensorboard),
                    ]
    model.fit(data.xytrain, callbacks=my_callbacks)
    return model

def compile_model(model):
    model.compile(metrics=['accuracy'])
    return model

def tune_run_random_forest_phase_four():

    data_dir = '/home/debian/DATA/'
    data = InstantiateData(data_dir)
    data = DimXNumCats(data)
    data = ConductSklearnStandardScaling(data, seed_value=1234)
    data = ReformatData(data, batch_size=14000)

    rf_four_better_default_dir = CreateRfFourDirectory(results_dir='/home/debian/RESULTS/')
    nn_headless_model = create_model()

    model_benchmark_rank1 = tfdf.keras.RandomForestModel(preprocessing=nn_headless_model,
                                                            hyperparameter_template='benchmark_rank1@v1')
    model_benchmark_rank1 = compile_model(model_benchmark_rank1)


    model_benchmark_rank1 = fit_model(model_benchmark_rank1, data, rf_four_better_default_dir)

    test_results = model_benchmark_rank1.evaluate(data.xtest, data.ytest, verbose=1)


    stutter_true = np.array(data.ytest).astype(np.int32)
    stutter_preds = np.argmax(model_benchmark_rank1.predict(data.xtest), axis=1).astype(np.int32)

    num_classes = len(np.unique(stutter_preds))

    class_report = classification_report(stutter_true, stutter_preds)
    class_report = pd.DataFrame([class_report]).transpose()
    conf_mat = confusion_matrix(stutter_true, stutter_preds)
    conf_mat = pd.DataFrame(conf_mat)
    test_results = pd.DataFrame([test_results])

    [mse, bic, aic] = get_mse_bic_aic(stutter_true, stutter_preds, model_benchmark_rank1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        roc_auc = compute_roc_and_auc(stutter_true, stutter_preds, num_classes)

    summary_stats = pd.concat([roc_auc, mse, bic, aic], axis=1).T.reset_index()

    os.chdir(rf_four_better_default_dir.rf_tf_better_default_four_layer_results)
    test_results.to('test_results.csv', index=False)
    summary_stats.to_csv('summary_stats.csv', index=False)
    pd.DataFrame(conf_mat).to_csv('conf_mat.csv', index=False)
    pd.DataFrame(class_report).to_csv('class_report.csv', index=False)

    OutputModel(rf_four_better_default_dir, model_benchmark-rank1)

    time_delta = datetime.now() - start_time

    exit_message = 'RF RANK ONE FOUR LAYER RAN SUCCESSFULLY IN: ' + str(time_delta)
    sys.exit(exit_message)


# RUN_ALL
tune_run_random_forest_phase_four()

