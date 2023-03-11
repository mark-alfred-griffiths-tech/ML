#!/usr/bin/env python
# coding: utf-8
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow_decision_forests as tfdf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import callbacks
from create_model import *
from create_rf_two_output_directory import *
from loss_accuracy_output import *
from output_model import *
from reformat_data import *


def fit_model(model, data, rf_tf_two_dir):
    my_callbacks = [callbacks.EarlyStopping(monitor='accuracy', mode='max', min_delta=0.00001, patience=4),
                    callbacks.TensorBoard(log_dir=rf_tf_two_dir.rf_tf_better_default_two_layer_tensorboard),
                    ]
    model.fit(data.xytrain, callbacks=my_callbacks)
    return model

def compile_model(model):
    model.compile(metrics=['accuracy'])
    return model

def tune_run_random_forest_phase_two():

    data_dir = '/home/debian/DATA/'
    data = InstantiateData(data_dir)
    data = DimXNumCats(data)
    data = ConductSklearnStandardScaling(data, seed_value=1234)
    data = ReformatData(data, batch_size=14000)

    rf_two_better_default_dir = CreateRfTwoDirectory(results_dir='/home/debian/RESULTS/')
    nn_headless_model = create_model()


    model_better_default = tfdf.keras.RandomForestModel(preprocessing=nn_headless_model,
                                                            hyperparameter_template='better_default@v1')

    model_better_default = compile_model(model_better_default)
    model_better_default = fit_model(model_better_default, data, rf_two_better_default_dir)

    OutputModel(rf_two_better_default_dir, df_and_nn_model)

    test_results = model_better_default.evaluate(data.xtest, data.ytest, verbose=1)

    stutter_test = np.array(data.ytest).astype(np.int32)
    stutter_preds = np.argmax(model_better_default.predict(data.xtest), axis=1)

    num_classes = len(np.unique(stutter_preds))

    class_report = classification_report(stutter_test, stutter_preds)
    class_report = pd.DataFrame([class_report]).transpose()
    conf_mat = confusion_matrix(stutter_test, stutter_preds)
    conf_mat = pd.DataFrame(conf_mat)
    test_results = pd.DataFrame([test_results])

    [mse, bic, aic] = get_mse_bic_aic(stutter_test, stutter_preds, model_better_default)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        roc_auc = compute_roc_and_auc(stutter_test, stutter_preds, num_classes)

    summary_stats = pd.concat([roc_auc, mse, bic, aic], axis=1).T.reset_index()

    os.chdir(rf_two_better_default_dir.rf_tf_better_default_two_layer_results)
    test_results.to('test_results.csv', index=False)
    summary_stats.to_csv('summary_stats.csv', index=False)
    pd.DataFrame(conf_mat).to_csv('conf_mat.csv', index=False)
    pd.DataFrame(class_report).to_csv('class_report.csv', index=False)

    time_delta = datetime.now() - start_time

    exit_message = 'RF BETTER DEFAULT TWO LAYER PARAMETERISED RAN SUCCESSFULLY IN: ' + str(time_delta)
    sys.exit(exit_message)

# RUN_ALL
tune_run_random_forest_phase_two()

