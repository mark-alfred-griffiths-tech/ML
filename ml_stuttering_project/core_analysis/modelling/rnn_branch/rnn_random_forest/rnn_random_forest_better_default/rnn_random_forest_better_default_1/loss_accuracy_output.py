#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def one_hot(y_metric):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y_metric)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    y_metric_oh = onehot_encoder.fit_transform(integer_encoded)
    return y_metric_oh


def compute_roc_and_auc(stutter_test, stutter_preds, num_classes):
    # COMPUTE ROC CURVE AND AREA FOR EACH CLASSES
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(0, num_classes):
        fpr[i], tpr[i], _ = roc_curve(one_hot(stutter_preds)[:, i], one_hot(stutter_test)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # COMPUTE MICRO-AVERAGE ROC CURVE AND AREA
    fpr['micro'], tpr['micro'], _ = roc_curve(one_hot(stutter_preds).ravel(), one_hot(stutter_test).ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
    roc_auc = roc_auc['micro']
    roc_auc = pd.DataFrame([roc_auc])
    roc_auc.columns = ['ROC_AUC']
    return roc_auc


def get_num_params(model):
    num_params = model.count_params()
    return num_params


def calculate_aic(n, mse, num_params):
    if mse == 0:
        aic = 2 * num_params
    else:
        aic = n * np.log(mse) + 2 * num_params
    return aic


def calculate_bic(n, mse, num_params):
    if mse == 0:
        bic = num_params * np.log(n)
    else:
        bic = n * np.log(mse) + num_params * np.log(n)
    return bic


def get_mse_bic_aic(stutter_test, stutter_preds, model):
    num_params = get_num_params(model)
    mse = mean_absolute_error(stutter_test, stutter_preds)
    aic = calculate_aic(len(stutter_test), mse, num_params)
    bic = calculate_bic(len(stutter_test), mse, num_params)
    mse = pd.DataFrame([mse])
    mse.columns = ['MSE']
    bic = pd.DataFrame([bic])
    bic.columns = ['BIC']
    aic = pd.DataFrame([aic])
    aic.columns = ['AIC']
    return [mse, bic, aic]
