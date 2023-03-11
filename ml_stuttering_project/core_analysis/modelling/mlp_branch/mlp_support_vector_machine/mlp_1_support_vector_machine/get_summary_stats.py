#!/usr/bin/env python
# coding: utf-8
from sklearn.metrics import mean_absolute_error, roc_curve, auc
import pandas as pd
import numpy as np
import sys

def compute_roc_and_auc(stutter_test, stutter_preds, num_classes):
    # COMPUTE ROC CURVE AND AREA FOR EACH CLASSES
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(0, num_classes):
        fpr[i], tpr[i], _ = roc_curve(np.array(pd.DataFrame(stutter_preds).iloc[i]), np.array(pd.DataFrame(stutter_test).iloc[i]), pos_label=1)
        roc_auc[i] = auc(fpr[i], tpr[i])
    # COMPUTE MICRO-AVERAGE ROC CURVE AND AREA
    stutter_preds[stutter_preds > 0] = 1
    stutter_test[stutter_test > 0] = 1

    fpr['micro'], tpr['micro'], _ = roc_curve(stutter_preds, stutter_test)
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
