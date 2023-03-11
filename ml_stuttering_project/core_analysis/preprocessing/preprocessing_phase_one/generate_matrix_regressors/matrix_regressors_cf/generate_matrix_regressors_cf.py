#!/usr/bin/env python
# coding: utf-8
import numpy as np
import os
import pandas as pd
import sys


def set_root():
    root = '/home/markgreenneuroscience_gmail_com/DATA/MASTER_FEATURE'
    return root


def set_output():
    output = '/home/markgreenneuroscience_gmail_com/TS_KF_TRANSFORMS/generate_matrix_regressors/matrix_regressors_cf'
    return output


def get_csv():
    csv = pd.read_csv('speech_features.csv')
    # DROP MASTER_IDX, SESS_IDX, SPEAKER_ID, SESS_ID
    csv = csv.drop(['master_idx', 'sess_idx', 'speaker_id', 'sess_id'], axis=1)
    # DROP STUTTER COLUMN
    csv = csv.drop(['stutter'], axis=1)
    [_, cols] = csv.shape
    return cols


def get_npl(c):
    npl = np.array([])
    for i in range(0, c):
        npl = np.append(npl, [i])
    return npl


def save_npl(npl):
    np.savetxt('MATRIX_CONTROL_FILE.txt', npl, fmt='%i', delimiter=',')


def run_all_generate_matrix_regressor_cf() -> None:
    root = set_root()
    output = set_output()
    os.chdir(root)
    cols = get_csv()
    npl = get_npl(cols)
    os.chdir(output)
    save_npl(npl)
    sys.exit('MATRIX CONTROL FILE GENERATOR PROCESSED CORRECTLY')


# RUN_ALL
run_all_generate_matrix_regressor_cf()
