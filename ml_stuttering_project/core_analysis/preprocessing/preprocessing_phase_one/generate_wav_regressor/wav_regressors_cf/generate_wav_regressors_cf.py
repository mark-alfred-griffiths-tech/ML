#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
from numpy import savetxt
import pandas as pd
import sys


def set_root():
    root = '/home/markgreenneuroscience_gmail_com/DATA/AUDIO'
    return root


def set_control_directory():
    control_directory = '/home/markgreenneuroscience_gmail_com/TS_KF_TRANSFORMS/generate_wav_regressor' \
                        '/wav_regressors_cf'
    return control_directory


def get_number_of_wav(root, control_directory):
    os.chdir(root)
    file_list = os.listdir(root)
    file_list = pd.DataFrame(file_list)
    file_list.columns = ['FILE_LIST']
    file_list = file_list[file_list['FILE_LIST'].str.contains('.wav')].reset_index(drop=True)
    file_list = file_list.reset_index()
    file_list = np.array(file_list['index']).reshape(-1, 1).astype(int)
    os.chdir(control_directory)
    savetxt('WAV_CONTROL_FILE.txt', file_list, fmt='%i', delimiter=',')


def run_all_generate_wav_regressor_cf() -> None:
    root = set_root()
    control_directory = set_control_directory()
    get_number_of_wav(root, control_directory)
    sys.exit('WAV CONTROL SCRIPT GENERATOR PROCESSED CORRECTLY')


# RUN_ALL
run_all_generate_wav_regressor_cf()
