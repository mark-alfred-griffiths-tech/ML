#!/usr/bin/env python
# coding: utf-8
import numpy as np
import os
import pandas as pd
import pathlib
from pykalman import KalmanFilter
import sys


def set_root():
    root = '/home/markgreenneuroscience_gmail_com/DATA/MASTER_FEATURE'
    return root


def output_folder(root):
    deriv_path = pathlib.Path.home().joinpath(root, 'STATS_FOLDER')
    if deriv_path.exists():
        pass
    else:
        os.makedirs(deriv_path)
    deriv_path = str(deriv_path)
    return deriv_path


def load_data(root):
    os.chdir(root)
    data = pd.read_csv('speech_features.csv')
    # DROP MASTER_IDX, SESS_IDX, SPEAKER_ID, SESS_ID COLUMNS
    data = data.drop(['master_idx', 'sess_idx', 'speaker_id', 'sess_id'], axis=1)
    # DROP STUTTER COLUMN
    data = data.drop(columns=['stutter'])
    return data


def normalise(data):
    if data.std() == 0:
        sys.exit('STD of 0')
    else:
        one = (data - data.mean()) / data.std()
    return one


def run_preparation(matrix_column):
    root = set_root()
    data = load_data(root)
    name = data.columns[matrix_column]
    data = data.iloc[:, matrix_column]
    deriv_path = output_folder(root)
    one = normalise(data)
    one = np.array(one)
    return [name, one, deriv_path]


def fourier_transform(signal):
    # TAKES ONLY REAL COMPONENT OF CALCULATION
    fourier_transform_signal = np.real(np.fft.fft(signal))
    return fourier_transform_signal


def kalman_filter(one, dt):
    # TIME_STEP
    dt = int(dt)
    one = one[~np.isnan(one)]
    # INITIAL MEAN
    x0 = one[0]
    # INITIAL COVARIANCE
    p0 = 0
    n_time_steps = len(one)
    n_dim_state = 1
    filtered_state_means = np.zeros((n_time_steps, n_dim_state)).astype(dtype=np.float32)
    filtered_state_covariances = np.zeros((n_time_steps, n_dim_state, n_dim_state)).astype(dtype=np.float32)
    # KALMAN FILTER INITIALISATION
    kf = KalmanFilter()
    # ITERATIVE UPDATE FOR EACH TIME STEP
    t = 0
    while t < n_time_steps:
        if t == 0:
            filtered_state_means[t] = x0
            filtered_state_covariances[t] = p0
        elif t != 0:
            filtered_state_means[t], filtered_state_covariances[t] = (
                kf.filter_update(
                    filtered_state_means[t - 1],
                    filtered_state_covariances[t - 1],
                    observation=one[t])
            )
        t = t + dt
    return filtered_state_means


def run_along_list_dt(list_dt, one, name, modality):
    kf_predictions_full = pd.DataFrame([])
    for i in range(0, len(list_dt)):
        dt = list_dt[i]
        filtered_state_means = kalman_filter(one, dt)
        filtered_state_means = pd.DataFrame([filtered_state_means.squeeze()]).T
        filtered_state_means.columns = [str(name) + '_' + str(modality) + '_kf_' + str(dt)]
        kf_predictions_full = pd.concat([kf_predictions_full, filtered_state_means], axis=1)
    return [kf_predictions_full]


def write_output(name, deriv_path, kf_full, modality):
    os.chdir(deriv_path)
    name_kf_predictions_full = str(name) + str(modality) + 'kf_predictions_full.csv'
    kf_full.to_csv(name_kf_predictions_full, index=None)
