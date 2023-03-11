#!/usr/bin/env python
# coding: utf-8
import sys
from pykalman import KalmanFilter
import pandas as pd
import numpy as np
import pathlib
import scipy.io.wavfile as wav
import os


def set_root():
    root = '/home/markgreenneuroscience_gmail_com/DATA'
    return root


def output_folder(root):
    os.chdir(root)
    file_directory = pathlib.Path.home().joinpath(root, 'AUDIO')
    os.chdir(file_directory)

    deriv_path = pathlib.Path.home().joinpath(file_directory, 'SUMMARY_STATS')
    if deriv_path.exists():
        pass
    else:
        os.makedirs(deriv_path)
    return [deriv_path, file_directory]


def get_file_list(file_directory):
    file_list = os.listdir(file_directory)
    file_list = pd.DataFrame(file_list)
    file_list.columns = ['FILE_LIST']
    file_list = file_list[file_list['FILE_LIST'].str.contains('.wav')]
    return file_list


def check_is_empty(data):
    if len(data) == 0:
        x = np.empty((0, 0), dtype=object)
        if x.size == 0:
            sys.exit('EMPTY DATA')
        else:
            pass


def normalise(data):
    check_is_empty(data)
    if np.std(np.array(data), axis=0) == 0:
        sys.exit('STD of 0')
    else:
        data_normalised = (data - data.mean()) / data.std()
    return data_normalised


def run_preparation(i):
    root = set_root()
    file_directory = pathlib.Path.home().joinpath(root, 'AUDIO')
    file_list = get_file_list(file_directory)
    file_one = file_list.iloc[i]
    file_path = pathlib.Path.home().joinpath(root, 'AUDIO', file_one[0])
    label_path = pathlib.Path.home().joinpath(root, 'LABELS', file_one[0])

    file_path = str(file_path)
    label_path = str(label_path)
    if os.path.exists(label_path[:-4] + '_syll.csv'):
        pass
    elif os.path.exists(label_path[:-4] + '_ortho.csv'):
        pass
    elif os.path.exists(label_path[:-4] + '_phono.csv'):
        pass
    elif os.path.exists(label_path[:-4] + '_pw.csv'):
        pass
    elif os.path.exists(label_path[:-4] + '_word.csv'):
        pass
    else:
        sys.exit("Can't find accompanying label file! It is presumably missing")

    [_, signal] = wav.read(file_path)
    signal = np.array(signal).ravel()
    return signal


def fourier_transform(signal):
    fourier_transform_signal = np.real(np.fft.fft(signal))
    return fourier_transform_signal


def kalman_filter(one, dt):
    # TIME_STEP
    check_is_empty(one)
    dt = dt
    # INITIAL MEAN
    x0 = one[0]
    # INITIAL COVARIANCE
    p0 = 0
    n_timesteps = len(one)
    n_dim_state = 1
    filtered_state_means = np.zeros((n_timesteps, n_dim_state))
    filtered_state_covariances = np.zeros((n_timesteps, n_dim_state, n_dim_state))
    # KALMAN FILTER INITIALISATION
    kf = KalmanFilter()
    t = 0
    while t < n_timesteps:
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
    return [filtered_state_means, filtered_state_covariances]


def add_in_labels(dt, kalman):
    name = dt
    name_df = pd.DataFrame([name])
    name_df.columns = ['DT']
    [j, _] = kalman.shape
    name_df = name_df.loc[name_df.index.repeat(j)].reset_index(drop=True)
    full_df = pd.concat([name_df, kalman], axis=1)
    full_df = full_df.loc[:, ~full_df.columns.duplicated()]
    return full_df


def drop_first_column(matrix):
    matrix = matrix.drop(matrix.columns[0], axis=1)
    return matrix


def squeeze(x):
    x = np.array(x).squeeze()
    return x


def run_kalman_on_single_wav(data, modality):
    list_dt = [2, 3, 4, 5]
    filtered_state_means_single_wav = pd.DataFrame([])
    filtered_state_covariances_single_wav = pd.DataFrame([])

    for i in range(0, len(list_dt)):
        dt = list_dt[i]
        dt_name_mean = 'mean_dt_' + str(modality) + '_' + str(dt)
        dt_name_cov = 'cov_dt_' + str(modality) + '_' + str(dt)

        [filtered_state_means, filtered_state_covariances] = kalman_filter(data, dt)
        filtered_state_means = squeeze(filtered_state_means).reshape(-1, 1)
        filtered_state_covariances = squeeze(filtered_state_covariances).reshape(-1, 1)
        filtered_state_means = pd.DataFrame(filtered_state_means)
        filtered_state_means.columns = [dt_name_mean]
        filtered_state_covariances = pd.DataFrame(filtered_state_covariances)
        filtered_state_covariances.columns = [dt_name_cov]
        filtered_state_means = pd.DataFrame(filtered_state_means)
        filtered_state_means.columns = [dt_name_mean]
        filtered_state_covariances = pd.DataFrame(filtered_state_covariances)
        filtered_state_covariances.columns = [dt_name_cov]
        filtered_state_means_single_wav = pd.concat(
            [filtered_state_means_single_wav, pd.DataFrame(filtered_state_means)], axis=1)
        filtered_state_covariances_single_wav = pd.concat(
            [filtered_state_covariances_single_wav, pd.DataFrame(filtered_state_covariances)], axis=1)
    return [filtered_state_means_single_wav, filtered_state_covariances_single_wav]


def concat_vertically(kf_predictions_full, kf_covariances_full, kf_predictions_single, kf_covariances_single):
    kf_predictions_full = pd.concat([kf_predictions_full, kf_predictions_single])
    kf_covariances_full = pd.concat([kf_covariances_full, kf_covariances_single])
    return [kf_predictions_full, kf_covariances_full]


def rescale(data, variable):
    variable_rescaled = (variable * data.std()[0]) + data.mean()[0]
    return variable_rescaled


def rescale_master(data, kf_predictions_single, kf_covariances_single, kf_predictions_full, kf_covariances_full):
    kf_predictions_single = rescale(data, kf_predictions_single)
    kf_covariances_single = rescale(data, kf_covariances_single)
    kf_predictions_full = rescale(data, kf_predictions_full)
    kf_covariances_full = rescale(data, kf_covariances_full)
    return [kf_predictions_single, kf_covariances_single, kf_predictions_full, kf_covariances_full]


def write_output(name, modality, kf_predictions_single, kf_covariances_single, deriv_path):
    name_modality = name + '_' + modality
    os.chdir(deriv_path)
    kf_predictions_single.to_csv(name_modality + '_kf_predictions.csv', index=False)
    kf_covariances_single.to_csv(name_modality + '_kf_covariances.csv', index=False)


def run_kf_script(name, modality, data_normalised, deriv_path):
    [filtered_state_means_single_wav, filtered_state_covariances_single_wav] = run_kalman_on_single_wav(data_normalised,
                                                                                                        modality)
    write_output(name, modality, filtered_state_means_single_wav, filtered_state_covariances_single_wav, deriv_path)
