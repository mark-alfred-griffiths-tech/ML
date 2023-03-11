#!/usr/bin/env python
# coding: utf-8
import sys
import pandas as pd
import numpy as np
from scipy import ndimage
import os
import math
import pathlib
import scipy.io.wavfile as wav


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
        sys.exit("STD of 0")
    else:
        data_normalised = (data - data.mean()) / data.std()
    return data_normalised


def get_x(data_normalised):
    n = np.size(data_normalised)
    x = np.linspace(0, n, n, endpoint=True)
    return x


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


def fdm(one):
    one = list(one)
    n = np.size(one)
    x = np.linspace(0, n, n, endpoint=True)
    f = one + .02 * (np.random.rand(n) - .5)
    dx = x[1] - x[0]
    dx2 = dx ** 2
    dx3 = dx ** 3
    d4 = dx ** 4
    d5 = dx ** 5
    # FIRST DERIVATIVE
    fd_estimate = ndimage.gaussian_filter1d(f, sigma=1, order=1, mode='wrap') / dx
    # SECOND DERIVATIVES
    sd_estimate = ndimage.gaussian_filter1d(f, sigma=1, order=2, mode='wrap') / dx2
    # THIRD DERIVATIVES
    thd_estimate = ndimage.gaussian_filter1d(f, sigma=1, order=3, mode='wrap') / dx3
    # FOURTH DERIVATIVE
    fod_estimate = ndimage.gaussian_filter1d(f, sigma=1, order=4, mode='wrap') / d4
    # FIFTH DERIVATIVE
    fid_estimate = ndimage.gaussian_filter1d(f, sigma=1, order=5, mode='wrap') / d5
    return [fd_estimate, sd_estimate, thd_estimate, fod_estimate, fid_estimate]


def get_middle(x):
    if (len(x) % 2) == 0:
        x = pd.DataFrame(x).loc[len(x) / 2][0]
    elif (len(x) % 2) == 1:
        x = (pd.DataFrame(x).loc[len(x) / 2 + 0.5][0]) + (pd.DataFrame(x).loc[len(x) / 2 - 0.5][0]) / 2
    return x


def get_fa(data_normalised, fd_estimate, sd_estimate, thd_estimate, fod_estimate, fid_estimate):
    data_normalised_middle = get_middle(data_normalised)
    fd_estimate_middle = get_middle(fd_estimate)
    sd_estimate_middle = get_middle(sd_estimate)
    thd_estimate_middle = get_middle(thd_estimate)
    fod_estimate_middle = get_middle(fod_estimate)
    fid_estimate_middle = get_middle(fid_estimate)
    return [data_normalised_middle, fd_estimate_middle, sd_estimate_middle, thd_estimate_middle, fod_estimate_middle,
            fid_estimate_middle]


def f_0(x):
    return x


def f_sub(a, n):
    a /= math.factorial(n)
    return a


def f_term(x, a, n):
    term = ((x - a) ** n)
    return term


def perform_taylor_series(x, fd_estimate_middle, sd_estimate_middle, thd_estimate_middle, fod_estimate_middle,
                          fid_estimate_middle):
    value = f_0(x)
    value_deriv_1 = (f_sub(fd_estimate_middle, 1) * f_term(x, fd_estimate_middle, 1))
    value_deriv_2 = (f_sub(sd_estimate_middle, 2) * f_term(x, sd_estimate_middle, 2))
    value_deriv_3 = (f_sub(thd_estimate_middle, 3) * f_term(x, thd_estimate_middle, 3))
    value_deriv_4 = (f_sub(fod_estimate_middle, 4) * f_term(x, fod_estimate_middle, 4))
    value_deriv_5 = (f_sub(fid_estimate_middle, 5) * f_term(x, fid_estimate_middle, 5))
    ts_value = value + value_deriv_1 + value_deriv_2 + value_deriv_3 + value_deriv_4 + value_deriv_5
    return ts_value


def get_x_length(x):
    len_x = len(x)
    return len_x


def find_residuals(data_normalised, ts_series):
    ts_resid = data_normalised - ts_series
    return ts_resid


def rescale(data, ts_series):
    series_rescaled = (np.array(ts_series) * np.array(data.std()) + np.array(data.mean()))
    return series_rescaled


def run_taylor_series(data_normalised):
    [fd_estimate, sd_estimate, thd_estimate, fod_estimate, fid_estimate] = fdm(data_normalised)
    ts_series = perform_taylor_series(data_normalised, fd_estimate, sd_estimate, thd_estimate, fod_estimate,
                                      fid_estimate)
    ts_resid = find_residuals(data_normalised, ts_series)
    ts_series = pd.DataFrame(ts_series)
    ts_resid = pd.DataFrame(ts_resid)
    return [ts_resid, ts_series, fd_estimate, sd_estimate, thd_estimate, fod_estimate, fid_estimate]


def fourier_transform(signal):
    # TAKES ONLY REAL COMPONENT OF CALCULATION
    fourier_transform_signal = np.real(np.fft.fft(signal))
    return fourier_transform_signal


def put_derivatives_into_dataframes(fd_estimate_ff, sd_estimate_ff, thd_estimate_ff, fod_estimate_ff, fid_estimate_ff,
                                    fd_estimate, sd_estimate, thd_estimate, fod_estimate, fid_estimate):
    derivatives_ff = pd.concat(
        [pd.DataFrame(fd_estimate_ff), pd.DataFrame(sd_estimate_ff), pd.DataFrame(thd_estimate_ff),
         pd.DataFrame(fod_estimate_ff), pd.DataFrame(fid_estimate_ff)], axis=1)
    derivatives = pd.concat(
        [pd.DataFrame(fd_estimate), pd.DataFrame(sd_estimate), pd.DataFrame(thd_estimate), pd.DataFrame(fod_estimate),
         pd.DataFrame(fid_estimate)], axis=1)
    derivatives_full = pd.concat([derivatives, derivatives_ff], axis=1)
    return derivatives_full


def put_taylor_series_and_residuals_into_dataframes(ts_series, ts_series_ff, ts_resid, ts_resid_ff):
    ts_concat = pd.concat([ts_series, ts_series_ff, ts_resid, ts_resid_ff], axis=1)
    return ts_concat


def rescale_output(data, derivatives_full, ts_concat):
    derivatives_full = rescale(data, derivatives_full)
    ts_concat = rescale(data, ts_concat)
    return [derivatives_full, ts_concat]
