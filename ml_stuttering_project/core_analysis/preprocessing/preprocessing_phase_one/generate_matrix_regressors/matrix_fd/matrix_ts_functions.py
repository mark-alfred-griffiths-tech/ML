#!/usr/bin/env python
# coding: utf-8
import math
import numpy as np
from numpy import ndarray
import os
import pandas as pd
import pathlib
from scipy import ndimage
import sys


def set_root():
    root = '/home/markgreenneuroscience_gmail_com/DATA/MASTER_FEATURE'
    return root


def output_folder(root):
    deriv_path = pathlib.Path.home().joinpath(root, 'STATS_FOLDER')
    if not deriv_path.exists():
        os.makedirs(deriv_path)
    else:
        pass
    return deriv_path


def load_data(root):
    os.chdir(root)
    data = pd.read_csv('speech_features.csv')
    # DROP MASTER_IDX, SESS_IDX, SPEAKER_ID, SESS_ID COLUMNS
    data = data.drop(['master_idx', 'sess_idx', 'speaker_id', 'sess_id'], axis=1)
    # DROP STUTTER COLUMN
    data = data.drop(['stutter'], axis=1)
    return data


def normalise(data):
    if data.std() == 0:
        sys.exit("STD of 0")
    else:
        one = (data - data.mean()) / data.std()
    return one


def check_is_empty(data_normalised):
    if len(data_normalised) == 0:
        x: ndarray = np.empty((0, 0), dtype=object)
        if x.size == 0:
            sys.exit("EMPTY DATA")
        else:
            pass


def fdm(one):
    one = list(one)
    n = np.size(one)
    x = np.linspace(0, n, n, endpoint=True)
    f = one + .02 * (np.random.rand(n) - .5)
    dx = x[1] - x[0]
    dx2 = dx ** 2
    dx3 = dx ** 3
    dx4 = dx ** 4
    dx5 = dx ** 5
    # FIRST DERIVATIVE
    fd_estimate = ndimage.gaussian_filter1d(f, sigma=1, order=1, mode='wrap') / dx
    # SECOND DERIVATIVES
    sd_estimate = ndimage.gaussian_filter1d(f, sigma=1, order=2, mode='wrap') / dx2
    # THIRD DERIVATIVES
    thd_estimate = ndimage.gaussian_filter1d(f, sigma=1, order=3, mode='wrap') / dx3
    # FOURTH DERIVATIVE
    fod_estimate = ndimage.gaussian_filter1d(f, sigma=1, order=4, mode='wrap') / dx4
    # FIFTH DERIVATIVE
    fid_estimate = ndimage.gaussian_filter1d(f, sigma=1, order=5, mode='wrap') / dx5
    return [fd_estimate, sd_estimate, thd_estimate, fod_estimate, fid_estimate]


def get_middle(x):
    if (len(x) % 2) == 0:
        x = pd.DataFrame(x).loc[len(x) / 2][0]
    elif (len(x) % 2) == 1:
        x = (pd.DataFrame(x).loc[len(x) / 2 + 0.5][0]) + (pd.DataFrame(x).loc[len(x) / 2 - 0.5][0]) / 2
    return x


def get_fa(one, fd_estimate, sd_estimate, thd_estimate, fod_estimate, fid_estimate):
    one_middle = get_middle(one)
    fd_estimate_middle = get_middle(fd_estimate)
    sd_estimate_middle = get_middle(sd_estimate)
    thd_estimate_middle = get_middle(thd_estimate)
    fod_estimate_middle = get_middle(fod_estimate)
    fid_estimate_middle = get_middle(fid_estimate)
    return [one_middle, fd_estimate_middle, sd_estimate_middle, thd_estimate_middle, fod_estimate_middle,
            fid_estimate_middle]


def f_0(x):
    return x


def f_sub(a, n):
    a = a / math.factorial(n)
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


def find_residuals(data_normalised, ts_series):
    ts_resid = data_normalised - ts_series
    return ts_resid


def rescale(data, ts_series):
    series_rescaled = (ts_series * data.std()) + data.mean()
    return series_rescaled


def get_x_length(x):
    len_x = len(x)
    return len_x


def get_mse(data_normalised, ts_series):
    mse = (np.square(data_normalised - ts_series)).mean(axis=0)
    return mse


def fourier_transform(signal):
    # TAKES ONLY REAL COMPONENT OF CALCULATION
    fourier_transform_signal = np.real(np.fft.fft(signal))
    return fourier_transform_signal


def run_taylor_series(one):
    [fd_estimate, sd_estimate, thd_estimate, fod_estimate, fid_estimate] = fdm(one)
    [_, fd_estimate_middle, sd_estimate_middle, thd_estimate_middle, fod_estimate_middle,
     fid_estimate_middle] = get_fa(one, fd_estimate, sd_estimate, thd_estimate, fod_estimate, fid_estimate)
    one = np.array(one)
    ts_series = perform_taylor_series(one, fd_estimate_middle, sd_estimate_middle, thd_estimate_middle,
                                      fod_estimate_middle, fid_estimate_middle)
    ts_resid = find_residuals(one, ts_series)
    return [ts_series, ts_resid, fd_estimate, sd_estimate, thd_estimate, fod_estimate, fid_estimate]


def get_name(one):
    name = pd.DataFrame(one).columns[0]
    return name


def run_preparation(iint):
    root = set_root()
    data = load_data(root)
    name = data.columns[iint]
    data = data.iloc[:, iint]
    deriv_path = output_folder(root)
    data_normalised = normalise(data)
    check_is_empty(data_normalised)
    return [name, data_normalised, deriv_path]


def rescale_output(data, derivatives_full, ts_series, ts_series_ff, ts_resid, ts_resid_ff):
    derivatives_full = derivatives_full.loc[:, ~derivatives_full.columns.isin(['Names'])]
    derivatives_full = rescale(data, derivatives_full)
    ts_series = rescale(data, ts_series)
    ts_series_ff = rescale(data, ts_series_ff)
    ts_resid = rescale(data, ts_resid)
    ts_resid_ff = rescale(data, ts_resid_ff)
    return [derivatives_full, ts_series, ts_series_ff, ts_resid, ts_resid_ff]


def put_derivatives_into_dataframes(name, fd_estimate_ff, sd_estimate_ff, thd_estimate_ff, fod_estimate_ff,
                                    fid_estimate_ff, fd_estimate, sd_estimate, thd_estimate, fod_estimate,
                                    fid_estimate):
    derivatives_ff = pd.concat(
        [pd.DataFrame(fd_estimate_ff), pd.DataFrame(sd_estimate_ff), pd.DataFrame(thd_estimate_ff),
         pd.DataFrame(fod_estimate_ff), pd.DataFrame(fid_estimate_ff)], axis=1)

    name_first_ff_derivative = str(name) + "_First_ff_derivative"
    name_second_ff_derivative = str(name) + "_Second_ff_derivative"
    name_third_ff_derivative = str(name) + "_Third_ff_derivative"
    name_fourth_ff_derivative = str(name) + "_Fourth_ff_derivative"
    name_fifth_ff_derivative = str(name) + "_Fifth_ff_derivative"

    name_first_derivative = str(name) + "_First_ff_derivative"
    name_second_derivative = str(name) + "_Second_ff_derivative"
    name_third_derivative = str(name) + "_Third_ff_derivative"
    name_fourth_derivative = str(name) + "_Fourth_ff_derivative"
    name_fifth_derivative = str(name) + "_Fifth_ff_derivative"

    derivatives_ff.columns = [name_first_ff_derivative, name_second_ff_derivative, name_third_ff_derivative,
                              name_fourth_ff_derivative, name_fifth_ff_derivative]
    derivatives = pd.concat(
        [pd.DataFrame(fd_estimate), pd.DataFrame(sd_estimate), pd.DataFrame(thd_estimate), pd.DataFrame(fod_estimate),
         pd.DataFrame(fid_estimate)], axis=1)
    derivatives.columns = [name_first_derivative, name_second_derivative, name_third_derivative, name_fourth_derivative,
                           name_fifth_derivative]
    derivatives_single = pd.concat([derivatives, derivatives_ff], axis=1)
    return derivatives_single


def add_in_names(data_normalised, i, derivatives_full):
    name = data_normalised.iloc[i][0]
    name_df = pd.DataFrame([name])
    name_df.columns = ["Names"]
    [j, _] = derivatives_full.shape
    name_df = name_df.loc[name_df.index.repeat(j)].reset_index(drop=True)
    full_df = pd.concat([name_df, derivatives_full], axis=1)
    full_df = full_df.loc[:, ~full_df.columns.duplicated()]
    return full_df


def write_output(deriv_path, root, data, col_num, derivatives_full, ts_series, ts_series_ff, ts_resid, ts_resid_ff):
    os.chdir(deriv_path)
    name = data.columns[col_num]

    name_derivatives_full = str(name) + "_derivatives_full.csv"
    name_ts_series = str(name) + "_ts_series_dataframe.csv"
    name_ts_series_ff = str(name) + "_ts_series_ff_dataframe.csv"
    name_ts_resid = str(name) + "_ts_resid_dataframe.csv"
    name_ts_resid_ff = str(name) + "_ts_resid_ff_dataframe.csv"

    ts_series = pd.DataFrame([ts_series]).T
    ts_series.columns = [str(name) + "_TS_SERIES"]
    ts_series_ff = pd.DataFrame([ts_series_ff]).T
    ts_series_ff.columns = [str(name) + "_TS_SERIES_FF"]
    ts_resid = pd.DataFrame([ts_resid]).T
    ts_resid.columns = [str(name) + "_TS_RESID"]
    ts_resid_ff = pd.DataFrame([ts_resid_ff]).T
    ts_resid_ff.columns = [str(name) + "_TS_RESID_FF"]

    derivatives_full.to_csv(name_derivatives_full, index=False)
    ts_series.to_csv(name_ts_series, index=False)
    ts_series_ff.to_csv(name_ts_series_ff, index=False)
    ts_resid.to_csv(name_ts_resid, index=False)
    ts_resid_ff.to_csv(name_ts_resid_ff, index=False)
    os.chdir(root)
