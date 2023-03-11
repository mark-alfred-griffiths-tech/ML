#!/usr/bin/env python
# coding: utf-8
import numpy as np
import os
import pandas as pd
import sys


def set_ordered_root():
    ordered_root = '/home/markgreenneuroscience_gmail_com/TS_KF_TRANSFORMS/add_new_regressors/assemble/assemble_cf'
    return ordered_root


def read_in_ordered(ordered_root):
    os.chdir(ordered_root)
    ordered = pd.read_csv('odered_labels.csv', header=None)
    ordered.columns = ['ORDERED']
    return ordered


def edit_ordered_labels(ordered):
    rescaled = pd.DataFrame(np.repeat('rescaled_', ordered.shape[0]), columns=['RESCALED'])
    rescaled = pd.concat([rescaled, ordered], axis=1)
    ordered = pd.DataFrame(np.array(rescaled['RESCALED'] + rescaled['ORDERED']), columns=['ORDERED'])
    return ordered


def create_wav_derivatives(ordered):
    wav_derivatives_rescaled = pd.DataFrame(ordered['ORDERED'] + '.wav_derivatives.csv')
    wav_derivatives_rescaled.columns = ['DERIVATIVES']
    wav_regular_rescaled = pd.DataFrame(ordered['ORDERED'] + '.wav_regular_kf_predictions.csv')
    wav_regular_rescaled.columns = ['REGULAR']
    wav_fourier_rescaled = pd.DataFrame(ordered['ORDERED'] + '.wav_fourier_kf_predictions.csv')
    wav_fourier_rescaled.columns = ['FOURIER']
    wav_taylor_series_rescaled = pd.DataFrame(ordered['ORDERED'] + '.wav_taylor_series.csv')
    wav_taylor_series_rescaled.columns = ['TAYLOR']

    ordered_full = pd.concat([wav_derivatives_rescaled, wav_regular_rescaled,
                              wav_fourier_rescaled, wav_taylor_series_rescaled], axis=1)

    return ordered_full


def output_ordered_full(ordered_root, ordered_full):
    os.chdir(ordered_root)
    ordered_full.to_csv('assemble_cf.txt', index=None)


def run_all_assemble_cf() -> None:
    ordered_root = set_ordered_root()
    ordered = read_in_ordered(ordered_root)
    ordered = edit_ordered_labels(ordered)
    ordered_full = create_wav_derivatives(ordered)
    output_ordered_full(ordered_root, ordered_full)
    sys.exit('assemble CONTROL FILE CREATED SUCCESSFULLY')


# RUN ALL
run_all_assemble_cf()
