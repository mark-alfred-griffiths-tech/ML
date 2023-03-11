#!/usr/bin/env python3
# coding: utf-8
from datetime import datetime
import os
import pandas as pd
import sys


def set_root():
    master_feature_root = os.path.join('/home/markgreenneuroscience_gmail_com/DATA/MASTER_FEATURE')
    return master_feature_root


def get_master_features():
    master_feature_root = set_root()
    os.chdir(master_feature_root)
    mf = pd.read_csv('speech_features_kf_ts.csv')
    return mf


def edit_titles(mf):
    mf.columns = mf.columns.str.lower()
    return mf


def move_stutter(mf):
    mf_stutter = mf['stutter']
    mf = mf.drop(columns=['stutter'])
    mf_stutter = pd.DataFrame(mf_stutter)
    mf_stutter.columns = ['stutter']
    mf = pd.concat([mf, mf_stutter], axis=1)
    return mf


def drop_unnamed(mf):
    mf = mf.loc[:, ~mf.columns.str.contains('Unnamed', regex=True)]
    return mf


def output(mf, master_feature_root):
    os.chdir(master_feature_root)
    mf.to_csv('master2.csv', index=False)


def run_all_correct_matrix() -> None:
    start_time = datetime.now()
    master_feature_root = set_root()
    mf = get_master_features()
    mf = edit_titles(mf)
    mf = move_stutter(mf)
    mf = drop_unnamed(mf)
    output(mf, master_feature_root)
    time_delta = (datetime.now() - start_time)
    exit_message = 'CORRECT COMPLETED SUCCESSFULLY IN: ' + str(time_delta)
    sys.exit(exit_message)


# RUN_ALL
run_all_correct_matrix()
