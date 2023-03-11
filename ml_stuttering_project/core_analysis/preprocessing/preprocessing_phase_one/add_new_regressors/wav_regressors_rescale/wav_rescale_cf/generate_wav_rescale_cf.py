#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import sys


def set_root():
    root = '/home/markgreenneuroscience_gmail_com/DATA/MASTER_FEATURE'
    return root


def set_control_directory():
    control_directory = '/home/markgreenneuroscience_gmail_com/TS_KF_TRANSFORMS/add_new_regressors' \
                        '/wav_regressors_rescale/wav_rescale_cf '
    return control_directory


def get_list(root):
    os.chdir(root)
    df = pd.DataFrame(os.listdir())
    df.columns = ['FILE_LIST']
    df = df[df['FILE_LIST'].str.contains('.csv')]
    df = df[~df['FILE_LIST'].str.contains('control_file_name.csv')].reset_index(drop=True)
    df = df[~df['FILE_LIST'].str.contains('DF_FULL.csv')].reset_index(drop=True)
    return df


def get_df_full(df):
    df_full = pd.DataFrame([])
    [r, _] = df.shape
    for i in range(r):
        name = df.iloc[i][0]
        df_one = pd.read_csv(name)
        [rows, _] = df_one.shape
        df1_rows = pd.DataFrame([rows])
        df1_rows.columns = ['ROWS']
        sessid = pd.DataFrame(set(df_one['sess_id'])).iloc[0][0].astype('int')
        df_one_name = pd.DataFrame([name])
        df_one_name.columns = ['NAME']
        df_one_sessid = pd.DataFrame([sessid])
        df_one_sessid.columns = ['SESSID']
        df_one_name_sessid_rows = pd.concat([df_one_name, df_one_sessid, df1_rows], axis=1)
        df_full = pd.concat([df_full, df_one_name_sessid_rows], axis=0)
        df_full = df_full.reset_index(drop=True)
        df_full = df_full.sort_values(by='SESSID', ascending=True).reset_index(drop=True)
    df_folders = pd.DataFrame({'STRING_TO_SEARCH': ['wav_derivatives.csv', 'wav_taylor_series.csv',
                                                    'fourier_kf_predictions.csv', 'regular_kf_predictions.csv'],
                               'FOLDER': ['WAV_FD_RESCALED', 'WAV_TS_RESCALED', 'WAV_KF_FOURIER_RESCALED',
                                          'WAV_KF_REGULAR_RESCALED']})
    df_folders['key'] = 0
    df_full['key'] = 0
    df_full = df_full.merge(df_folders, on='key', how='outer').drop('key', 1)
    return df_full


def output_result(control_directory, df_full):
    os.chdir(control_directory)
    df_full['NAME'] = df_full['NAME'].str.replace(r'.csv', '')
    df_full.to_csv('WAV_RESCALE_KF_TS_CONTROL_FILE.txt', index=False, header=None, sep='\t')


def run_all_generate_wav_kf_ts_rescale_cf() -> None:
    root = set_root()
    control_directory = set_control_directory()
    df = get_list(root)
    df_full = get_df_full(df)
    output_result(control_directory, df_full)
    sys.exit('WAV RESCALE CONTROL FILE CREATED SUCCESSFULLY')


# RUN_ALL
run_all_generate_wav_kf_ts_rescale_cf()
