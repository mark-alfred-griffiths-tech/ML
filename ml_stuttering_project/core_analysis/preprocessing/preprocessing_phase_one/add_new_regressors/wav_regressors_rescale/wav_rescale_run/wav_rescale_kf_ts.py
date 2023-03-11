#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import pathlib
import sys
from skimage.transform import resize

from datetime import datetime


def set_summary_stats_dir():
    summary_stats_dir = '/home/markgreenneuroscience_gmail_com/DATA/AUDIO/SUMMARY_STATS'
    return summary_stats_dir


def create_subfolder(summary_stats_dir, subfolder):
    subfolder = pathlib.Path.home().joinpath(summary_stats_dir, subfolder)
    if subfolder.exists():
        pass
    else:
        os.makedirs(subfolder)
    subfolder = str(subfolder)
    return subfolder


def get_wav_files(summary_stats_dir):
    os.chdir(summary_stats_dir)
    list_of_wav_files = os.listdir(summary_stats_dir)
    list_of_wav_files = pd.DataFrame(list_of_wav_files)
    list_of_wav_files.columns = ['FILE_LIST']
    return list_of_wav_files


def get_file_lists(summary_stats_dir, string_to_search):
    list_of_wav_files = get_wav_files(summary_stats_dir)
    list_of_selected_wav_files = list_of_wav_files[list_of_wav_files['FILE_LIST'].str.contains(string_to_search)]
    list_of_selected_wav_files = list_of_selected_wav_files.reset_index(drop=True)
    return list_of_selected_wav_files


def get_wav(list_of_selected_wav_files, name_of_wav_file):
    name = list_of_selected_wav_files[list_of_selected_wav_files['FILE_LIST'].str.contains(name_of_wav_file)]
    if len(name) == 0:
        sys.exit('FILE NOT FOUND')
    name = name.iloc[0][0]
    wav = pd.read_csv(name)
    return wav


def shrink(wav, length_of_wav_file_in_matrix):
    wav_new = pd.DataFrame([])
    for key, value in wav.iteritems():
        temp = value.to_numpy() / value.abs().max()  # normalize
        resampled = resize(temp, (length_of_wav_file_in_matrix, 1),
                           mode='edge') * value.abs().max()  # de-normalize
        wav_new[key] = resampled.flatten()
    return wav_new


def run_all_rescale_wav_kf_ts() -> None:
    start_time = datetime.now()
    name_of_wav_file = str(sys.argv[1]) + '.wav'
    length_of_wav_file_in_matrix = int(sys.argv[2])
    string_to_search = str(sys.argv[3])
    wav_folder = str(sys.argv[4])
    summary_stats_dir = set_summary_stats_dir()
    subfolder = create_subfolder(summary_stats_dir, wav_folder)
    list_of_selected_wav_files = get_file_lists(summary_stats_dir, string_to_search)
    os.chdir(summary_stats_dir)
    wav = get_wav(list_of_selected_wav_files, name_of_wav_file)
    wav = wav.reset_index(drop=True)
    wav_new = shrink(wav, length_of_wav_file_in_matrix)
    os.chdir(subfolder)
    name = list_of_selected_wav_files[list_of_selected_wav_files['FILE_LIST'].str.contains(name_of_wav_file)].iloc[0][0]
    rescaled_name = 'rescaled_' + name
    wav_new.to_csv(rescaled_name, index=False)
    time_delta = (datetime.now() - start_time)
    exit_message = '{0} RESCALE COMPLETED SUCCESSFULLY IN: {1}'.format(str(name), str(time_delta))
    sys.exit(exit_message)


# RUN_ALL
run_all_rescale_wav_kf_ts()
