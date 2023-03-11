#!/usr/bin/env python
# coding: utf-8
from datetime import datetime
import os
import pandas as pd
import sys


def set_wav_roots():
    """
    Set root directories for wav files
    """
    master_rescaled_audio_fd_root = get_rescaled_audio_fd_dir()
    master_rescaled_audio_ts_root = get_rescaled_audio_ts_dir()
    master_rescaled_audio_kf_regular_root = get_rescaled_audio_kf_regular_dir()
    master_rescaled_audio_kf_fourier_root = get_rescaled_audio_kf_fourier_dir()
    return [master_rescaled_audio_fd_root, master_rescaled_audio_ts_root,
            master_rescaled_audio_kf_regular_root, master_rescaled_audio_kf_fourier_root]


def get_rescaled_audio_fd_dir():
    """
    :return: return rescaled wav audio fd dir
    """
    master_regular_rescaled_audio_fd_root = os.path.join(
        '/home/markgreenneuroscience_gmail_com/DATA/AUDIO/SUMMARY_STATS/WAV_FD_RESCALED')
    return master_regular_rescaled_audio_fd_root


def get_rescaled_audio_ts_dir():
    """
    :return: return rescaled wav audio ts dir
    """
    master_regular_rescaled_audio_ts_root = os.path.join(
        '/home/markgreenneuroscience_gmail_com/DATA/AUDIO/SUMMARY_STATS/WAV_TS_RESCALED')
    return master_regular_rescaled_audio_ts_root


def get_rescaled_audio_kf_regular_dir():
    """
    :return: return rescaled wav audio kf fourier dir
    """
    master_regular_rescaled_audio_kf_regular_root = os.path.join(
        '/home/markgreenneuroscience_gmail_com/DATA/AUDIO/SUMMARY_STATS/WAV_KF_REGULAR_RESCALED')
    return master_regular_rescaled_audio_kf_regular_root


def get_rescaled_audio_kf_fourier_dir():
    """
    :return: return rescaled wav audio kf fourier dir
    """
    master_regular_rescaled_audio_kf_fourier_root = os.path.join(
        '/home/markgreenneuroscience_gmail_com/DATA/AUDIO/SUMMARY_STATS/WAV_KF_FOURIER_RESCALED')
    return master_regular_rescaled_audio_kf_fourier_root


def set_wav_file_list_root():
    file_list_root = '/home/markgreenneuroscience_gmail_com/DATA/AUDIO/add_new_regressors/assemble'
    return file_list_root


def get_lists_of_wav_files(file_list_root):
    os.chdir(file_list_root)
    full_list_of_files = pd.read_csv('assemble_cf.txt')
    derivatives_list = pd.DataFrame(full_list_of_files['DERIVATIVES'], columns=['DERIVATIVES'])
    regular_list = pd.DataFrame(full_list_of_files['REGULAR'], columns=['REGULAR'])
    fourier_list = pd.DataFrame(full_list_of_files['FOURIER'], columns=['FOURIER'])
    taylor_list = pd.DataFrame(full_list_of_files['TAYLOR'], columns=['TAYLOR'])
    return [derivatives_list, regular_list, fourier_list, taylor_list]


def concatenate_loose_wav_files(root, list_of_files):
    """
    :param root: dir containing list_of_files for wav
    :param list_of_files: dataframe containing files to concatenate
    :return: dataframe of concatenated files
    """
    os.chdir(root)
    [r, _] = list_of_files.shape
    full_wav = pd.DataFrame([])
    for i in range(r):
        name = list_of_files.iloc[i][0]
        one = pd.read_csv(name)
        full_wav = pd.concat([full_wav, one], axis=0)
        sys.stdout.write('WAV: ' + str(name))
    full_wav = full_wav.reset_index(drop=True)
    return full_wav


def concatenate_wav_regressors_stage_one():
    """
    Concatenate matrix regressors into subgroups
    """
    file_list_root = set_wav_file_list_root()
    [master_rescaled_audio_fd_root, master_rescaled_audio_ts_root,
     master_rescaled_audio_kf_regular_root, master_rescaled_audio_kf_fourier_root] = set_wav_roots()
    [derivatives_list, regular_list, fourier_list, taylor_list] = get_lists_of_wav_files(file_list_root)
    full_wav_kf_regular = concatenate_loose_wav_files(master_rescaled_audio_kf_regular_root,
                                                      regular_list)
    full_wav_kf_fourier = concatenate_loose_wav_files(master_rescaled_audio_kf_fourier_root,
                                                      fourier_list)
    full_wav_fd = concatenate_loose_wav_files(master_rescaled_audio_fd_root, derivatives_list)
    full_wav_ts = concatenate_loose_wav_files(master_rescaled_audio_ts_root, taylor_list)
    sys.stdout.write("CONCATENATED WAV REGRESSORS STAGE ONE\n")
    return [full_wav_kf_regular, full_wav_kf_fourier, full_wav_fd, full_wav_ts]


def concatenate_wav_regressors_stage_two(full_wav_fd, full_wav_ts, full_wav_kf_regular, full_wav_kf_fourier):
    """
    Concatenate subgroups of wav regressors
    :param full_wav_fd:
    :param full_wav_ts:
    :param full_wav_kf_regular:
    :param full_wav_kf_fourier:
    """
    wav_fd_ts_kf = pd.concat([full_wav_fd, full_wav_ts, full_wav_kf_regular, full_wav_kf_fourier], axis=1)
    wav_fd_ts_kf = wav_fd_ts_kf.reset_index(drop=True)
    sys.stdout.write('WAV CONCATENATED\n')
    return wav_fd_ts_kf


def get_rescaled_matrix_ts_dir():
    """
    :return: return rescaled matrix ts dir
    """
    master_matrix_ts_root = os.path.join(
        '/home/markgreenneuroscience_gmail_com/DATA/STATS_FOLDER')
    return master_matrix_ts_root


def get_rescaled_matrix_fd_dir():
    """
    :return: return rescaled matrix kf regular dir
    """
    master_matrix_fd_root = os.path.join(
        '/home/markgreenneuroscience_gmail_com/DATA/STATS_FOLDER')
    return master_matrix_fd_root


def get_rescaled_matrix_kf_regular_dir():
    """
    :return: return rescaled matrix kf regular dir
    """
    master_matrix_kf_regular_root = os.path.join(
        '/home/markgreenneuroscience_gmail_com/DATA/STATS_FOLDER/')
    return master_matrix_kf_regular_root


def get_rescaled_matrix_kf_fourier_dir():
    """
    :return: return rescaled matrix kf fourier dir
    """
    master_matrix_kf_fourier_root = os.path.join(
        '/home/markgreenneuroscience_gmail_com/DATA/STATS_FOLDER')
    return master_matrix_kf_fourier_root


def get_matrix_files(root, string_to_search="default"):
    """
    :param root: dir to search for files with string to search in their title
    :param string_to_search: string in title of files to select
    :return: dataframe of files in root dir with string_to_search in their title
    """
    os.chdir(root)
    listmatrix = os.listdir(root)
    listmatrix = pd.DataFrame(listmatrix)
    listmatrix.columns = ['FILE_LIST']
    list_of_files = listmatrix[listmatrix['FILE_LIST'].str.contains(string_to_search)]
    list_of_files = list_of_files.reset_index(drop=True)
    return list_of_files


def get_matrix_fd_kf_file_lists(master_matrix_fd_root, master_matrix_kf_regular_root, master_matrix_kf_fourier_root):
    """
    Search root directories for file lists conforming to relevant search strings
    :param master_matrix_fd_root:
    :param master_matrix_kf_regular_root:
    :param master_matrix_kf_fourier_root:
    :return: Lists of files
    """
    list_of_matrix_fd_files = get_matrix_files(master_matrix_fd_root, string_to_search='derivatives_full.csv')
    list_of_matrix_kf_regular_files = get_matrix_files(master_matrix_kf_regular_root,
                                                       string_to_search='fourier_kf_predictions_full.csv')
    list_of_matrix_kf_fourier_files = get_matrix_files(master_matrix_kf_fourier_root,
                                                       string_to_search='regular_kf_predictions_full.csv')
    sys.stdout.write('GOT LISTS OF FILES FOR MATRIX')
    return [list_of_matrix_fd_files, list_of_matrix_kf_regular_files, list_of_matrix_kf_fourier_files]


def concatenate_loose_matrix_files(root, list_of_files):
    """
    :param root: dir containing list_of_files for matrix
    :param list_of_files: dataframe containing files to concatenate
    :return: dataframe of concatenated files
    """
    os.chdir(root)
    [r, _] = list_of_files.shape
    full_matrix = pd.DataFrame([])
    for i in range(r):
        name = list_of_files.iloc[i][0]
        one = pd.read_csv(name)
        full_matrix = pd.concat([full_matrix, one], axis=1)
        sys.stdout.write('MATRIX: ' + str(name))
    full_matrix = full_matrix.reset_index(drop=True)
    return full_matrix


def set_matrix_roots():
    """
    Set root directories for matrix files
    """
    master_matrix_fd_root = get_rescaled_matrix_fd_dir()
    master_matrix_kf_regular_root = get_rescaled_matrix_kf_regular_dir()
    master_matrix_kf_fourier_root = get_rescaled_matrix_kf_fourier_dir()
    return [master_matrix_fd_root, master_matrix_kf_regular_root, master_matrix_kf_fourier_root]


def concatenate_matrix_regressors_stage_one():
    """
    Concatenate matrix regressors into subgroups
    """
    [master_matrix_fd_root, master_matrix_kf_regular_root, master_matrix_kf_fourier_root] = set_matrix_roots()
    [list_of_matrix_fd_files, list_of_matrix_kf_regular_files, list_of_matrix_kf_fourier_files] = \
        get_matrix_fd_kf_file_lists(master_matrix_fd_root, master_matrix_kf_regular_root, master_matrix_kf_fourier_root)
    full_matrix_fd = concatenate_loose_matrix_files(master_matrix_fd_root, list_of_matrix_fd_files)
    full_matrix_kf_regular = concatenate_loose_matrix_files(master_matrix_kf_regular_root,
                                                            list_of_matrix_kf_regular_files)
    full_matrix_kf_fourier = concatenate_loose_matrix_files(master_matrix_kf_fourier_root,
                                                            list_of_matrix_kf_fourier_files)
    sys.stdout.write('CONCATENATED MATRIX REGRESSORS STAGE ONE')
    return [full_matrix_fd, full_matrix_kf_regular, full_matrix_kf_fourier]


def concatenate_matrix_regressors_stage_two(full_matrix_fd, full_matrix_kf_regular, full_matrix_kf_fourier):
    """
    Concatenate subgroups of matrix regressors
    :param full_matrix_fd:
    :param full_matrix_kf_regular:
    :param full_matrix_kf_fourier:
    """
    matrix_fd_kf = pd.concat([full_matrix_fd, full_matrix_kf_regular, full_matrix_kf_fourier], axis=1)
    matrix_fd_kf = matrix_fd_kf.reset_index(drop=True)
    sys.stdout.write('MATRIX CONCATENATED')
    return matrix_fd_kf


def concatenate_into_matrix_two(mf, matrix_fd_kf, wav_fd_ts_kf):
    """
    Take in matrix one and concatenated subgroups of matrix and wav regressors and output mf2
    :param mf:
    :param matrix_fd_kf:
    :param wav_fd_ts_kf:
    :return: mf2: matrix two
    """
    mf2 = pd.concat([mf, matrix_fd_kf, wav_fd_ts_kf], axis=1)
    sys.stdout.write('MATRIX AND WAV CONCATENATED')
    mf2.columns = mf2.columns.str.replace(r"[()]", "_")
    mf2 = mf2.loc[:, ~mf2.columns.str.contains('Unnamed')]
    return mf2


def set_master_feature_root():
    """
    :return: root for master.csv
    """
    master_feature_root = os.path.join('/home/markgreenneuroscience_gmail_com/DATA')
    return master_feature_root


def get_master_features():
    """
    :return: master.csv
    """
    master_feature_root = set_master_feature_root()
    os.chdir(master_feature_root)
    mf = pd.read_csv('speech_features.csv')
    sys.stdout.write('GOT MASTER FEATURES')
    return mf


def fully_assemble_matrix_two():
    """
    :return: return assembled master2 dataframe, with KF and TS for matrix, and rescaled KF and TS for wav
    """
    [full_matrix_fd, full_matrix_kf_regular, full_matrix_kf_fourier] = concatenate_matrix_regressors_stage_one()
    [full_wav_kf_regular, full_wav_kf_fourier, full_wav_fd, full_wav_ts] = concatenate_wav_regressors_stage_one()
    matrix_fd_kf = concatenate_matrix_regressors_stage_two(full_matrix_fd, full_matrix_kf_regular,
                                                           full_matrix_kf_fourier)
    wav_fd_ts_kf = concatenate_wav_regressors_stage_two(full_wav_fd, full_wav_ts, full_wav_kf_regular,
                                                        full_wav_kf_fourier)

    mf = get_master_features()
    mf2 = concatenate_into_matrix_two(mf, matrix_fd_kf, wav_fd_ts_kf)
    return mf2


def output(mf2):
    """
    :param mf2: master2 dataframe containing new KF and TS features
    """
    master_feature_root = set_master_feature_root()
    os.chdir(master_feature_root)
    mf2.to_csv('speech_features_kf_ts.csv', index=False)
    sys.stdout.write('MASTER 2 SAVED')


def run_all_assemble_matrix() -> None:
    """
    Run full assembly and output
    """
    start_time = datetime.now()
    mf2 = fully_assemble_matrix_two()
    output(mf2)
    time_delta = (datetime.now() - start_time)
    exit_message = 'assemble COMPLETED SUCCESSFULLY IN: ' + str(time_delta)
    sys.exit(exit_message)


# RUN ALL
run_all_assemble_matrix()
