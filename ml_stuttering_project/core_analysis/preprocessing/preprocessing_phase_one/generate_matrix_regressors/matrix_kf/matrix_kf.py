#!/usr/bin/env python
# coding: utf-8
from datetime import datetime
from matrix_kf_functions import *


def run_all_matrix_kf() -> None:
    start_time = datetime.now()

    # GET feat_num
    feat_num = int(float(sys.argv[1]))

    # RUN_SCRIPT
    [name, one, deriv_path] = run_preparation(feat_num)
    list_dt = [2, 3, 4, 5]
    [kf_predictions_regular_full] = run_along_list_dt(list_dt, one, name, 'regular')
    fourier_transform_one = fourier_transform(one)
    [kf_predictions_fourier_full] = run_along_list_dt(list_dt, fourier_transform_one, name, 'ff')

    write_output(name, deriv_path, kf_predictions_regular_full, '_regular_')
    write_output(name, deriv_path, kf_predictions_fourier_full, '_fourier_')

    time_delta = datetime.now() - start_time
    exit_message = str(name) + ' MATRIX KF RAN SUCCESSFULLY: ' + str(time_delta)
    sys.exit(exit_message)


# RUN_ALL
run_all_matrix_kf()
