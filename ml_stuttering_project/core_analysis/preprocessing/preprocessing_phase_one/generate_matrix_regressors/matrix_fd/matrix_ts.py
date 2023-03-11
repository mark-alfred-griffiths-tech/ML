#!/usr/bin/env python
# coding: utf-8
from datetime import datetime
from matrix_ts_functions import *


def run_all_matrix_fd() -> None:
    start_time = datetime.now()

    # GET feat_num
    feat_num = int(float(sys.argv[1]))

    # GET PREREQUISITES
    root = set_root()
    data = load_data(root)

    # RUN PREPARATION
    [name, one, deriv_path] = run_preparation(feat_num)

    # RUN TAYLOR SERIES ON DATA
    [ts_series, ts_resid, fd_estimate, sd_estimate, thd_estimate, fod_estimate, fid_estimate] = run_taylor_series(one)

    # FOURIER TRANSFORM
    fourier_transform_one = fourier_transform(one)

    # RUN TAYLOR SERIES ON FOURIER TRANSFORMED DATA
    [ts_series_ff, ts_resid_ff, fd_estimate_ff, sd_estimate_ff, thd_estimate_ff, fod_estimate_ff, fid_estimate_ff] \
        = run_taylor_series(fourier_transform_one)

    # PUT DERIVATIVES INTO DATAFRAMES
    derivatives_full = put_derivatives_into_dataframes(name, fd_estimate_ff, sd_estimate_ff, thd_estimate_ff,
                                                       fod_estimate_ff, fid_estimate_ff, fd_estimate, sd_estimate,
                                                       thd_estimate, fod_estimate, fid_estimate)
    derivatives_full = add_in_names(data, feat_num, derivatives_full)
    [derivatives_full, ts_series, ts_series_ff, ts_resid, ts_resid_ff] = rescale_output(one,
                                                                                        derivatives_full,
                                                                                        ts_series, ts_series_ff,
                                                                                        ts_resid,
                                                                                        ts_resid_ff)

    # OUTPUT TO CSV
    write_output(deriv_path, root, data, feat_num, derivatives_full, ts_series, ts_series_ff, ts_resid, ts_resid_ff)

    time_delta = datetime.now() - start_time
    exit_message = str(name) + ' MATRIX FD RAN SUCCESSFULLY IN: ' + str(time_delta)
    sys.exit(exit_message)


# RUN_ALL
run_all_matrix_fd()
