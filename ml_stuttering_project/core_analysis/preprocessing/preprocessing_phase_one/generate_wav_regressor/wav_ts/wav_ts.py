#!/usr/bin/env python
from wav_ts_functions import *

from datetime import datetime


def run_all_wav_correct_fd() -> None:
    start_time = datetime.now()

    # GET WAV NUM
    wav_num = int(float(sys.argv[1]))

    # IMPORT PREREQUISITES
    root = set_root()
    [deriv_path, file_directory] = output_folder(root)
    file_list = get_file_list(file_directory)

    # GET NAME
    name = str(file_list.iloc[wav_num][0])

    # OPEN RAW DATA

    data = run_preparation(wav_num)

    # FOURIER TRANSFORM
    data_fourier_transform = fourier_transform(data)

    # NORMALISATION STEP
    data_normalised = normalise(data)
    data_fourier_transform_normalised = normalise(data_fourier_transform)

    # RUN TAYLOR SERIES ON DATA
    [ts_resid, ts_series, fd_estimate, sd_estimate, thd_estimate, fod_estimate, fid_estimate] = \
        run_taylor_series(data_normalised)

    # RUN TAYLOR SERIES ON FREQUENCY FORM OF DATA
    [ts_resid_ff, ts_series_ff, fd_estimate_ff, sd_estimate_ff, thd_estimate_ff, fod_estimate_ff,
     fid_estimate_ff] = run_taylor_series(data_fourier_transform_normalised)

    # GET DERIVATIVES
    derivatives_full = put_derivatives_into_dataframes(fd_estimate_ff, sd_estimate_ff, thd_estimate_ff, fod_estimate_ff,
                                                       fid_estimate_ff, fd_estimate, sd_estimate, thd_estimate,
                                                       fod_estimate, fid_estimate)

    # GET TS_CONCAT
    ts_concat = put_taylor_series_and_residuals_into_dataframes(ts_series, ts_series_ff, ts_resid, ts_resid_ff)

    # RESCALE
    [derivatives_full, ts_concat] = rescale_output(data, derivatives_full, ts_concat)

    # CREATE OUTPUT NAMES
    derivatives_name = name + '_derivatives.csv'
    ts_name = name + '_taylor_series.csv'

    # OUTPUT TO CSV
    os.chdir(deriv_path)

    derivatives_full_dataframe = pd.DataFrame(derivatives_full)
    ts_concat = pd.DataFrame(ts_concat)

    derivatives_full_dataframe.columns = ['fd_estimate', 'sd_estimate', 'thd_estimate', 'fod_estimate', 'fid_estimate',
                                          'fd_estimate_ff', 'sd_estimate_ff', 'thd_estimate_ff', 'fod_estimate_ff',
                                          'fid_estimate_ff']

    ts_concat.columns = ['ts_series', 'ts_series_ff', 'ts_resid', 'ts_resid_ff']

    derivatives_full_dataframe.to_csv(derivatives_name, index=False)
    ts_concat.to_csv(ts_name, index=False)

    time_delta = datetime.now() - start_time
    exit_message: str = str(name) + ' WAV FD PROCESSED IN: ' + str(time_delta)
    sys.exit(exit_message)


# RUN_ALL
run_all_wav_correct_fd()
