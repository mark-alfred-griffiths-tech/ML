#!/usr/bin/env python
from wav_kf_functions import *
from datetime import datetime


def run_all_wav_correct_kf() -> None:
    start_time = datetime.now()

    # GET NUMBER OF WAV FILE
    num_wav = int(float(sys.argv[1]))

    # IMPORT PREREQUISITES
    root = set_root()
    [deriv_path, file_directory] = output_folder(root)
    file_list = get_file_list(file_directory)

    # GET NAME
    name = str(file_list.iloc[num_wav][0])

    # OPEN RAW DATA
    data = run_preparation(num_wav)

    # FOURIER TRANSFORM
    data_fourier_transform = fourier_transform(data)

    # NORMALISATION STEP
    data_normalised = normalise(data)
    data_fourier_transform_normalised = normalise(data_fourier_transform)

    # RUN SCRIPT
    os.chdir(deriv_path)
    modality = "regular"
    run_kf_script(name, modality, data_normalised, deriv_path)
    modality = "fourier"
    run_kf_script(name, modality, data_fourier_transform_normalised, deriv_path)

    end_time = datetime.now()
    delta_time = end_time - start_time
    exit_message = str(name) + ' WAV KF PROCESSED IN: ' + str(delta_time)
    sys.exit(exit_message)


# RUN_ALL
run_all_wav_correct_kf()
