#!/bin/bash -l
#SBATCH --exclusive
#SBATCH -c 6
#SBATCH --mem=59000
#SBATCH --job-name=wav_rescale_kf_ts
#SBATCH --time=0-03:00:00
#SBATCH --partition=partition-1
#SBATCH --ntasks=1
#SBATCH --output=/home/markgreenneuroscience_gmail_com/LOG/TS_KF_TRANSFORMS/add_new_regressors/wav_regressors_rescale/wav_rescale_run/wav_rescale_kf_ts-%j.output
#SBATCH --error=/home/markgreenneuroscience_gmail_com/LOG/TS_KF_TRANSFORMS/add_new_regressors/wav_regressors_rescale/wav_rescale_run/wav_rescale_kf_ts-%j.error

wd=/home/markgreenneuroscience_gmail_com/TS_KF_TRANSFORMS/ADD_NEW_REGRESSORS/WAV_REGRESSORS_RESCALE/RUN_WAV_RESCALE
control_file=/home/markgreenneuroscience_gmail_com/TS_KF_TRANSFORMS/ADD_NEW_REGRESSORS/WAV_REGRESSORS_RESCALE/WAV_RESCALE_CF
sge_index=${control_file}/WAV_RESCALE_KF_TS_CONTROL_FILE.txt

wav_name=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${sge_index} | awk '{print $1}')
length_of_wav_file_in_matrix=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${sge_index} | awk '{print $3}')
string_to_search=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${sge_index} | awk '{print $4}')
wav_folder=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${sge_index} | awk '{print $5}')

source /etc/bash.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 WAV_RESCALE.py "$wav_name" "$length_of_wav_file_in_matrix" "$string_to_search" "$wav_folder"
