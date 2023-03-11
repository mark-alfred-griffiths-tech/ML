#!/bin/bash -l
#SBATCH --exclusive
#SBATCH -c 6
#SBATCH --mem=59000
#SBATCH --job-name=wav_fd
#SBATCH --time=0-02:00:00
#SBATCH --partition=partition-1
#SBATCH --ntasks=1
#SBATCH --output=/home/markgreenneuroscience_gmail_com/LOG/TS_KF_TRANSFORMS/generate_wav_regressor/wav_ts/wav_fd-%j.output
#SBATCH --error=/home/markgreenneuroscience_gmail_com/LOG/TS_KF_TRANSFORMS/generate_wav_regressor/wav_ts/wav_fd-%j.error

wd=/home/markgreenneuroscience_gmail_com/TS_KF_TRANSFORMS/GENERATE_WAV_REGRESSORS/WAV_FD

control_file_dir=/home/markgreenneuroscience_gmail_com/TS_KF_TRANSFORMS/GENERATE_WAV_REGRESSORS/WAV_REGRESSORS_CF
wav_control_file=${control_file_dir}/WAV_CONTROL_FILE.txt

wav_num=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${wav_control_file} | awk '{print $1}')

source /etc/bash.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 WAV_FD.py "$wav_num"
