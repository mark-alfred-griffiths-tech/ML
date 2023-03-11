#!/bin/bash -l
#SBATCH --exclusive
#SBATCH -c 6
#SBATCH --mem=59000
#SBATCH --job-name=generate_log_reg_cf
#SBATCH --time=0-00:05:00
#SBATCH --partition=partition-1
#SBATCH --ntasks=1
#SBATCH --output=/home/markgreenneuroscience_gmail_com/LOG/TS_KF_TRANSFORMS/generate_wav_regressor/wav_regressors_cf/wav_fd_kf_cf-%j.output
#SBATCH --error=/home/markgreenneuroscience_gmail_com/LOG/TS_KF_TRANSFORMS/generate_wav_regressor/wav_regressors_cf/wav_fd_kf_cf-%j.error

wd=/home/markgreenneuroscience_gmail_com/TS_KF_TRANSFORMS/GENERATE_WAV_REGRESSORS/WAV_REGRESSORS_CF

source /etc/bash.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 GENERATE_WAV_REGRESSORS_CF.py
