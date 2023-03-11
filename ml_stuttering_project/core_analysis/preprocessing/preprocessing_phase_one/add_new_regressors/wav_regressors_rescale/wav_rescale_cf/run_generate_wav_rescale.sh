#!/bin/bash -l
#SBATCH --exclusive
#SBATCH -c 6
#SBATCH --mem=59000
#SBATCH --job-name=generate_wav_rescale_cf
#SBATCH --time=0-01:00:00
#SBATCH --partition=partition-1
#SBATCH --ntasks=1
#SBATCH --output=/home/markgreenneuroscience_gmail_com/LOG/TS_KF_TRANSFORMS/add_new_regressors/wav_regressors_rescale/wav_rescale_cf/generate_wav_rescale_cf-%j.output
#SBATCH --error=/home/markgreenneuroscience_gmail_com/LOG/TS_KF_TRANSFORMS/add_new_regressors/wav_regressors_rescale/wav_rescale_cf/generate_wav_rescale_cf-%j.error

wd=/home/markgreenneuroscience_gmail_com/TS_KF_TRANSFORMS/ADD_NEW_REGRESSORS/WAV_REGRESSORS_RESCALE/WAV_RESCALE_CF

source /etc/bash.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 GENERATE_WAV_FD_TS_KF_RESCALE_CF.py
