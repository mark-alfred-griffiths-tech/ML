#!/bin/bash -l
#SBATCH --exclusive
#SBATCH -c 6
#SBATCH --mem=59000
#SBATCH --job-name=correct_matrix
#SBATCH --time=0-00:15:00
#SBATCH --partition=partition-1
#SBATCH --ntasks=1
#SBATCH --output=/home/markgreenneuroscience_gmail_com/LOG/TS_KF_TRANSFORMS/add_new_regressors/correct_matrix/correct_matrix-%j.output
#SBATCH --error=/home/markgreenneuroscience_gmail_com/LOG/TS_KF_TRANSFORMS/add_new_regressors/correct_matrix/correct_matrix-%j.error

wd=/home/markgreenneuroscience_gmail_com/TS_KF_TRANSFORMS/ADD_NEW_REGRESSORS/CORRECT_MATRIX

source /etc/bash.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 CORRECT_MATRIX.py
