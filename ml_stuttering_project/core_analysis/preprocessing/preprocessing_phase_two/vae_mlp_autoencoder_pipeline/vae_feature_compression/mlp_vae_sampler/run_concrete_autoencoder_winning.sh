#!/bin/bash -l
#SBATCH -c 10
#SBATCH --mem=10G
#SBATCH --job-name=concrete_autoencoder
#SBATCH --time=0-48:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --output=/users/k1754828/LOG/ML/CONCRETE_AUTOENCODER/CONCRETE_AUTOENCODER_WINNING/concrete_autoencoder_winning-%j.output
#SBATCH --error=/users/k1754828/LOG/ML/CONCRETE_AUTOENCODER/CONCRETE_AUTOENCODER_WINNING/concrete_autoencoder_winning-%j.error

wd=/users/k1754828/SCRIPTS/PREPROCESSING_PHASE_TWO/CONCRETE_AUTOENCODER/CONCRETE_AUTOENCODER_WINNING


source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 main_reopen.py

