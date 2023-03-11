#!/bin/bash -l
#SBATCH -c 10
#SBATCH --mem=10G
#SBATCH --job-name=rnn_2_hyperband
#SBATCH --time=0-48:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --output=$log/modelling/rnn_branch/rnn_hyperbanding/rnn_3_hyperbanding/rnn_3_hyperband-%j.output
#SBATCH --error=$log/modelling/rnn_branch/rnn_hyperbanding/rnn_3_hyperbanding/rnn_3_hyperband-%j.error

wd=$modelling/modelling/rnn_branch/rnn_hyperbanding/rnn_3_hyperbanding


source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 main.py

