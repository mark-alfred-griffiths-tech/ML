#!/bin/bash -l
#SBATCH -c 10
#SBATCH --mem=10G
#SBATCH --job-name=rnn_random_forest_better_default_3
#SBATCH --time=0-48:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --output=$log/ml_stuttering_project/core_analysis/modelling/rnn_branch/rnn_random_forest/rnn_random_forest_better_default_3/rnn_random_forest_better_default_3-%j.output
#SBATCH --error=$log/ml_stuttering_project/core_analysis/modelling/rnn_branch/rnn_random_forest/rnn_random_forest_better_default_3/rnn_random_forest_better_default_3-%j.error

wd=$modelling/ml_stuttering_project/core_analysis/modelling/rnn_branch/rnn_random_forest/rnn_random_forest_better_default/rnn_random_forest_better_default_3



source ~/.bashrc
eval "$(conda  shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 main.py

