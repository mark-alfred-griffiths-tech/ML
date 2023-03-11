#!/bin/bash -l
#SBATCH -c 10
#SBATCH --mem=10G
#SBATCH --job-name=load_shuffle
#SBATCH --time=0-01:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --output=$log/ml_stuttering_project/core_analysis/preprocessing/shuffle_and_split/split/run_split/load_shuffle-%j.output
#SBATCH --error=$log/ml_stuttering_project/core_analysis/preprocessing/shuffle_and_split/split/run_split/load_shuffle-%j.error

wd=$modellling/ml_stuttering_project/core_analysis/preprocessing/shuffle_and_split/split/run_split


source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 split.py

