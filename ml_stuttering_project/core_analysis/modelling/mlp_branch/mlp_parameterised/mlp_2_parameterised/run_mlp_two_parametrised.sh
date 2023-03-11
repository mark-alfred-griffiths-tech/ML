#!/bin/bash -l
#SBATCH -c 10
#SBATCH --mem=10G
#SBATCH --job-name=mlp_2_parametrised
#SBATCH --time=0-48:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --output=$log/core_analysis/modelling/mlp_branch/mlp_2_parametrised/mlp_2_parametrised-%j.output
#SBATCH --error=$log/core_analysis/modelling/mlp_branch/mlp_2_parametrised/mlp_2_parametrised-%j.error

wd=$modelling/core_analysis/modelling/mlp_branch/mlp_parametrised/mlp_2_parametrised


source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 main.py

