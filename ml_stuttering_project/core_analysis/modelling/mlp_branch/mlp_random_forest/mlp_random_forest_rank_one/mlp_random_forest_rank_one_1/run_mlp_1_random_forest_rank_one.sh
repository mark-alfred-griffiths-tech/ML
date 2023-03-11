#!/bin/bash -l
#SBATCH -c 10
#SBATCH --mem=10G
#SBATCH --job-name=mlp_1_random_forest_rank_one
#SBATCH --time=0-48:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --output=$log/core_analysis/modelling/mlp_branch/mlp_random_forest_rank_one/mlp_1_random_forest_rank_one/mlp_1_random_forest_rank_one-%j.output
#SBATCH --error=$log/core_analysis/modelling/mlp_branch/mlp_random_forest_rank_one/mlp_1_random_forest_rank_one/mlp_1_random_forest_rank_one-%j.error

wd=$modelling/core_analysis/modelling/mlp_branch/mlp_random_forest_default/mlp_1_random_forest_rank_one


source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 main.py

