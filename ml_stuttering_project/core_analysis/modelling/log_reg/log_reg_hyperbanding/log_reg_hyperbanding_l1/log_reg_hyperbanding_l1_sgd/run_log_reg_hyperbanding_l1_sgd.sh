#!/bin/bash -l
#SBATCH -c 10
#SBATCH --mem=10G
#SBATCH --job-name=l1_sgd
#SBATCH --time=0-48:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --output=$log/core_analysis/modelling/log_reg/log_reg_hyperbanding/log_reg_hyperbanding_l1/log_reg_hperbanding_l1_sgd/log_reg_hyperbanding_l1_sgd-%j.output
#SBATCH --error=$log/core_analysis/modelling/log_reg/log_reg_hyperbanding/log_reg_hyperbanding_l1/log_reg_hperbanding_l1_sgd/log_reg_hyperbanding_l1_sgd-%j.error

wd=$modelling/ocre_analysis/modelling/log_reg/log_reg_hyperbanding/log_reg_hyperbanding_l1/log_reg_hyperbanding_l1_sgd

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 main.py

