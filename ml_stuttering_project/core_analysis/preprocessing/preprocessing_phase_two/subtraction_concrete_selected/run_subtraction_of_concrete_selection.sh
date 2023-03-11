#!/bin/bash -l
#SBATCH -c 10
#SBATCH --mem=10G
#SBATCH --job-name=subtraction_concrete_selected
#SBATCH --time=0-48:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --output=$log/core_analysis/preprocessing_phase_two/subtraction_concrete_selected/subtraction_concrete_selected-%j.output
#SBATCH --error=$log/core_analysis/preprocessing_phase_two/subtraction_concrete_selected/subtraction_concrete_selected-%j.error

wd=$modelling/core_analysis/preprocessing_phase_two/subtraction_concrete_selected/


source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 main.py

