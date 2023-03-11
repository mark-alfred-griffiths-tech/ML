#!/bin/bash -l
#SBATCH -c 10
#SBATCH --mem=10G
#SBATCH --job-name=secondary_concrete_vae_parametrised
#SBATCH --time=0-48:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --output=log/core_analysis/preprocessing_phase_two/secondary_concrete_feature_selection/secondary_concrete_vae_parametrised/secondary_concrete_vae_parametrised-%j.output
#SBATCH --error=log/core_analysis/preprocessing_phase_two/secondary_concrete_feature_selection/secondary_concrete_vae_parametrised/secondary_concrete_vae_parametrised-%j.error

wd=$modelling/core_analysis/preprocessing_phase_two/secondary_concrete_feature_selection/secondary_concrete_vae_parametrised/


source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 main.py

