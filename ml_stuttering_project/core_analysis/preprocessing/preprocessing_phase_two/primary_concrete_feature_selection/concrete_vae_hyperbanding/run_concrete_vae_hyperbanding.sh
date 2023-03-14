#!/bin/bash -l
#SBATCH -c 10
#SBATCH --mem=10G
#SBATCH --job-name=secondary_concrete_vae_hyperbanding
#SBATCH --time=0-48:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --output=log/core_analysis/preprocessing_phase_two/primary_concrete_feature_selection/primary_concrete_vae_hyperbanding/primary_concrete_vae_hyperbanding-%j.output
#SBATCH --error=log/core_analysis/preprocessing_phase_two/primary_concrete_feature_selection/primary_concrete_vae_hyperbanding/primary_concrete_vae_hyperbanding-%j.error



source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 main.py

