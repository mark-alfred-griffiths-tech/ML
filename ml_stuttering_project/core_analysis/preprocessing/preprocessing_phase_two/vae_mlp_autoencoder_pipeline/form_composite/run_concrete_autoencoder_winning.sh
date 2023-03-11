#!/bin/bash -l
#SBATCH -c 10
#SBATCH --mem=10G
#SBATCH --job-name=vae_mlp_autoencoder_pipeline_form_composite
#SBATCH --time=0-48:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --output=log/core_analysis/preprocessing_phase_two/vae_mlp_autoencoder_pipeline/form_composite/form_composite-%j.output
#SBATCH --error=log/core_analysis/preprocessing_phase_two/vae_mlp_autoencoder_pipeline/form_composite/form_composite-%j.error

wd=$modelling/core_analysis/preprocessing_phase_two/vae_mlp_autoencoder_pipeline/form_composite


source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 main.py

