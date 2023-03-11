#!/bin/bash -l
#SBATCH -c 10
#SBATCH --mem=10GB
#SBATCH --job-name=log_reg
#SBATCH --partition=partition-1
#SBATCH --nodes=1
#SBATCH --output=/users/k1754828/LOG/ML/VAE_TWO/RUN_VAE_TWO/run_variational_autoencoder_two-%j.output
#SBATCH --error=/users/k1754828/LOG/ML/VAE_TWO/RUN_VAE_TWO/run_variational_autoencoder_two-%j.error

wd=/users/k1754828/SCRIPTS/VAE_TWO/VAE_TWO_RUN

control_file_dir=/users/k1754828/SCRIPTS/VAE_TWO/VAE_TWO_CF
vae_two_control_file=${control_file_dir}/vae_two_cf.txt
intermediate_dim=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${vae_two_control_file} | awk '{print $1}')
latent_z_feat_num=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${vae_two_control_file} | awk '{print $2}')

source /etc/bash.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710
cd $wd || exit
python3 run_variational_autoencoder.py "$intermediate_dim" "$latent_z_feat_num"





