#!/bin/bash -l
#!/bin/bash -l
#SBATCH -c 10
#SBATCH --mem=10G
#SBATCH --job-name=load_shuffle
#SBATCH --time=0-01:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --output=$log/ml_stuttering_project/core_analysis/preprocessing/preprocessing_phase_two/vae_rnn_autoencoder_pipeline/vae_feature_compression/rnn_vae_param/run_variational_autoencoder.sh
#SBATCH --output=$log/ml_stuttering_project/core_analysis/preprocessing/preprocessing_phase_two/vae_rnn_autoencoder_pipeline/vae_feature_compression/rnn_vae_param/run_variational_autoencoder.sh

wd=$modellling/ml_stuttering_project/core_analysis/preprocessing/preprocessing_phase_two/vae_rnn_autoencoder_pipeline/vae_feature_compression/rnn_vae_param



source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 run_variational_autoencoder.py



