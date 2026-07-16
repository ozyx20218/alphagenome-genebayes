#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --job-name=alphagenome_extract
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/nfs/roberts/project/pi_hz27/ox8/extraction_%j.out
#SBATCH --error=/nfs/roberts/project/pi_hz27/ox8/extraction_%j.err

module load miniconda
source activate alphagenome311

export CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt
export NO_GCE_CHECK=true
export TF_CPP_MIN_LOG_LEVEL=3
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

python -u /nfs/roberts/project/pi_hz27/ox8/extract_alphagenome_bouchet.py
