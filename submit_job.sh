#!/bin/bash
#SBATCH --partition=ml
#SBATCH --job-name=cryonet-train
#SBATCH --output=./logs/output-%j.txt --error=./logs/output-%j.txt
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH -t 3:00:00
#SBATCH -n 32
#SBATCH --mem=131072

echo "betagal_real_supervised config file"

ROOT_PATH='/sdf/home/h/hgupta/ondemand/CryoPoseGAN/'
SCRIPT_PATH=${ROOT_PATH}

cd $SCRIPT_PATH 

singularity exec --nv /sdf/group/ml/CryoNet/singularity_images/cryonettorch-atomic-primal_latest.sif  \
python main_supervised_cryoposegan.py \
--config_path $1 \
--snr_val $2 \
--symmetrized_loss_tomo $3 \
--symmetrized_loss_sup $4 \
--progressive_supervision $5 \
--suffix_name $6 \
--ewald_radius $7

