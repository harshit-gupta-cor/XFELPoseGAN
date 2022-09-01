#!/bin/bash
#SBATCH --partition=cryoem
#SBATCH --job-name=cryonet-train
#SBATCH --output=output-%j.txt --error=output-%j.txt
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH -t 14:00:00
#SBATCH -n 32
#SBATCH --mem=131072

ROOT_PATH='/sdf/home/h/hgupta/ondemand/CryoPoseGAN/'
SCRIPT_PATH=${ROOT_PATH}

cd $SCRIPT_PATH 
singularity exec --nv /sdf/group/ml/CryoNet/singularity_images/cryonettorch-atomic-primal_latest.sif python main_cryoposegan.py --config_path /sdf/home/h/hgupta/ondemand/CryoPoseGAN/configs/configs/splice_xfel_simulated.yaml 
