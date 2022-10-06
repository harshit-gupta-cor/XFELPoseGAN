# XFELPoseGAN

The jupyter notebook xfel_colab.ipynb contains commands to run the program on google colab.

This projects run XFEL pose GAN. The algorithms runs a multiresolution method to reconstruct the 3D structure of protein from its XFEL image.
It uses the config file in yaml format for the input parameters.

First download 3D structure of splice protein from https://drive.google.com/file/d/1FTEGj-6mNRpC4O3j5GFJNPHPJbScf8jB/view into ./figs folder. If the figs folder is not there, then create it.

The method can be run using

python main.py config_path configs/xfel_splice_simulated_poses_supervised_multires.yaml

There are two ways to install dependecies:
1. Use the singularity image /sdf/group/ml/CryoNet/singularity_images/cryonettorch-atomic-primal_latest.sif or
2. Install conda envoronment using

   conda env create --file environment.yml
   
   conda activate xfel 


The tensorboard log files are saved in the folder /logs/. This tensorboard log files can be accessed by
1. singularity exec -B /sdf --nv /sdf/group/ml/CryoNet/singularity_images/cryonettorch-atomic-primal_latest.sif tensorboard --logdir="path_to_logs_folder"
2. or by simply launching tensorboard if you have access to it.

The main file first generates data (if the data already doesn't exist for this configuration).
It then launches the wrapper object called SupervisedXFELwrapper in wrapper.py which runs the algorithm.

First it launches a GAN based reconstruction for a specified number of iteration, followed by training an encoder using the fake data from previous step and 
then finally reconstructing the protein using the poses assigned by deploying the encoder on real data.

The autoencoder based version can be run using 

python main.py config_path configs/xfel_splice_autoencoder_simulated_poses_supervised_multires.yaml
