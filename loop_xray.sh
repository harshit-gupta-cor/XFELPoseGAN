#!/bin/bash

config_path1=configs/xfel_splice_autoencoder_simulated_poses_supervised_multires.yaml
config_path2=configs/xfel_splice_simulated_poses_supervised_multires.yaml

for symmetrized_loss_tomo in 1 ;
do
  for symmetrized_loss_sup in  1  ;
  do
     for progressive_supervision in 0  ;
     do
        for snr in  20;
        do
          for suffix_name in   {0..1};
          do
            for ewald_radius in {1 2 5 10 1000};
            do
              sbatch ./submit_job.sh $config_path2 \
               $snr \
              $symmetrized_loss_tomo \
               $symmetrized_loss_sup \
               $progressive_supervision \
               $suffix_name \
               $ewald_radius




            done
          done

        done
     done
  done
done

