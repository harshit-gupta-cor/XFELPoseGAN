
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import mrcfile
sys.path.insert(0, "/sdf/home/h/hgupta/ondemand/CryoPoseGAN/")
from utils import to_numpy
from plot3d_utils import generate_rotmats_video
from loss_utils import pairwise_cos_sim
from transforms import downsample_avgpool, downsample_avgpool_3D
from pytorch3d.transforms import matrix_to_euler_angles
from src.fsc_utils import calc_fsc, fsc2res, align_volumes, rotate_volume
import time
''' Some utils '''


def normalize_proj(proj):
    """Written by J.N. Martel."""
    # assumes proj is N,C,H,W
    num_projs = proj.shape[0]

    vmin, _ = torch.min(proj.reshape(num_projs, -1), dim=-1)
    vmin = vmin[:, None, None, None]
    vmax, _ = torch.max(proj.reshape(num_projs, -1), dim=-1)
    vmax = vmax[:, None, None, None]
    proj_norm = (proj - vmin) / (vmax - vmin)

    return proj_norm


def visualize_rot(rotmat_real, rotmat_rec, writer, global_step, summary_prefix):
    """Written by J.N. Martel."""
    color_real = torch.tensor([[1, 0, 0]])  # red
    color_rec = torch.tensor([[0, 0, 1]])  # blue

    all_colors = torch.cat((color_real.repeat(rotmat_real.shape[0], 1),
                            color_rec.repeat(rotmat_real.shape[0], 1)), dim=0)

    video_rots, err = generate_rotmats_video(rotmat_real, rotmat_rec,
                                        all_colors)
    writer.add_video(f"{summary_prefix}: Rotations", video_rots[None, ...], global_step=global_step, fps=25)
    return err


def generate_mesh_from_volume(volume, name):
    """Written by J.N. Martel."""
    vertices, triangles = mcubes.marching_cubes(volume, 1e-2)
    mcubes.export_obj(vertices, triangles, f"{name}.obj")


''' The summary function '''

def write_summary(real, fake, rec, volume_dict, writer, total_steps,
                   summary_prefix='', compute_fsc=False, multi_res_factor=1, computation_time=None):

    volume = volume_dict["rec"]

    print(volume.shape)
    """Written by J.N. Martel."""
    # Output the training set reciction and real projs

    # above, but there could be more.

    ''' Plot non normalized figure with colorbar '''
    if real is not None and "proj" in real:
        fig = plt.figure(dpi=96)

        plt.imshow(to_numpy(real['proj'][0, ...].squeeze()), cmap='plasma')
        plt.colorbar()
        plt.tight_layout()
        writer.add_figure(f"{summary_prefix}/Proj Images/real (colorbar)", fig, global_step=total_steps)

    if fake is not None and "proj" in fake:
        fig = plt.figure(dpi=96)
        plt.imshow(to_numpy(fake['proj'][0, ...].squeeze()), cmap='plasma')
        plt.colorbar()
        plt.tight_layout()
        writer.add_figure(f"{summary_prefix}/Proj Images/fake (colorbar)", fig, global_step=total_steps)

    if rec is not None and "proj" in rec:
        fig = plt.figure(dpi=96)
        plt.imshow(to_numpy(rec['proj'][0, ...].squeeze()), cmap='plasma')
        plt.colorbar()
        plt.tight_layout()
        writer.add_figure(f"{summary_prefix}/Proj Images/rec (colorbar)", fig, global_step=total_steps)



    ''' Plot the rotations '''
    rotmat_real = real['rotmat'] if real is not None and "rotmat" in real else None
    rotmat_rec = rec['rotmat']  if rec is not None and "rotmat" in rec else None
    rotmat_fake = fake['rotmat']  if fake is not None and "rotmat" in fake else None
    
    if rotmat_rec is not None:
        rot_err = visualize_rot(rotmat_real, rotmat_rec, writer, total_steps, summary_prefix)
        writer.add_scalar(f'{summary_prefix}: rec_rot_mae (degree)', rot_err.mean(), global_step=total_steps)

        ''' Visualize a latent code similarity '''
        latent_rec = rec['rotmat'].flatten(1)
        if latent_rec is not None:
            fig = plt.figure(dpi=96)
            plt.imshow(to_numpy(pairwise_cos_sim(latent_rec)), cmap='plasma', vmin=0, vmax=1,
                       interpolation='nearest')
            plt.colorbar()
            plt.tight_layout()
            writer.add_figure(f"{summary_prefix}: latent_cos_sim", fig, global_step=total_steps)

  
    
    center=volume.shape[0]//2
    
    vol_2d=to_numpy(normalize_proj(torch.stack([volume[center,:,:].squeeze(),
                                                volume[:,center, :].squeeze(), 
                                                volume[:, :, center].squeeze() 
                                               ], 
                                                   0)[:, None, :,:]))
    writer.add_video(f"{summary_prefix}: ortho slices", vol_2d[:, None, :,:, :], global_step=total_steps)



    if compute_fsc and "gt" in volume_dict and volume.sum()>1e-8:

        volume_gt = volume_dict["gt"]
        if volume_gt.shape[-1] > volume.shape[-1]:
            volume_gt=downsample_avgpool_3D(volume_gt, size=volume.shape[-1])

        tic = time.time()

        opt_q, volume_aligned = align_volumes(volume, volume_gt)
        print("Orientation search: " + str(time.time() - tic))
        volume_gt = volume_gt.numpy()
        volume = volume.numpy()
        fsc = calc_fsc(volume_aligned, volume_gt, side=volume.shape[0])
        resn, x, y, resx = fsc2res(fsc, return_plot=True)
        print("Total time for FSC computation: " + str(time.time() - tic))
        fig = plt.figure(dpi=96)
        plt.plot(fsc[:, 0], fsc[:, 0] * 0 + 0.5, 'k--')
        plt.plot(fsc[:, 0], fsc[:, 0] * 0 + 0.143, 'k--')
        plt.plot(fsc[:, 0], fsc[:, 1], 'o')
        plt.plot(x, y, 'k-')
        plt.plot([resx], [0.5], 'ro')
        plt.xlabel('Resolution (1/$\mathrm{\AA}$)')
        plt.ylabel('Fourier Shell Correlation')
        writer.add_figure(f"Fourier Shell Correlation", fig, global_step=total_steps)
        writer.add_scalar("Resolution", resn, total_steps)
        writer.add_scalar("Absolute Resolution", resn * multi_res_factor, total_steps)
        if computation_time:
            writer.add_scalar("Resolution_vs_time", resn, computation_time)
            writer.add_scalar("Absolute Resolution_vs_time", resn * multi_res_factor, computation_time)



    

def write_euler_histogram(writer, total_steps, rotmat_true, rotmat_pred, name, align=False):
   
        
    euler_true=np.degrees(to_numpy(matrix_to_euler_angles(rotmat_true, convention="ZYZ")))
    euler_pred=np.degrees(to_numpy(matrix_to_euler_angles(rotmat_pred, convention="ZYZ")))
    
    val_rot_true, bins_rot=np.histogram(euler_true[:,0], bins=36, range=(-180, 180))
    val_tilt_true,bins_tilt=np.histogram(euler_true[:,1], bins=36, range=(0, 180))
    val_psi_true, bins_psi=np.histogram(euler_true[:,2], bins=36, range=(-180, 180))
    
    val_rot_fake, _=np.histogram(euler_pred[:,0], bins=36, range=(-180, 180))
    val_tilt_fake,_=np.histogram(euler_pred[:,1], bins=36, range=(0, 180))
    val_psi_fake, _=np.histogram(euler_pred[:,2], bins=36, range=(-180, 180))
    

    batch_size=euler_true.shape[0]
    fig = plt.figure(9, figsize=(12, 12))
    plt.subplot(331)
    plt.scatter(euler_true[:,0], euler_pred[:,0], c=np.linspace(0,4,batch_size ))
    plt.xlim(left=-180, right=180 )
    plt.ylim(bottom=-180, top=180 )
    plt.title("Rot")
    
    plt.subplot(332)
    plt.scatter(euler_true[:,1], euler_pred[:,1], c=np.linspace(0,4,batch_size ))
    plt.title("Tilt")
    plt.xlim(left=0, right=180 )
    plt.ylim(bottom=0, top=180 )
        
    plt.subplot(333)
    plt.scatter(euler_true[:,2], euler_pred[:,2],  c=np.linspace(0,4,batch_size ))
    plt.title("Psi")
    plt.xlim(left=-180, right=180 )
    plt.ylim(bottom=-180, top=180 )
    
    plt.subplot(334)
    plt.bar(bins_rot[1:], val_rot_true, width=30)
    plt.title("True Rot")
    plt.subplot(335)
    plt.bar(bins_tilt[1:], val_tilt_true, width=30)
    plt.title("True Tilt")
    plt.subplot(336)
    plt.bar(bins_psi[1:], val_psi_true, width=30)
    plt.title("True Psi")
      
    plt.subplot(337)
    plt.bar(bins_rot[1:], val_rot_fake, width=30)
    plt.title("Pred Rot")
    plt.subplot(338)
    plt.bar(bins_tilt[1:], val_tilt_fake, width=30)
    plt.title("Pred Tilt")
    plt.subplot(339)
    plt.bar(bins_psi[1:], val_psi_fake, width=30)
    plt.title("Pred Psi")
    
    
    #plt.tight_layout()
    
    writer.add_figure(f"{name}: GT vs Pred Poses", fig, global_step=total_steps)

