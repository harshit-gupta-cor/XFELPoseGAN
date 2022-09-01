import os
import matplotlib.backends.backend_agg as plt_backend_agg
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import torch
import pytorch3d.transforms
from scipy import ndimage
def render_to_rgb(figure):
    canvas = plt_backend_agg.FigureCanvasAgg(figure)
    canvas.draw()
    data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w, h = figure.canvas.get_width_height()
    image_hwc = data.reshape([h, w, 4])
    image_chw = np.moveaxis(image_hwc, source=2, destination=0)
    return image_chw

def to_numpy(x):
    return x.data.cpu().numpy()



def align_rotmat(rotmat_gt, rotmat_pred):
    from scipy.spatial.transform import Rotation as R

    ''' align the two set of vectors'''
    num_rots_gt = rotmat_gt.shape[0]
    unitvec_gt = torch.tensor([[0, 0, 1]], dtype=torch.float32).to(rotmat_gt.device)
    rot_unitvecs_gt = torch.bmm(unitvec_gt.repeat(num_rots_gt, 1, 1), rotmat_gt)

    num_rots_pred = rotmat_pred.shape[0]
    unitvec_pred = torch.tensor([[0, 0, 1]], dtype=torch.float32).to(rotmat_pred.device)
    rot_unitvecs_pred = torch.bmm(unitvec_pred.repeat(num_rots_pred, 1, 1), rotmat_pred)

    rot_unitvecs_gt = to_numpy(rot_unitvecs_gt.squeeze())
    rot_unitvecs_pred = to_numpy(rot_unitvecs_pred.squeeze())
    relativeR, rmsd = R.align_vectors(rot_unitvecs_gt, rot_unitvecs_pred)
    rot_unitvecs_pred = relativeR.apply(rot_unitvecs_pred)
    diff = np.degrees(np.arccos([np.dot(x[0], x[1]) for x in zip(rot_unitvecs_gt, rot_unitvecs_pred)]))

    R_matrix = torch.tensor(relativeR.as_matrix(), dtype=torch.float32).to(rotmat_pred.device)
    # "right-wise multiplication convention" in the equations above
    rotmat_pred_aligned = torch.bmm(rotmat_pred, R_matrix.repeat(num_rots_pred, 1, 1).permute(0, 2, 1))
    
    return rotmat_pred_aligned
    
    
    
def generate_rotmats_video(rotmat_gt, rotmat_pred, colors):
    from matplotlib import pyplot as plt
    from scipy.spatial.transform import Rotation as R

    ''' align the two set of vectors'''
    num_rots_gt = rotmat_gt.shape[0]
    unitvec_gt = torch.tensor([[0, 0, 1]], dtype=torch.float32).to(rotmat_gt.device)
    rot_unitvecs_gt = torch.bmm(unitvec_gt.repeat(num_rots_gt, 1, 1), rotmat_gt)

    num_rots_pred = rotmat_pred.shape[0]
    unitvec_pred = torch.tensor([[0, 0, 1]], dtype=torch.float32).to(rotmat_pred.device)
    rot_unitvecs_pred = torch.bmm(unitvec_pred.repeat(num_rots_pred, 1, 1), rotmat_pred)

    rot_unitvecs_gt = to_numpy(rot_unitvecs_gt.squeeze())
    rot_unitvecs_pred = to_numpy(rot_unitvecs_pred.squeeze())
    relativeR, rmsd = R.align_vectors(rot_unitvecs_gt, rot_unitvecs_pred)
    rot_unitvecs_pred = relativeR.apply(rot_unitvecs_pred)
    diff = np.degrees(np.arccos([np.dot(x[0], x[1]) for x in zip(rot_unitvecs_gt, rot_unitvecs_pred)]))

    R_matrix = torch.tensor(relativeR.as_matrix(), dtype=torch.float32).to(rotmat_pred.device)
    # "right-wise multiplication convention" in the equations above
    rotmat_pred_aligned = torch.bmm(rotmat_pred, R_matrix.repeat(num_rots_pred, 1, 1).permute(0, 2, 1))
    se_frob = np.linalg.norm(to_numpy(rotmat_gt - rotmat_pred_aligned), axis=(1, 2)) ** 2
    mse_frob = np.mean(se_frob)
    median_frob = np.median(se_frob)
    # try to compensate for chiral configuration
    chir = torch.tensor(np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]), dtype=torch.float32).to(rotmat_gt.device)
    se_frob_chir = np.linalg.norm(to_numpy(torch.bmm(
        chir.repeat(num_rots_gt, 1, 1), rotmat_gt) - rotmat_pred_aligned), axis=(1, 2)) ** 2
    mse_frob_chir = np.mean(se_frob_chir)
    median_frob_chir = np.median(se_frob_chir)
    frob = [mse_frob, median_frob, mse_frob_chir, median_frob_chir]

    rot_unitvecs = np.concatenate([rot_unitvecs_gt,rot_unitvecs_pred],axis=0)

    # generate latitutde/longitude lines along unit sphere
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))

    # plot coordinates and unit sphere mesh
    fig_imgs = []
    def plot_scatter3d(ax,pts,colors,elev,azim):
        ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1, alpha=0.2)
        ax.scatter(pts[:, 0],
                   pts[:, 1],
                   pts[:, 2],
                   s=10, c=colors.detach().numpy(), zorder=10)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.view_init(elev=elev, azim=azim)
        plt.axis('off')
        plt.tight_layout()
        return ax

    for azim in range(60,120):
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})

        plot_scatter3d(ax,rot_unitvecs,colors,
                            elev=30.,azim=azim)

        # Draw lines for first part of data
        pts1 = rot_unitvecs_gt
        pts2 = rot_unitvecs_pred
        segments = [(pts1[i,:], pts2[i,:]) for i in range(pts1.shape[0])]
        lc = Line3DCollection(segments)
        ax.add_collection3d(lc)

        fig_imgs.append(torch.from_numpy(render_to_rgb(fig))[None,...])
        plt.close(fig)

    for azim in reversed(range(60,120)):
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})

        plot_scatter3d(ax,rot_unitvecs,colors,
                            elev=30.,azim=azim)

        fig_imgs.append(torch.from_numpy(render_to_rgb(fig))[None,...])
        plt.close(fig)

    video = torch.cat(fig_imgs,dim=0)
    return video, rmsd, relativeR, diff, frob
