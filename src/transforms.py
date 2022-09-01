import torch
from pytorch3d.transforms import matrix_to_euler_angles, euler_angles_to_matrix
import numpy as np


def rotmat_pi_added(rotmat):
    eulers = matrix_to_euler_angles(rotmat, convention="ZYZ")
    eulers[:, 0] = eulers[:, 0] + np.pi
    rotmat_pi = euler_angles_to_matrix(eulers, convention="ZYZ")
    rotmat_cat=torch.cat([rotmat, rotmat_pi], 0)
    return rotmat_cat



def primal_to_fourier_2D(r):
    r = torch.fft.ifftshift(r, dim=(-2, -1))
    return torch.fft.fftshift(torch.fft.fftn(r, s=(r.shape[-2], r.shape[-1]), dim=(-2, -1)), dim=(-2, -1))

def fourier_to_primal_2D(f):
    f = torch.fft.ifftshift(f, dim=(-2, -1))
    return torch.fft.fftshift(torch.fft.ifftn(f, s=(f.shape[-2], f.shape[-1]), dim=(-2, -1)), dim=(-2, -1))



def primal_to_fourier_3D(r):
    r = torch.fft.ifftshift(r, dim=(-3, -2, -1))
    return torch.fft.fftshift(torch.fft.fftn(r, s=(r.shape[-3],r.shape[-2], r.shape[-1]), dim=(-3,-2, -1)), dim=(-3, -2, -1))

def fourier_to_primal_3D(f):
    f = torch.fft.ifftshift(f, dim=(-3, -2, -1))
    return torch.fft.fftshift(torch.fft.ifftn(f, s=(f.shape[-3],f.shape[-2], f.shape[-1]), dim=(-3, -2, -1)), dim=(-3, -2, -1))




def downsample_fourier_crop(proj, size):

    sidelen=proj.shape[-1]
    center=sidelen//2
    l_min=center-size//2
    l_max=center+size//2+size%2
    
    proj_ft=primal_to_fourier_2D(proj)
    proj_ft_crop=proj_ft[:,:,l_min:l_max,l_min:l_max]
    proj_crop_down=fourier_to_primal_2D(proj_ft_crop)
    
    factor=(float(sidelen)/size)**2
    return proj_crop_down.real/factor


def downsample_avgpool(proj, size):

     return torch.nn.AvgPool2d(kernel_size=proj.shape[-1]//size, stride=proj.shape[-1]//size)(proj)


def downsample_avgpool_3D(vol, size):
    vol=vol[None, None,:,:,:]
    vol_3d=torch.nn.AvgPool3d(kernel_size=vol.shape[-1]//size, stride=vol.shape[-1]//size)(vol)
    return vol_3d.squeeze()


def downsample_fourier_crop_3D(vol, size):
    
    sidelen=vol.shape[-1]
    center=sidelen//2
    l_min=center-size//2
    l_max=center+size//2+size%2
    
    vol_ft=primal_to_fourier_3D(vol)
    vol_ft_crop=vol_ft[l_min:l_max,l_min:l_max, l_min:l_max]
    vol_crop_down=fourier_to_primal_3D(vol_ft_crop)
    
    return vol_crop_down.real


def get_rot_mat_2D(n):
    theta = 2*np.pi*torch.rand(n,1,1)
    row_1=torch.cat([torch.cos(theta), -torch.sin(theta), 0*theta],-1)
    row_2=torch.cat([torch.sin(theta), torch.cos(theta), 0*theta],-1)
    rotmat=torch.cat([row_1,row_2], 1)
    return rotmat


def rot_img(x, rot_mat):
    
    grid = torch.nn.functional.affine_grid(rot_mat, x.size())
    x = torch.nn.functional.grid_sample(x, grid)
    return x

def random_rotate_2D(img):
    rotmat=get_rot_mat_2D(img.shape[0]).to(img.device)
    return rot_img(img, rotmat)
    
def SymCreatorD2(x):
    x=x.unsqueeze(0)
    xd1=x[:,:x.shape[1]//2,:x.shape[2]//2,:]
    xd2=torch.flip(xd1, [2,3])
 
    im1=torch.cat([xd1, xd2],2)
    im2=torch.flip(im1,[1,2])
    im=torch.cat([im1,im2], 1)
    return im.squeeze()

def conjugate_3d(vol):
        return torch.flip(vol, dims=(-1, -2, -3))

def vol_to_autocorr(vol):
    shape=vol.shape[-1]
    vol_corr_double = torch.zeros(1, 1, 2*shape, 2*shape, 2*shape).to(vol.device)
    vol_corr_double[0, 0, shape//2:shape//2+shape,shape//2:shape//2+shape,shape//2:shape//2+shape] = 1e4 * vol
    vol_corr_out = torch.nn.functional.conv3d(vol_corr_double, (vol)[None, None, :, :, :], padding="same")
    return vol_corr_out[0, 0, shape//2:shape//2+shape,shape//2:shape//2+shape,shape//2:shape//2+shape]

def half_so3(rotmat):
    eulers = matrix_to_euler_angles(rotmat, convention="ZYZ")
    eulers[:, 0] = eulers[:, 0]/2.0
    return  euler_angles_to_matrix(eulers, convention="ZYZ")
