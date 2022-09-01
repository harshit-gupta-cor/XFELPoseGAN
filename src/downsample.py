from transforms import fourier_to_primal_2D, primal_to_fourier_2D, fourier_to_primal_3D, primal_to_fourier_3D

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

