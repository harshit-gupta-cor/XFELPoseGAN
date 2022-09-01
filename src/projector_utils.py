"""Class to generate tomographic projection."""

import torch
from pytorch3d.transforms import Rotate
import numpy as np
#from transforms import random_rotate_2D, SymCreatorD2
from abc import ABCMeta, abstractstaticmethod

from src.transforms import fourier_to_primal_2D
class ProjectorFactory:
        """Class to instantiate projector from a factory of choices."""

        def get_projector(config, mode):
            print(config.multires)
            if config.multires:
                # if "xfel-autocorr" in config.exp_name:
                #     projector = XFELMultiRes(config, mode="autocorr")
                #
                # elif "xfel-intensity" in config.exp_name:
                #     projector = XFELMultiRes(config)
                # else:
                projector = ProjectorMultiRes(config)

            else:
                projector = Projector(config, mode)
            return projector





class EwaldProjector(torch.nn.Module):
    """Class to generate tomographic projection.
    Written by J.N. Martel, Y. S. G. Nashed, and Harshit Gupta.
    Parameters
    ----------
    config: class
        Class containing parameters of the Projector
    """

    def __init__(self, config, mode="projector"):
        """Initialize volume grid."""
        super(EwaldProjector, self).__init__()

        self.config = config
        self.vol = torch.zeros([self.config.side_len] * 3, dtype=torch.float32)
        self.vol=torch.nn.Parameter(self.vol)
        lin_coords = torch.linspace(-1.0, 1.0, self.config.side_len)
        [x, y] = torch.meshgrid(
            [
                lin_coords,
            ]
            * 2
        )
        r2=(x**2+y**2)
        z=config.ewald_radius-(config.ewald_radius**2-r2)**0.5
        coords = torch.stack([y, x, z], dim=-1)
        self.register_buffer("vol_coords", coords.reshape(-1, 3))
        self.mode=mode
        print(f"projector mode is {self.mode}")

    def make_vol(self):
        if self.config.protein=="betagal":
                self.vol.data=SymCreatorD2(self.vol.data)

        return  self.vol.data

    def constraint(self):
        if self.config.positivity:
            self.vol.data.clamp_(min=0)

        if self.config.mask_3d:
            self.vol.data[self.mask_3d] = 0

    def forward(self, rotmat, proj_axis=-1):
        """Output the tomographic projection of the volume.
        First rotate the volume and then sum it along an axis.
        The volume is assumed to be cube. The output image
        follows (batch x channel x height x width) convention of pytorch.
        Therefore, a dummy channel dimension is added at the end to projection.
        Parameters
        ----------
        rot_params: dict of type str to {tensor}
            Dictionary containing parameters for rotation, with keys
                rotmat: str map to tensor
                    rotation matrix (batch_size x 3 x 3) to rotate the volume
        proj_axis: int
            index along which summation is done of the rotated volume
        Returns
        -------
        projection: tensor
            Tensor containing tomographic projection
            (batch_size x 1 x sidelen x sidelen)
        """


        batch_sz = rotmat.shape[0]
        t = Rotate(rotmat, device=self.vol_coords.device)
        rot_vol_coords = t.transform_points(self.vol_coords.repeat(batch_sz, 1, 1))
        rot_vol_coords=rot_vol_coords.reshape(1,-1,3)
        vol=self.vol


        rot_vol = torch.nn.functional.grid_sample(
            vol.repeat(1, 1, 1, 1, 1),
            rot_vol_coords[:, None, None, :, :],
            align_corners=True,
        )
        projection = rot_vol.reshape(
                batch_sz,1,
                self.config.side_len,
                self.config.side_len,
            )

        projection=fourier_to_primal_2D(projection).real
        print(projection.norm())
        return projection


class Projector(torch.nn.Module):
    """Class to generate tomographic projection.
    Written by J.N. Martel, Y. S. G. Nashed, and Harshit Gupta.
    Parameters
    ----------
    config: class
        Class containing parameters of the Projector
    """

    def __init__(self, config, mode="projector"):
        """Initialize volume grid."""
        super(Projector, self).__init__()

        self.config = config
        self.vol = torch.zeros([self.config.side_len] * 3, dtype=torch.float32)
        self.vol=torch.nn.Parameter(self.vol)
        lin_coords = torch.linspace(-1.0, 1.0, self.config.side_len)
        [x, y, z] = torch.meshgrid(
            [
                lin_coords,
            ]
            * 3
        )
        coords = torch.stack([y, x, z], dim=-1)
        self.register_buffer("vol_coords", coords.reshape(-1, 3))
        self.mode=mode
        print(f"projector mode is {self.mode}")

        if self.config.mask_3d:
            length = self.vol.shape[-1]
            ax = torch.linspace(-length / 2.0, length / 2.0, length)
            diameter = self.config.mask_3d_diameter
            xx, yy, zz = torch.meshgrid(ax, ax, ax)
            if "new_idea" in self.config.exp_name:
                mask_3d = (zz ** 2 + yy ** 2).sqrt() > length / 2.0 * diameter

            else:
               mask_3d = (xx ** 2 + yy ** 2 + zz ** 2).sqrt() > length / 2.0 * diameter
            self.register_buffer("mask_3d",mask_3d)
        
    def make_vol(self):
        if self.config.protein=="betagal":
                self.vol.data=SymCreatorD2(self.vol.data)
      
        return  self.vol.data

    def constraint(self):
        if self.config.positivity:
            self.vol.data.clamp_(min=0)

        if self.config.mask_3d:
            self.vol.data[self.mask_3d] = 0

    def forward(self, rotmat, proj_axis=-1):
        """Output the tomographic projection of the volume.
        First rotate the volume and then sum it along an axis.
        The volume is assumed to be cube. The output image
        follows (batch x channel x height x width) convention of pytorch.
        Therefore, a dummy channel dimension is added at the end to projection.
        Parameters
        ----------
        rot_params: dict of type str to {tensor}
            Dictionary containing parameters for rotation, with keys
                rotmat: str map to tensor
                    rotation matrix (batch_size x 3 x 3) to rotate the volume
        proj_axis: int
            index along which summation is done of the rotated volume
        Returns
        -------
        projection: tensor
            Tensor containing tomographic projection
            (batch_size x 1 x sidelen x sidelen)
        """
        if self.mode=="new_idea":        
            indices=np.random.randint(self.config.side_len, size=self.config.batch_size)
            projection=self.vol[indices].unsqueeze(1)
         
            projection=random_rotate_2D(projection)
            
            
        else:
            
   
            batch_sz = rotmat.shape[0]
            t = Rotate(rotmat, device=self.vol_coords.device)
            rot_vol_coords = t.transform_points(self.vol_coords.repeat(batch_sz, 1, 1))
            
            
            if self.config.protein=="betagal":
               vol=SymCreatorD2(self.vol)
            else:
                vol=self.vol
                
                
            rot_vol = torch.nn.functional.grid_sample(
                vol.repeat(batch_sz, 1, 1, 1, 1),
                rot_vol_coords[:, None, None, :, :],
                align_corners=True,
            )
            projection = torch.sum(
                rot_vol.reshape(
                    batch_sz,
                    self.config.side_len,
                    self.config.side_len,
                    self.config.side_len,
                ),
                dim=proj_axis,
            )
            projection = projection[:, None, :, :]
       


        return projection


class ProjectorMultiRes(torch.nn.Module):
    """Class to generate tomographic projection.
    Written by Harshit Gupta.
    Parameters
    ----------
    config: class
        Class containing parameters of the Projector
    """

    def __init__(self, config):
        """Initialize volume grid."""
        super(ProjectorMultiRes, self).__init__()

        self.config = config
        self.scale = int(np.log2(config.side_len))
        full_scale=self.scale
        self.vol_dict={}
        for scale in range(5, full_scale+1):

            vol=torch.zeros([2 ** (scale)] * 3, dtype=torch.float32)
            length=vol.shape[-1]

            lin_coords = torch.linspace(-1.0, 1.0, length)
            [x, y, z] = torch.meshgrid([lin_coords,]* 3)
            self.register_buffer("coords_"+str(scale),torch.stack([y, x, z], dim=-1).reshape(-1, 3))


            if self.config.mask_3d:
                ax=torch.linspace(-length/2.0,length/2.0,length )
                diameter=self.config.mask_3d_diameter
                xx,yy,zz=torch.meshgrid(ax, ax,ax)

                self.register_buffer("mask_3d_"+str(scale), (xx**2+yy**2+zz**2).sqrt()>length/2.0*diameter)



            self.vol_dict.update({"vol_"+str(scale):torch.nn.Parameter(vol)})
        self.vol_dict=torch.nn.ParameterDict(self.vol_dict)

    def conjugate_3d(self, vol):
        return torch.flip(vol, dims=(0,1,2)).roll((1,1,1), dims=(2,1,0))

    def fill_volume(self, scale_1, scale_2):
        "save volume at scale 1 in scale 2"

        with torch.no_grad():
            vol_1=self.vol_dict["vol_"+str(int(scale_1))]
            if scale_1<scale_2:
                vol_2=torch.nn.functional.interpolate(vol_1[None,None,:,:,:],scale_factor=2**(scale_2-scale_1), mode="trilinear" ).squeeze()

            else:
                vol_2=torch.nn.AvgPool3d(kernel_size=2**(scale_1-scale_2))(vol_1[None,None,:,:,:]).squeeze()
            vol_2=vol_2/2**((scale_2-scale_1)) # (vol_1.sum()/(vol_2.sum()+1e-8))*vol_2
            print(f"Upscaling in projector from scale {scale_1}: sum {vol_1.sum().item()} to scale {scale_2}: sum {vol_2.sum().item()}")
            self.vol_dict["vol_"+str(int(scale_2))].data=vol_2

    def make_vol(self):
        self.vol=self.vol_dict["vol_"+str(int(self.scale))].data
        if self.config.protein == "betagal":
            self.vol.data = SymCreatorD2(self.vol)

        return self.vol

    def constraint(self):
        if self.config.positivity:
            self.vol_dict["vol_"+str(int(self.scale))].data.clamp_(min=0)

        if self.config.mask_3d:

            self.vol_dict["vol_"+str(int(self.scale))].data[self.__getattr__("mask_3d_"+str(int(self.scale)))]=0



    def forward(self, rotmat, proj_axis=-1):


        vol=self.vol_dict["vol_"+str(int(self.scale))]
        # if "autocorr" in self.config.exp_name:
        #     vol=0.5*self.conjugate_3d(vol)+0.5*vol
        vol_coords=self.__getattr__("coords_"+str(int(self.scale)))
        batch_sz = rotmat.shape[0]
        t = Rotate(rotmat, device=vol_coords.device)
        rot_vol_coords = t.transform_points(vol_coords.repeat(batch_sz, 1, 1))

        if self.config.protein == "betagal":
            vol = SymCreatorD2(vol)


        rot_vol = torch.nn.functional.grid_sample(
            vol.repeat(batch_sz, 1, 1, 1, 1),
            rot_vol_coords[:, None, None, :, :],
            align_corners=True,
        )
        projection = torch.sum(
            rot_vol.reshape(
                batch_sz,
                vol.shape[-1],
                vol.shape[-1],
                vol.shape[-1],
            ),
            dim=proj_axis,
        )
        projection = projection[:, None, :, :]

        return projection

    
'''
class XFELProjector(torch.nn.Module):
    """Class to generate tomographic projection.
    Written by Harshit Gupta.
    Parameters
    ----------
    config: class
        Class containing parameters of the Projector
    """

    def __init__(self, config, mode="projection"):
        """Initialize volume grid."""
        super(XFELProjector, self).__init__()

        self.config = config
        self.vol = torch.zeros([self.config.side_len] * 3, dtype=torch.float32)
        self.vol=torch.nn.Parameter(self.vol)
        lin_coords = torch.linspace(-1.0, 1.0, self.config.side_len)
        [x, y, z] = torch.meshgrid(
            [
                lin_coords,
            ]
            * 3
        )
        coords = torch.stack([y, x, z], dim=-1)
        self.register_buffer("vol_coords", coords.reshape(-1, 3))
        self.mode=mode
        print(f"projector module is {self.mode}")

    def forward(self, rotmat, proj_axis=-1):
        """Output the tomographic projection of the volume.
        First rotate the volume and then sum it along an axis.
        The volume is assumed to be cube. The output image
        follows (batch x channel x height x width) convention of pytorch.
        Therefore, a dummy channel dimension is added at the end to projection.
        Parameters
        ----------
        rot_params: dict of type str to {tensor}
            Dictionary containing parameters for rotation, with keys
                rotmat: str map to tensor
                    rotation matrix (batch_size x 3 x 3) to rotate the volume
        proj_axis: int
            index along which summation is done of the rotated volume
        Returns
        -------
        projection: tensor
            Tensor containing tomographic projection
            (batch_size x 1 x sidelen x sidelen)
        """
        self.vol.data=0.5*torch.flip(self.vol.data, dims=(0,1,2))+0.5*self.vol.data
        if self.mode=="projector":
   
            batch_sz = rotmat.shape[0]
            t = Rotate(rotmat, device=self.vol_coords.device)
            rot_vol_coords = t.transform_points(self.vol_coords.repeat(batch_sz, 1, 1))

            rot_vol = torch.nn.functional.grid_sample(
                self.vol.repeat(batch_sz, 1, 1, 1, 1),
                rot_vol_coords[:, None, None, :, :],
                align_corners=True,
            )
            rot_vol = rot_vol.reshape(
                    batch_sz,
                    self.config.side_len,
                    self.config.side_len,
                    self.config.side_len,
                )
            projection=rot_vol[:,self.config.side_len//2,:,:].squeeze()
       
            projection =( projection[:, None, :, :]).log()
        elif self.mode=="new_idea":
            indices=np.random.randint(self.config.side_len, size=self.config.batch_size)
            projection=self.vol[indices].unsqueeze(1)
        return projection
        
'''

class XFELMultiRes(torch.nn.Module):
    """Class to generate XFEL projection.
    Written by Harshit Gupta.
    Parameters
    ----------
    config: class
        Class containing parameters of the Projector
    """

    def __init__(self, config, mode=None):
        """Initialize volume grid."""
        super(XFELMultiRes, self).__init__()

        self.config = config
        self.scale = int(np.log2(config.side_len))
        self.mode=mode
        full_scale=self.scale
        self.vol_dict={}
        for scale in range(5, full_scale+1):

            vol=torch.zeros([2 ** (scale)] * 3, dtype=torch.float32)
            length=vol.shape[-1]

            lin_coords = torch.linspace(-1.0, 1.0, length)
            [x, y, z] = torch.meshgrid([lin_coords, ] * 3)
            self.register_buffer("coords_" + str(scale), torch.stack([y, x, z], dim=-1).reshape(-1, 3))

            # lin_coords = torch.linspace(-1.0, 1.0, length)
            # lin_coords = lin_coords - lin_coords[length // 2]
            # [x, y, z] = torch.meshgrid([lin_coords,]* 3)
            self.register_buffer("coords_"+str(scale),torch.stack([y, x, z], dim=-1).reshape(-1, 3))


            if self.config.mask_3d:
                ax=torch.linspace(-length/2.0,length/2.0,length )
                diameter=self.config.mask_3d_diameter
                xx,yy,zz=torch.meshgrid(ax, ax,ax)

                self.register_buffer("mask_3d_"+str(scale), (xx**2+yy**2+zz**2).sqrt()>length/2.0*diameter)



            self.vol_dict.update({"vol_"+str(scale):torch.nn.Parameter(vol)})
        self.vol_dict=torch.nn.ParameterDict(self.vol_dict)

        if self.config.mask_2d:
            length = self.config.ProjectionSize
            ax = torch.linspace(-length / 2.0, length / 2.0, length)
            ax=ax-ax[length//2]
            diameter = self.config.mask_2d_diameter
            xx, yy = torch.meshgrid(ax, ax)
            self.register_buffer("mask_2d", (xx ** 2 + yy ** 2).sqrt() < length / 2.0 * diameter)

    def fill_volume(self, scale_1, scale_2):
        "save volume at scale 1 in scale 2"
        with torch.no_grad():
            vol_1=self.vol_dict["vol_"+str(int(scale_1))]
            if scale_1<scale_2:
                vol_2=torch.nn.functional.interpolate(vol_1[None,None,:,:,:],scale_factor=2**(scale_2-scale_1), mode="trilinear" ).squeeze()

            else:
                vol_2=torch.nn.AvgPool3d(kernel_size=2**(scale_1-scale_2))(vol_1[None,None,:,:,:]).squeeze()
            vol_2=vol_2/2**((scale_2-scale_1)) # (vol_1.sum()/(vol_2.sum()+1e-8))*vol_2
            print(f"Upscaling in projector from scale {scale_1}: sum {vol_1.sum().item()} to scale {scale_2}: sum {vol_2.sum().item()}")
            self.vol_dict["vol_"+str(int(scale_2))].data=vol_2

    def make_vol(self):
        self.vol=self.vol_dict["vol_"+str(int(self.scale))].data
        if self.config.protein == "betagal":
            self.vol.data = SymCreatorD2(self.vol)

        return self.vol

    def constraint(self):
        if self.config.positivity:
            self.vol_dict["vol_"+str(int(self.scale))].data.clamp_(min=0)

        if self.config.mask_3d:

            self.vol_dict["vol_"+str(int(self.scale))].data[self.__getattr__("mask_3d_"+str(int(self.scale)))]=0
        # vol_temp=self.vol_dict["vol_" + str(int(self.scale))].data
        # self.vol_dict["vol_" + str(int(self.scale))].data=0.5*self.conjugate_3d(vol_temp).data\
        #                                              +0.5*vol_temp

    def conjugate_3d(self, vol):
        return torch.flip(vol, dims=(0,1,2)).roll((1,1,1), dims=(2,1,0))

    def conjugate_2d(self, vol):
        return torch.flip(vol, dims=( -2, -1)).roll((1, 1), dims=(-1, -2))

    def forward(self, rotmat, proj_axis=-1):

            vol=self.vol_dict["vol_"+str(int(self.scale))]
            vol_coords=self.__getattr__("coords_"+str(int(self.scale)))
            batch_sz = rotmat.shape[0]
            t = Rotate(rotmat, device=vol_coords.device)
            rot_vol_coords = t.transform_points(vol_coords.repeat(batch_sz, 1, 1))

            if self.config.protein == "betagal":
                vol = SymCreatorD2(vol)


            rot_vol = torch.nn.functional.grid_sample(
                vol.repeat(batch_sz, 1, 1, 1, 1),
                rot_vol_coords[:, None, None, :, :],
                align_corners=True,
            )

            length=int(2**self.scale)
            rot_vol = rot_vol.reshape(
                batch_sz,
                self.config.side_len,
                self.config.side_len,
                self.config.side_len,
            )
            projection = rot_vol[:, length//2, :, :].squeeze()
            projection = projection[:, None, :, :]
            #projection=0.5*projection+0.5*self.conjugate_2d(projection)
            if self.config.mask_2d:
                projection= projection * self.mask_2d
            return projection
