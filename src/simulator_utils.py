"""contains simulator and its components classes."""

import os
import mrcfile
import torch
from ctf_utils import CTF
from noise_utils import Noise
from shift_utils import Shift
from transforms import primal_to_fourier_2D, fourier_to_primal_2D, primal_to_fourier_3D, fourier_to_primal_3D
import sys
sys.path.insert(0, "/sdf/home/h/hgupta/ondemand/XFELPoseGAN/")
from ml_modules import Unet
import numpy as np
from src.projector_utils import ProjectorFactory
def init_cube(sidelen):
    """Create a volume with cube.

    Parameters
    ----------
    sidelen: int
        size of the volume

    Returns
    -------
    volume: torch.Tensor
        Tensor (sidelen,sidelen,sidelen) with a cube
        of size (sidelen//2,sidelen//2,sidelen//2)
    """
    L = sidelen // 2
    length = sidelen // 8
    volume = torch.zeros([sidelen] * 3)
    volume[
        L - length : L + length, L - length : L + length, L - length : L + length
    ] = 1
    return volume



"""Module to generate data using using liner forward model."""


class LinearSimulator(torch.nn.Module):
    """Class to generate data using liner forward model.
    Parameters
    ----------
    config: class
        Class containing parameters of the simulator
    """

    def __init__(self, config, mode="projector"):
        super(LinearSimulator, self).__init__()

        self.projector = ProjectorFactory.get_projector(config, mode)
        # class for tomographic projector
        #self.init_volume()  # changes the volume inside the projector
        if config.ctf or config.shift:
            
            if "betagal" in config.protein:
                config.pixel_size=0.637*(384.0/config.side_len)
            config.ctf_size=config.side_len
            self.ctf = CTF(config)  # class for ctf
            self.shift = Shift(config)  # class for shifts
        self.noise = Noise(config)  # class for noise
        self.proj_scalar= torch.nn.Parameter(torch.Tensor([0]))
        self.scalar=torch.nn.Parameter(torch.Tensor([0]))
        self.config = config

       

    def forward(self, rotmat, ctf_params, shift_params, noise_params):
        """Create cryoEM measurements using input parameters.
        Parameters
        ----------
        rot_params: dict of type str to {tensor}
            Dictionary of rotation parameters for a projection chunk
        ctf_params: dict of type str to {tensor}
            Dictionary of Contrast Transfer Function (CTF) parameters
             for a projection chunk
        shift_params: dict of type str to {tensor}
            Dictionary of shift parameters for a projection chunk
        Returns
        -------
        projection.real : tensor
            Tensor ([chunks,1,sidelen,sidelen]) contains cryoEM measurement
        """
    
        
        proj={}
        proj.update({"rotmat": rotmat,
              "ctf_params": ctf_params,
              "shift_params": shift_params,
              "noise_params": noise_params})
        
        projection_tomo = self.projector(rotmat)
        
        if ctf_params is not None or shift_params is not None:
            f_projection_tomo = primal_to_fourier_2D(projection_tomo)
            f_projection_ctf = self.ctf(f_projection_tomo, ctf_params)
            f_projection_shift = self.shift(f_projection_ctf, shift_params)
            projection_clean = fourier_to_primal_2D(f_projection_shift).real.float()
            projection_ctf=fourier_to_primal_2D(f_projection_ctf).real
            proj.update({"ctf": projection_ctf})
            
        else:
            projection_clean=projection_tomo
            
        projection_clean=torch.exp(self.proj_scalar[0]) *projection_clean


        if "xray_autocorr" in self.config.exp_name:
            projection=torch.exp(self.scalar[0])*projection_clean
        else:
            projection = torch.exp(self.scalar[0])*self.noise(projection_clean, noise_params)

        projection_clean = torch.exp(self.scalar[0]) * projection_clean
        
        proj.update({"tomo":projection_tomo, 
                     "clean":projection_clean,
                     "proj":projection
                    })
        return proj

    def init_volume(self):
        """Initialize the volume inside the projector.
        Initializes the mrcfile whose path is given in config.input_volume_path.
        If the path is not given or doesn't exist then the volume
        is initialized with a cube.
        """
        if (
            self.config.input_volume_path == ""
            or not os.path.exists(os.path.join(os.getcwd(), self.config.input_volume_path))
           
        ):

            print(
                "No input volume specified or the path doesn't exist. "
                "Using cube as the default volume."
            )
            volume = init_cube(self.config.side_len)
        else:
            with mrcfile.open(self.config.input_volume_path, "r") as m:
                volume = torch.from_numpy(m.data.copy()).to(self.projector.vol.device)

        self.projector.vol = volume


    def save_snr_configuration(self, fake_params):

        self.save_snr_dict={}
        for keys, val in fake_params.items():
            self.save_snr_dict.update({keys:val})
        self.save_snr_dict.update({"snr_"+str(int(self.projector.scale)):self.noise.current_snr,
                           "scale": int(self.projector.scale)})

        print([(keys, val) for keys, val in self.save_snr_dict.items()])

    def adjust_snr(self):
        #snr to adjust to
        #scale at which the adjustment should happen
        fake_params=self.save_snr_dict
        print([(keys, val) for keys, val in fake_params.items()])
        with torch.no_grad():
            print(f"Current scale is {self.projector.scale}")
            ctf_params = fake_params["ctf_params"] if "ctf_params" in fake_params else None
            shift_params = fake_params["shift_params"] if "shift_params" in fake_params else None
            noise_params = fake_params["noise_params"] if "noise_params" in fake_params else None


            out_dict=self.forward(fake_params["rotmat"],
                                      ctf_params,
                                      shift_params,
                                      noise_params)

            from transforms import downsample_avgpool

            scale=self.save_snr_dict["scale"]
            clean_down=downsample_avgpool(out_dict["clean"], size=2**scale)
            noisy_down=downsample_avgpool(out_dict["proj"], size=2**scale)
            snr_down=self.noise.snr_calculator( clean_down, noisy_down)
            snr_old=self.save_snr_dict["snr_"+str(scale)]
            factor=np.log(10.0) * (snr_old-snr_down) / 20.0
            self.proj_scalar.data[0]= factor+self.proj_scalar.data[0]

            print(f"adjusted  SNR at scale {scale} to {snr_old} from {snr_down}")



            ctf_params = fake_params["ctf_params"] if "ctf_params" in fake_params else None
            shift_params = fake_params["shift_params"] if "shift_params" in fake_params else None
            noise_params = fake_params["noise_params"] if "noise_params" in fake_params else None


            out_dict=self.forward(fake_params["rotmat"],
                                      ctf_params,
                                      shift_params,
                                      noise_params)

            clean_down = downsample_avgpool(out_dict["clean"], size=2 ** scale)
            noisy_down = downsample_avgpool(out_dict["proj"], size=2 ** scale)
            snr_down = self.noise.snr_calculator(clean_down, noisy_down)

            print(f"New SNR at scale {scale} is indeed {snr_down}")





        

        
        


class DenoiseSimulator(torch.nn.Module):
    """Class to generate data using liner forward model.
    Parameters
    ----------
    config: class
        Class containing parameters of the simulator
    """

    def __init__(self, config, mode="projector"):
        super(DenoiseSimulator, self).__init__()

        
        self.projector = Unet(in_channels=1, out_channels=1,config=config)  # class for tomographic projector
 
        #self.init_volume()  # changes the volume inside the projector
        if config.ctf or config.shift:
            
            if "betagal" in config.protein:
                config.pixel_size=0.637*(384.0/config.side_len)
            config.ctf_size=config.side_len
            self.ctf = CTF(config)  # class for ctf
            self.shift = Shift(config)  # class for shifts
        self.noise = Noise(config)  # class for noise
        self.proj_scalar= torch.nn.Parameter(torch.Tensor([0]))
        self.scalar=torch.nn.Parameter(torch.Tensor([0]))
        self.config = config
        print(self.projector)
        
                
        self.projector.vol = torch.zeros([self.config.side_len] * 3, dtype=torch.float32)
        self.projector.vol=torch.nn.Parameter(self.projector.vol)
       

    def forward(self, rotmat, ctf_params, shift_params, noise_params):
        """Create cryoEM measurements using input parameters.
        Parameters
        ----------
        rot_params: dict of type str to {tensor}
            Dictionary of rotation parameters for a projection chunk
        ctf_params: dict of type str to {tensor}
            Dictionary of Contrast Transfer Function (CTF) parameters
             for a projection chunk
        shift_params: dict of type str to {tensor}
            Dictionary of shift parameters for a projection chunk
        Returns
        -------
        projection.real : tensor
            Tensor ([chunks,1,sidelen,sidelen]) contains cryoEM measurement
        """
  
        proj={}
        proj.update({"rotmat": rotmat,
              "ctf_params": ctf_params,
              "shift_params": shift_params,
              "noise_params": noise_params})
        
        projection_tomo = self.projector(noise_params["real_proj"])
        
        if (ctf_params is not None or shift_params is not None) and (self.config.ctf or self.config.shift):
    
            f_projection_tomo = primal_to_fourier_2D(projection_tomo)
            f_projection_ctf = self.ctf(f_projection_tomo, ctf_params)
            f_projection_shift = self.shift(f_projection_ctf, shift_params)
            projection_clean = fourier_to_primal_2D(f_projection_shift).real  
            projection_ctf=fourier_to_primal_2D(f_projection_ctf).real
            proj.update({"ctf": projection_ctf})
            
        else:
            projection_clean=projection_tomo
            
        projection_clean=torch.exp(self.proj_scalar[0]) *projection_clean
        
        projection = torch.exp(self.scalar[0])*self.noise(projection_clean, noise_params)
        
        proj.update({"tomo":projection_tomo, 
                     "clean":projection_clean,
                     "proj":projection
                    })
        return proj

    def init_volume(self):
        """Initialize the volume inside the projector.
        Initializes the mrcfile whose path is given in config.input_volume_path.
        If the path is not given or doesn't exist then the volume
        is initialized with a cube.
        """
        if (
            self.config.input_volume_path == ""
            or not os.path.exists(os.path.join(os.getcwd(), self.config.input_volume_path))
           
        ):

            print(
                "No input volume specified or the path doesn't exist. "
                "Using cube as the default volume."
            )
            volume = init_cube(self.config.side_len)
        else:
            with mrcfile.open(self.config.input_volume_path, "r") as m:
                volume = torch.from_numpy(m.data.copy()).to(self.projector.vol.device)

        self.projector.vol = volume
        
    
        
        
        

class XFELSimulator(torch.nn.Module):
    """Class to generate data using XFEL forward model.
    Parameters
    ----------
    config: class
        Class containing parameters of the simulator
    """

    def __init__(self, config, mode="projector"):
        super(XFELSimulator, self).__init__()

        
        self.projector = XFELProjector(config, mode)  # class for tomographic projector
        #self.init_volume()  # changes the volume inside the projector
        if config.shift:
            self.shift = Shift(config)  # class for shifts
        self.proj_scalar= torch.nn.Parameter(torch.Tensor([0]))
        self.config = config
        self.noise = Noise(config)

        #self.init_volume()
        
        
        
    def forward(self, rotmat, ctf_params, shift_params, noise_params):
        """Create cryoEM measurements using input parameters.
        Parameters
        ----------
        rot_params: dict of type str to {tensor}
            Dictionary of rotation parameters for a projection chunk
        shift_params: dict of type str to {tensor}
            Dictionary of shift parameters for a projection chunk
        Returns
        -------
        projection.real : tensor
            Tensor ([chunks,1,sidelen,sidelen]) contains cryoEM measurement
        """
        proj={}
        proj.update({"rotmat": rotmat})
        
        projection_tomo = self.projector(rotmat)
       
        f_projection_tomo = primal_to_fourier_2D(projection_tomo)
        
        if self.config.shift:
            f_projection_clean = self.shift(f_projection_tomo, shift_params)
        else:
            f_projection_clean= f_projection_tomo
            
        
     
        output_dict={}
        projection=f_projection_clean
        output_dict.update({"tomo":projection_tomo, 
                     "f_tomo": f_projection_tomo,
                     "f_clean":f_projection_clean,
                     "proj": projection_tomo
                    })
        return output_dict

    def init_volume(self):
        """Initialize the volume inside the projector.
        Initializes the mrcfile whose path is given in config.input_volume_path.
        If the path is not given or doesn't exist then the volume
        is initialized with a cube.
        """
        if not hasattr(self.config, "input_volume_path") or self.config.input_volume_path == "":

            print(
                "No input volume specified or the path doesn't exist. "
                "Using cube as the default volume."
            )
            volume = init_cube(self.config.side_len)
        else:
            with mrcfile.open(self.config.input_volume_path, "r") as m:
                volume = torch.from_numpy(m.data.copy()).to(self.projector.vol.device)

        self.projector.vol.data[:,:,:] = volume[:,:,:]